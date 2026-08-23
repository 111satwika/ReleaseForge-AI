import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel

from app.agents.graph import build_graph, is_affirmative
from app.eval.metrics import evaluate_platform
from app.eval.rag_eval import judge_faithfulness
from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider
from app.mcp.factory import start_mcp_client, stop_mcp_client
from app.memory import repository as memory
from app.memory.db import init_db
from app.observability.tracing import get_tracer, record_exception, setup_tracing
from app.rag.factory import get_graph_store, get_vector_store
from app.rag.ingest import ingest_incident_record, ingest_project_docs, ingest_release_record
from app.rag.retriever import hybrid_retrieve
from app.security.audit import recent_audit_events

logger = logging.getLogger(__name__)

NEGATIVE = {"no", "n", "nah", "stop", "cancel", "don't", "dont", "nevermind", "never mind", "decline", "rejected", "reject"}

CHECKPOINT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoints.db"
CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _looks_like_new_instruction(text: str) -> bool:
    """A decline reply that isn't affirmative *or* a plain "no" is probably not a
    rejection at all — it's the user moving on to a different request mid-approval."""
    normalized = text.strip().lower()
    if not normalized or normalized in NEGATIVE:
        return False
    return len(normalized.split()) >= 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    # First, before anything else gets a chance to make an LLM call or an MCP call that
    # should be traced — a TracerProvider configured after spans have already started
    # would just silently drop them.
    setup_tracing()

    await init_db()

    # Spawns the filesystem/docker/kubernetes MCP servers as subprocesses and discovers
    # their tools once, here — not lazily on first use mid-request, since starting three
    # subprocesses and doing an MCP handshake with each isn't something that should
    # happen accidentally inside a request's critical path.
    await start_mcp_client()

    # SQLite-backed, not in-memory — a paused approval (or an entire conversation)
    # now survives a backend restart. MemorySaver lost everything the moment the
    # process died, which is exactly what happened to a real in-progress approval.
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        app.state.graph = build_graph(checkpointer)

        # Best-effort: this needs Ollama running with an embedding model pulled
        # (RELEASEFORGE_EMBEDDING_MODEL, default nomic-embed-text). A backend without
        # that available should still start and serve chat/tasks normally — RAG is an
        # enhancement to chat, not a dependency of the release pipeline.
        try:
            indexed = await ingest_project_docs(get_llm_provider(), get_vector_store())
            if indexed:
                logger.info("Indexed %d doc chunks into the RAG vector store", indexed)
        except Exception:
            logger.warning("Skipping doc ingestion — embedding call failed (is Ollama running?)", exc_info=True)

        try:
            yield
        finally:
            await stop_mcp_client()


app = FastAPI(title="ReleaseForge AI Core", lifespan=lifespan)
# Automatic request/response spans for every route — the design doc's "workflow
# duration" tracking starts at the HTTP boundary, not just inside the agent graph.
FastAPIInstrumentor.instrument_app(app)


class PingRequest(BaseModel):
    workspace_path: str
    message: str = "Say hello in one short sentence."


class FaithfulnessCheckRequest(BaseModel):
    question: str
    answer: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/eval/summary")
async def eval_summary(workspace_path: str | None = None):
    """Design doc Section 16's "evaluate the complete platform," made real: agent
    completion rate, coding (test pass / review acceptance) rate, and deployment
    success/rollback rate, computed from actual ReleaseRecord history rather than
    simulated numbers. Every rate is `null`, not a fake 0 or 100%, until there's
    genuinely at least one record to compute it from — see app/eval/metrics.py."""
    report = await evaluate_platform(workspace_path)
    return {
        "records_evaluated": report["records_evaluated"],
        "agent": asdict(report["agent"]),
        "coding": asdict(report["coding"]),
        "deployment": asdict(report["deployment"]),
    }


@app.post("/eval/faithfulness")
async def eval_faithfulness(req: FaithfulnessCheckRequest):
    """Design doc Section 16's faithfulness metric, the one rag_eval.py originally
    didn't attempt — closed after a real, live-observed retrieval-was-right-but-the-
    answer-still-drifted case (see the Phase 6 build-plan addendum). Re-retrieves the
    same passages chat_reply() would have used for this question, then asks the LLM to
    check the given answer against them — so this checks a real answer against real
    retrieval, not a fabricated pair."""
    llm = get_llm_provider()
    hits = await hybrid_retrieve(req.question, llm.embed, get_vector_store(), k=4)
    passages = [h["text"] for h in hits]
    result = await judge_faithfulness(llm, req.question, req.answer, passages)
    return {
        "faithful": result.faithful,
        "unsupported_claims": result.unsupported_claims,
        "explanation": result.explanation,
        "passages_checked": len(passages),
    }


@app.get("/audit/log")
async def audit_log(limit: int = 100, workspace_path: str | None = None):
    """Design doc Section 14's "audit logs" requirement — every MCP tool call this
    backend has attempted, successful or not, including scope-denied attempts (an agent
    role reaching for a tool outside its least-privilege scope). See
    app/security/scopes.py and app/mcp/client.py for where each row gets written."""
    events = await recent_audit_events(limit=limit, workspace_path=workspace_path)
    return {
        "events": [
            {
                "id": e.id,
                "role": e.role,
                "tool_name": e.tool_name,
                "server_name": e.server_name,
                "workspace_path": e.workspace_path,
                "status": e.status,
                "detail": e.detail,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ]
    }


@app.get("/incidents")
async def incidents(workspace_path: str, limit: int = 20):
    """Closes the "incident memory" gap flagged since Phase 5: every rollback's real
    root-cause analysis (Phase 4's Recovery agent), actually queryable now instead of
    existing only once, in the chat reply that first showed it."""
    records = await memory.recent_incidents(workspace_path, limit=limit)
    return {
        "incidents": [
            {
                "id": r.id,
                "workspace_path": r.workspace_path,
                "release_id": r.release_id,
                "deployment_strategy": r.deployment_strategy,
                "action": r.action,
                "root_cause": r.root_cause,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    }


@app.post("/ping")
async def ping(req: PingRequest):
    llm = get_llm_provider()
    response = await llm.chat([LLMMessage(role="user", content=req.message)])
    return {
        "workspace_path": req.workspace_path,
        "llm_reply": response.content,
        "model": response.model,
    }


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    workspace_path: str
    messages: list[ChatTurn]


async def _maybe_record_release(graph, config: dict, workspace_path: str) -> None:
    """Called after a graph run that did NOT pause for approval — i.e. it actually
    reached an end. Persists one row to release memory if a task (not plain chat)
    produced an outcome, whatever that outcome was — declined attempts count too."""
    final_state = await graph.aget_state(config)
    if final_state.next:
        return  # still paused on an approval gate, nothing to record yet

    values = final_state.values
    outcome = values.get("outcome")
    if not outcome:
        return  # plain chat turn, or nothing ever got planned

    record = await memory.record_release(
        workspace_path=workspace_path,
        target_file=values.get("target_file") or None,
        plan=values.get("plan") or None,
        change_description=values.get("change_description") or None,
        test_passed=values.get("test_passed"),
        review_status=values.get("review_status") or None,
        review_risk=values.get("review_risk") or None,
        readiness_score=values.get("readiness_score") or None,
        readiness_recommendation=values.get("readiness_recommendation") or None,
        branch_name=values.get("branch_name") or None,
        pr_url=values.get("pr_url") or None,
        deployment_strategy=values.get("deployment_strategy") or None,
        outcome=outcome,
    )

    # Best-effort, same reasoning as the doc ingest in lifespan(): a release is recorded
    # (and the response already sent) whether or not RAG indexing succeeds.
    try:
        await ingest_release_record(get_llm_provider(), get_vector_store(), get_graph_store(), record)
    except Exception:
        logger.warning("Skipping release-record ingestion — embedding call failed", exc_info=True)

    # A rollback gate resolving is the one place Phase 4's Recovery agent's root-cause
    # analysis exists at all — persist it here or it's gone the moment this response is
    # sent, exactly the gap flagged and left open since Phase 5.
    if outcome in ("rolled_back", "declined_at_rollback"):
        incident = await memory.record_incident(
            workspace_path=workspace_path,
            release_id=record.id,
            deployment_strategy=values.get("deployment_strategy") or None,
            action="rolled_back" if outcome == "rolled_back" else "left_as_is",
            root_cause=values.get("recovery_analysis") or "(no root-cause analysis captured)",
            deploy_log=values.get("deploy_log") or None,
        )
        try:
            await ingest_incident_record(get_llm_provider(), get_vector_store(), get_graph_store(), incident)
        except Exception:
            logger.warning("Skipping incident-record ingestion — embedding call failed", exc_info=True)


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    tracer = get_tracer()
    with tracer.start_as_current_span("workflow.chat") as span:
        span.set_attribute("workspace_path", req.workspace_path)
        try:
            graph = request.app.state.graph
            # One active run per repo for now — real per-conversation sessions are a Phase 5.5+ concern.
            config = {"configurable": {"thread_id": req.workspace_path}}
            last_message = req.messages[-1].content if req.messages else ""

            pending = await graph.aget_state(config)
            span.set_attribute("workflow.resumed", bool(pending.next))
            redirected = False
            if pending.next:
                result = await graph.ainvoke(Command(resume=last_message), config=config)
                if not is_affirmative(last_message) and _looks_like_new_instruction(last_message):
                    # That wasn't really a decline — it's a new request. The resume above already
                    # safely stopped the pending change (never assume "yes" from ambiguous text);
                    # now start the new one for real instead of just discarding what they said.
                    redirected = True
                    result = await graph.ainvoke(
                        {"workspace_path": req.workspace_path, "user_message": last_message},
                        config=config,
                    )
            else:
                result = await graph.ainvoke(
                    {"workspace_path": req.workspace_path, "user_message": last_message},
                    config=config,
                )
            span.set_attribute("workflow.redirected", redirected)

            diff = None
            awaiting_approval = "__interrupt__" in result
            if awaiting_approval:
                # Each await_*_approval node builds its own message — the gates ask genuinely
                # different questions (approve the plan? the code? pushing it?), not one shared template.
                span.set_attribute("workflow.interrupted", True)
                reply = result["__interrupt__"][0].value.get("message", "")
                state_values = (await graph.aget_state(config)).values
                target_file = state_values.get("target_file")
                if target_file and state_values.get("new_content"):
                    diff = {
                        "target_file": target_file,
                        "original_content": state_values.get("original_content"),
                        "new_content": state_values.get("new_content", ""),
                    }
            else:
                span.set_attribute("workflow.interrupted", False)
                reply = result.get("reply", "(the agent finished without producing a reply)")
                span.set_attribute("workflow.outcome", result.get("outcome") or "chat")
                await _maybe_record_release(graph, config, req.workspace_path)

            if redirected:
                reply = f"(Stopped the previous change — starting this instead.)\n\n{reply}"

            # `awaiting_approval` lets the extension render real Approve/Decline buttons
            # instead of the user having to type "approve" by hand — a structured signal,
            # not the UI trying to pattern-match the reply text for "Reply 'approve' to...".
            return {
                "reply": reply,
                "model": "releaseforge-agent",
                "diff": diff,
                "awaiting_approval": awaiting_approval,
            }
        except Exception as exc:
            record_exception(span, exc)
            raise
