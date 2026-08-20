from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from app.agents.graph import build_graph, is_affirmative
from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider
from app.memory import repository as memory
from app.memory.db import init_db

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
    await init_db()
    # SQLite-backed, not in-memory — a paused approval (or an entire conversation)
    # now survives a backend restart. MemorySaver lost everything the moment the
    # process died, which is exactly what happened to a real in-progress approval.
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        yield


app = FastAPI(title="DeployGenie Core", lifespan=lifespan)


class PingRequest(BaseModel):
    workspace_path: str
    message: str = "Say hello in one short sentence."


@app.get("/health")
async def health():
    return {"status": "ok"}


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

    await memory.record_release(
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
        outcome=outcome,
    )


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    graph = request.app.state.graph
    # One active run per repo for now — real per-conversation sessions are a Phase 5.5+ concern.
    config = {"configurable": {"thread_id": req.workspace_path}}
    last_message = req.messages[-1].content if req.messages else ""

    pending = await graph.aget_state(config)
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

    diff = None
    if "__interrupt__" in result:
        # Each await_*_approval node builds its own message — the gates ask genuinely
        # different questions (approve the plan? the code? pushing it?), not one shared template.
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
        reply = result.get("reply", "(the agent finished without producing a reply)")
        await _maybe_record_release(graph, config, req.workspace_path)

    if redirected:
        reply = f"(Stopped the previous change — starting this instead.)\n\n{reply}"

    return {"reply": reply, "model": "deploygenie-agent", "diff": diff}
