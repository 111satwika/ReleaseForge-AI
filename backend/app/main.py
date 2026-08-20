from fastapi import FastAPI
from langgraph.types import Command
from pydantic import BaseModel

from app.agents.graph import get_graph, is_affirmative
from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider

NEGATIVE = {"no", "n", "nah", "stop", "cancel", "don't", "dont", "nevermind", "never mind", "decline", "rejected", "reject"}


def _looks_like_new_instruction(text: str) -> bool:
    """A decline reply that isn't affirmative *or* a plain "no" is probably not a
    rejection at all — it's the user moving on to a different request mid-approval."""
    normalized = text.strip().lower()
    if not normalized or normalized in NEGATIVE:
        return False
    return len(normalized.split()) >= 3


app = FastAPI(title="DeployGenie Core")


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


@app.post("/chat")
async def chat(req: ChatRequest):
    graph = get_graph()
    # One active run per repo for now — real per-conversation sessions arrive with Phase 5's memory layer.
    config = {"configurable": {"thread_id": req.workspace_path}}
    last_message = req.messages[-1].content if req.messages else ""

    pending = graph.get_state(config)
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
        state_values = graph.get_state(config).values
        target_file = state_values.get("target_file")
        if target_file and state_values.get("new_content"):
            diff = {
                "target_file": target_file,
                "original_content": state_values.get("original_content"),
                "new_content": state_values.get("new_content", ""),
            }
    else:
        reply = result.get("reply", "(the agent finished without producing a reply)")

    if redirected:
        reply = f"(Stopped the previous change — starting this instead.)\n\n{reply}"

    return {"reply": reply, "model": "deploygenie-agent", "diff": diff}
