from fastapi import FastAPI
from langgraph.types import Command
from pydantic import BaseModel

from app.agents.graph import get_graph
from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider

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
    if pending.next:
        result = await graph.ainvoke(Command(resume=last_message), config=config)
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

    return {"reply": reply, "model": "deploygenie-agent", "diff": diff}
