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


def _format_approval_prompt(payload: dict) -> str:
    test_passed = payload.get("test_passed")
    if test_passed is True:
        test_status = "passed"
    elif test_passed is False:
        test_status = f"FAILED\n{payload.get('test_output', '')}"
    else:
        test_status = f"skipped — {payload.get('test_output', '')}"

    return (
        f"Plan: {payload.get('plan')}\n"
        f"File: {payload.get('target_file')}\n"
        f"Tests: {test_status}\n\n"
        "Reply 'approve' to push a branch and open a pull request, or anything else to stop here "
        "(the file change stays on disk, uncommitted)."
    )


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

    if "__interrupt__" in result:
        reply = _format_approval_prompt(result["__interrupt__"][0].value)
    else:
        reply = result.get("reply", "(the agent finished without producing a reply)")

    return {"reply": reply, "model": "deploygenie-agent"}
