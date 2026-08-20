from fastapi import FastAPI
from pydantic import BaseModel

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
    llm = get_llm_provider()

    system_prompt = (
        "You are DeployGenie, an AI release engineering assistant. "
        f"The user's current repository is at: {req.workspace_path}. "
        "You do not yet have tools to read files, run commands, or make changes — "
        "that arrives in a later development phase. If asked to inspect or modify "
        "the repo, say that plainly instead of pretending to have done it."
    )
    llm_messages = [LLMMessage(role="system", content=system_prompt)]
    llm_messages += [LLMMessage(role=m.role, content=m.content) for m in req.messages]

    response = await llm.chat(llm_messages)
    return {"reply": response.content, "model": response.model}
