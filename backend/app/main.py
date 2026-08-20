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
