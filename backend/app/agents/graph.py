import json
import re
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents import tools
from app.agents.state import AgentState
from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider

AFFIRMATIVE = {"yes", "y", "approve", "approved", "go ahead", "ship it", "do it", "lgtm", "ok", "okay"}


def is_affirmative(text: str) -> bool:
    return text.strip().lower() in AFFIRMATIVE


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def classify_intent(state: AgentState) -> dict:
    llm = get_llm_provider()
    prompt = (
        "Classify the user's message as exactly one word: TASK or CHAT.\n"
        "TASK = they want you to inspect, add, change, fix, or test something in their repository.\n"
        "CHAT = anything else (questions, greetings, general conversation).\n\n"
        f"Message: {state['user_message']!r}\n\n"
        "Reply with exactly one word: TASK or CHAT."
    )
    response = await llm.chat([LLMMessage(role="user", content=prompt)])
    intent = "task" if "TASK" in response.content.upper() else "chat"
    return {"intent": intent}


def route_intent(state: AgentState) -> str:
    return state["intent"]


async def chat_reply(state: AgentState) -> dict:
    llm = get_llm_provider()
    system = (
        "You are DeployGenie, an AI release engineering assistant. "
        f"The user's current repository is at: {state['workspace_path']}. "
        "This message doesn't look like a repo task, so respond conversationally. "
        "If they do want a change made, tell them to describe it directly and specifically."
    )
    response = await llm.chat(
        [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=state["user_message"]),
        ]
    )
    return {"reply": response.content}


async def plan(state: AgentState) -> dict:
    llm = get_llm_provider()
    files = tools.list_files(state["workspace_path"])
    prompt = (
        "You are the Planner in an AI release engineering pipeline. A user asked for a change "
        "in their repository. Pick exactly ONE existing or new file to modify to satisfy the "
        "request — keep this first change small and focused.\n\n"
        f"User request: {state['user_message']!r}\n\n"
        "Files currently in the repo (may be truncated):\n" + "\n".join(files) + "\n\n"
        "Reply with ONLY a JSON object, no other text, shaped exactly like:\n"
        '{"plan": "one or two sentence plan", "target_file": "relative/path.ext", '
        '"change_description": "precise description of the change to make in that file"}'
    )
    response = await llm.chat([LLMMessage(role="user", content=prompt)])
    parsed = _extract_json(response.content)

    if not parsed or not parsed.get("target_file"):
        return {
            "reply": (
                "I couldn't turn that into a plan for a single file — could you name a "
                "specific file, endpoint, or feature to change?"
            )
        }

    return {
        "plan": parsed.get("plan", ""),
        "target_file": parsed["target_file"],
        "change_description": parsed.get("change_description", state["user_message"]),
    }


def route_after_plan(state: AgentState) -> str:
    return "code" if state.get("target_file") else "end"


async def code(state: AgentState) -> dict:
    llm = get_llm_provider()
    target = state["target_file"]
    original = tools.read_file(state["workspace_path"], target)

    prompt = (
        f"Write the complete new contents of the file `{target}`.\n\n"
        f"Change to make: {state['change_description']}\n\n"
        + (
            f"Current file contents:\n```\n{original}\n```\n\n"
            if original is not None
            else "This file does not exist yet — create it.\n\n"
        )
        + "Reply with ONLY the raw file contents. No explanation, no markdown code fences."
    )
    response = await llm.chat([LLMMessage(role="user", content=prompt)])
    new_content = response.content.strip()
    fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", new_content, re.DOTALL)
    if fence:
        new_content = fence.group(1)

    tools.write_file(state["workspace_path"], target, new_content)
    return {"original_content": original, "new_content": new_content}


async def test(state: AgentState) -> dict:
    passed, output = tools.run_tests(state["workspace_path"])
    return {"test_passed": passed, "test_output": output}


def await_approval(state: AgentState) -> dict:
    summary = {
        "plan": state.get("plan", ""),
        "target_file": state.get("target_file", ""),
        "test_passed": state.get("test_passed"),
        "test_output": (state.get("test_output") or "")[-800:],
    }
    response = interrupt(summary)
    return {"approval_response": response}


async def create_pr(state: AgentState) -> dict:
    if not is_affirmative(state.get("approval_response", "")):
        return {
            "reply": (
                "Okay — the change is still in your working directory, but I haven't "
                "committed, pushed, or opened a PR. Nothing left this machine."
            )
        }

    workspace = state["workspace_path"]
    target = state["target_file"]
    base_branch = tools.current_branch(workspace)
    slug = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")[:40]
    branch_name = f"deploygenie/{slug}-{uuid.uuid4().hex[:6]}"
    description = state.get("change_description", target)
    if len(description) > 72:
        description = description[:72].rsplit(" ", 1)[0] + "..."
    commit_message = f"DeployGenie: {description}"

    pushed, log = tools.commit_and_push(workspace, branch_name, [target], commit_message)
    if not pushed:
        return {"reply": f"Approved, but `git push` failed:\n\n{log}"}

    ok, pr_result = tools.open_pull_request(
        workspace,
        branch_name,
        base_branch,
        title=commit_message,
        body=f"{state.get('plan', '')}\n\nOpened by DeployGenie's Phase 1 vertical-slice agent.",
    )
    if ok:
        return {
            "branch_name": branch_name,
            "base_branch": base_branch,
            "pr_url": pr_result,
            "reply": f"Pushed `{branch_name}` and opened a PR: {pr_result}",
        }
    return {
        "branch_name": branch_name,
        "base_branch": base_branch,
        "pr_error": pr_result,
        "reply": f"Pushed `{branch_name}`, but couldn't open a PR automatically: {pr_result}",
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("chat_reply", chat_reply)
    graph.add_node("plan", plan)
    graph.add_node("code", code)
    graph.add_node("test", test)
    graph.add_node("await_approval", await_approval)
    graph.add_node("create_pr", create_pr)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent, {"chat": "chat_reply", "task": "plan"})
    graph.add_edge("chat_reply", END)
    graph.add_conditional_edges("plan", route_after_plan, {"code": "code", "end": END})
    graph.add_edge("code", "test")
    graph.add_edge("test", "await_approval")
    graph.add_edge("await_approval", "create_pr")
    graph.add_edge("create_pr", END)

    return graph.compile(checkpointer=MemorySaver())


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
