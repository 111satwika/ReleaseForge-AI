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
MAX_REVIEW_ROUNDS = 2


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
    return {
        "intent": intent,
        # A fresh task on this thread — clear anything a previous, unrelated task on the
        # same repo left behind, so e.g. publish() doesn't mistake this for a retry of an
        # old PR that has nothing to do with the new request.
        "plan": "",
        "target_file": "",
        "change_description": "",
        "original_content": None,
        "new_content": "",
        "test_passed": None,
        "test_output": "",
        "review_status": "",
        "review_feedback": "",
        "review_risk": "",
        "review_round": 0,
        "branch_name": "",
        "base_branch": "",
        "pr_url": "",
        "pr_error": "",
        "readiness_score": 0,
        "readiness_recommendation": "",
        "readiness_breakdown": [],
        "reply": "",
    }


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


def _resolve_target_file(target_file: str, files: list[str]) -> str:
    """If the model returned a bare/shortened filename instead of the real repo-relative
    path, and exactly one file in the repo has that name, use the real path instead of
    trusting the guess. Ambiguous or genuinely-new files are left as the model chose."""
    if target_file in files:
        return target_file
    basename = target_file.rsplit("/", 1)[-1]
    matches = [f for f in files if f.rsplit("/", 1)[-1] == basename]
    if len(matches) == 1:
        return matches[0]
    return target_file


async def plan(state: AgentState) -> dict:
    llm = get_llm_provider()
    files = tools.list_files(state["workspace_path"])
    prompt = (
        "You are the Planner in an AI release engineering pipeline. A user asked for a change "
        "in their repository. Pick exactly ONE existing or new file to modify to satisfy the "
        "request — keep this first change small and focused. If the file already appears in "
        "the list below, copy its path EXACTLY as shown there — do not shorten, guess, or drop "
        "its folder.\n\n"
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

    target_file = _resolve_target_file(parsed["target_file"], files)

    return {
        "plan": parsed.get("plan", ""),
        "target_file": target_file,
        "change_description": parsed.get("change_description", state["user_message"]),
    }


def route_after_plan(state: AgentState) -> str:
    return "await_plan_approval" if state.get("target_file") else "end"


def await_plan_approval(state: AgentState) -> dict:
    message = (
        f"Plan: {state.get('plan', '')}\n"
        f"File: {state.get('target_file', '')}\n\n"
        "Reply 'approve' to start making this change, or anything else to stop here — "
        "nothing has been touched yet."
    )
    response = interrupt({"message": message})
    return {"approval_response": response}


def route_after_plan_approval(state: AgentState) -> str:
    return "code" if is_affirmative(state.get("approval_response", "")) else "declined_at_plan"


def declined_at_plan(state: AgentState) -> dict:
    return {"reply": "Okay — I won't make this change."}


async def code(state: AgentState) -> dict:
    llm = get_llm_provider()
    target = state["target_file"]

    # review_round > 0 means this is a retry (review already ran once and sent us back) —
    # original_content is the *pristine* pre-edit version, don't re-read from disk, that
    # would pick up our own last attempt instead of the true original.
    is_retry = state.get("review_round", 0) > 0
    original = state.get("original_content") if is_retry else tools.read_file(state["workspace_path"], target)

    retry_block = ""
    if is_retry and state.get("review_feedback"):
        retry_block = (
            "\nYour previous attempt:\n```\n" + state.get("new_content", "") + "\n```\n\n"
            "An independent reviewer requested changes to that attempt:\n"
            f"{state['review_feedback']}\n\n"
            "Write a new, corrected version that addresses this feedback.\n"
        )

    prompt = (
        f"Write the complete new contents of the file `{target}`.\n\n"
        f"Change to make: {state['change_description']}\n\n"
        + (
            f"Current file contents:\n```\n{original}\n```\n\n"
            if original is not None
            else "This file does not exist yet — create it.\n\n"
        )
        + retry_block
        + "Reply with ONLY the raw file contents. No explanation, no markdown code fences."
    )
    response = await llm.chat([LLMMessage(role="user", content=prompt)])
    new_content = response.content.strip()
    fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", new_content, re.DOTALL)
    if fence:
        new_content = fence.group(1)

    tools.write_file(state["workspace_path"], target, new_content)
    return {"original_content": original, "new_content": new_content}


def await_code_approval(state: AgentState) -> dict:
    is_retry = state.get("review_round", 0) > 0
    header = "Here's a fix based on the reviewer's feedback:" if is_retry else "Here's the change:"
    message = (
        f"{header}\n"
        f"File: {state.get('target_file', '')}\n\n"
        "Reply 'approve' to run the tests, or anything else to stop here — "
        "the change stays on disk, uncommitted."
    )
    response = interrupt({"message": message})
    return {"approval_response": response}


def route_after_code_approval(state: AgentState) -> str:
    return "test" if is_affirmative(state.get("approval_response", "")) else "declined_at_code"


def declined_at_code(state: AgentState) -> dict:
    note = ""
    if state.get("branch_name"):
        note = f" The existing PR still reflects the previous version: {state.get('pr_url') or state['branch_name']}."
    return {"reply": f"Okay — the change is still on disk, uncommitted. Nothing else happened.{note}"}


async def test(state: AgentState) -> dict:
    passed, output = tools.run_tests(state["workspace_path"])
    return {"test_passed": passed, "test_output": output}


def await_test_approval(state: AgentState) -> dict:
    if state.get("test_passed") is True:
        test_status = "passed"
    elif state.get("test_passed") is False:
        test_status = f"FAILED\n{(state.get('test_output') or '')[-800:]}"
    else:
        test_status = f"skipped — {state.get('test_output', '')}"

    is_retry = bool(state.get("branch_name"))
    action = "push this fix to the existing pull request" if is_retry else "push a branch and open a pull request"
    message = f"Tests: {test_status}\n\nReply 'approve' to {action}, or anything else to stop here."
    response = interrupt({"message": message})
    return {"approval_response": response}


def route_after_test_approval(state: AgentState) -> str:
    return "publish" if is_affirmative(state.get("approval_response", "")) else "declined_at_test"


def declined_at_test(state: AgentState) -> dict:
    if state.get("branch_name"):
        return {
            "reply": (
                "Okay — I won't push this fix. The existing PR still reflects the previous "
                f"version: {state.get('pr_url') or state['branch_name']}."
            )
        }
    return {"reply": "Okay — I won't create a PR. The change is still on disk, uncommitted."}


async def publish(state: AgentState) -> dict:
    workspace = state["workspace_path"]
    target = state["target_file"]
    description = state.get("change_description", target)
    if len(description) > 72:
        description = description[:72].rsplit(" ", 1)[0] + "..."

    if state.get("branch_name"):
        # Retry: push a fix commit to the already-open PR's branch instead of opening a new one.
        commit_message = f"DeployGenie: fix per review — {description}"
        pushed, log = tools.push_fix_commit(workspace, [target], commit_message)
        if not pushed:
            return {"reply": f"Approved, but pushing the fix failed:\n\n{log}"}
        return {}

    base_branch = tools.current_branch(workspace)
    slug = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")[:40]
    branch_name = f"deploygenie/{slug}-{uuid.uuid4().hex[:6]}"
    commit_message = f"DeployGenie: {description}"

    pushed, log = tools.commit_and_push(workspace, branch_name, [target], commit_message)
    if not pushed:
        return {"reply": f"Approved, but `git push` failed:\n\n{log}"}

    ok, pr_result = tools.open_pull_request(
        workspace,
        branch_name,
        base_branch,
        title=commit_message,
        body=f"{state.get('plan', '')}\n\nOpened by DeployGenie, before independent review.",
    )
    if ok:
        return {"branch_name": branch_name, "base_branch": base_branch, "pr_url": pr_result}
    return {"branch_name": branch_name, "base_branch": base_branch, "pr_error": pr_result}


def route_after_publish(state: AgentState) -> str:
    # publish() sets "reply" itself only when it hit an error (push/PR failure) — stop there
    # instead of trying to review a PR that was never actually created or updated.
    return "end" if state.get("reply") else "review"


async def review(state: AgentState) -> dict:
    llm = get_llm_provider()
    round_num = state.get("review_round", 0) + 1

    if state.get("test_passed") is True:
        test_status = "PASSED"
    elif state.get("test_passed") is False:
        test_status = "FAILED"
    else:
        test_status = "no tests found"

    prompt = (
        "You are an independent code reviewer, reviewing a pull request someone else opened. "
        "You did not write this change — a separate Coding Agent did. Review it skeptically: "
        "does it actually satisfy the request, is it correct, is it safe? Do not approve just "
        "to be agreeable. If the test result is FAILED, you must request changes and list it "
        "as critical.\n\n"
        f"Requested change: {state.get('change_description', '')}\n\n"
        f"File: {state.get('target_file', '')}\n\n"
        "Before:\n```\n" + (state.get("original_content") or "(new file)") + "\n```\n\n"
        "After:\n```\n" + state.get("new_content", "") + "\n```\n\n"
        f"Test result: {test_status}\n\n"
        "Reply with ONLY a JSON object, no other text, shaped exactly like:\n"
        '{"status": "APPROVED" or "CHANGES_REQUESTED", "critical": ["..."], '
        '"warnings": ["..."], "suggestions": ["..."], "risk": "Low", "Medium", or "High"}'
    )
    response = await llm.chat([LLMMessage(role="user", content=prompt)])
    parsed = _extract_json(response.content)

    if not parsed:
        # Can't parse the review — fail open rather than looping forever on a formatting hiccup.
        # The human approval gates downstream are still the real safety net either way.
        return {"review_status": "APPROVED", "review_feedback": "", "review_round": round_num}

    status = str(parsed.get("status", "APPROVED")).upper()
    if status not in ("APPROVED", "CHANGES_REQUESTED"):
        status = "APPROVED"

    lines = []
    for item in parsed.get("critical") or []:
        lines.append(f"Critical: {item}")
    for item in parsed.get("warnings") or []:
        lines.append(f"Warning: {item}")
    for item in parsed.get("suggestions") or []:
        lines.append(f"Suggestion: {item}")

    return {
        "review_status": status,
        "review_feedback": "\n".join(lines),
        "review_risk": parsed.get("risk", "Unknown"),
        "review_round": round_num,
    }


def route_after_review(state: AgentState) -> str:
    if state.get("review_status") == "CHANGES_REQUESTED" and state.get("review_round", 0) < MAX_REVIEW_ROUNDS:
        return "code"
    if state.get("review_status") == "APPROVED":
        return "release_readiness"
    return "finish_unresolved"


def finish_unresolved(state: AgentState) -> dict:
    # Only reached when review still says CHANGES_REQUESTED after hitting the round cap —
    # report it plainly rather than proceeding to readiness scoring as if it were approved.
    pr_url = state.get("pr_url") or state.get("branch_name", "")
    reply = (
        f"Review still has concerns after {state.get('review_round', 1)} round(s), but I've "
        f"stopped retrying:\n{state.get('review_feedback', '')}\n\n"
        f"The PR is open for you to look at: {pr_url}"
    )
    return {"reply": reply}


def release_readiness(state: AgentState) -> dict:
    """Deterministic, not an LLM guess — the score is only ever built from signals
    DeployGenie actually has (tests, review). No security/dependency/infra checks exist
    yet, so it doesn't pretend to have run any."""
    score = 0
    breakdown: list[str] = []

    if state.get("test_passed") is True:
        score += 50
        breakdown.append("Tests passed (+50)")
    elif state.get("test_passed") is None:
        score += 20
        breakdown.append("No tests found to run (+20 — untested, not known to be broken)")
    else:
        breakdown.append("Tests failed (+0)")

    if state.get("review_status") == "APPROVED":
        score += 40
        breakdown.append("Independent review approved (+40)")

    risk = state.get("review_risk", "Unknown")
    risk_points = {"Low": 10, "Medium": 5}.get(risk, 0)
    score += risk_points
    breakdown.append(f"Review risk: {risk} (+{risk_points})")

    return {
        "readiness_score": score,
        "readiness_recommendation": "GO" if score >= 70 else "NO-GO",
        "readiness_breakdown": breakdown,
    }


def await_readiness_approval(state: AgentState) -> dict:
    breakdown = "\n".join(f"- {line}" for line in state.get("readiness_breakdown", []))
    message = (
        f"Release Readiness: {state.get('readiness_score', 0)}/100\n"
        f"Recommendation: {state.get('readiness_recommendation', 'NO-GO')}\n"
        f"{breakdown}\n\n"
        "This score only reflects tests and independent review — there's no security scan, "
        "dependency check, or infrastructure readiness check built yet.\n\n"
        "Reply 'approve' to generate a Dockerfile (if one doesn't exist) and build a local "
        "image, or anything else to stop here — the PR stays open either way."
    )
    response = interrupt({"message": message})
    return {"approval_response": response}


def route_after_readiness_approval(state: AgentState) -> str:
    return "build_docker_image" if is_affirmative(state.get("approval_response", "")) else "declined_at_readiness"


def declined_at_readiness(state: AgentState) -> dict:
    pr_url = state.get("pr_url") or state.get("branch_name", "")
    return {"reply": f"Okay — I won't build an image. The PR is still open: {pr_url}"}


async def build_docker_image(state: AgentState) -> dict:
    workspace = state["workspace_path"]
    pr_url = state.get("pr_url") or state.get("branch_name", "")

    generated_note = ""
    if not tools.has_dockerfile(workspace):
        llm = get_llm_provider()
        files = tools.list_files(workspace)
        prompt = (
            "Write a Dockerfile for this repository. Look at the file list to infer the "
            "language and framework. Keep it simple and correct — a working Dockerfile "
            "matters far more than a clever one.\n\n"
            "Files in the repo:\n" + "\n".join(files) + "\n\n"
            "Reply with ONLY the raw Dockerfile contents. No explanation, no markdown fences."
        )
        response = await llm.chat([LLMMessage(role="user", content=prompt)])
        content = response.content.strip()
        fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", content, re.DOTALL)
        if fence:
            content = fence.group(1)
        tools.write_file(workspace, "Dockerfile", content)

        pushed, log = tools.push_fix_commit(workspace, ["Dockerfile"], "DeployGenie: add Dockerfile")
        generated_note = (
            "Generated a Dockerfile and added it to the PR.\n\n"
            if pushed
            else f"Generated a Dockerfile, but couldn't push it:\n{log}\n\n"
        )

    branch_name = state.get("branch_name", "")
    tag_suffix = branch_name.split("/")[-1] if branch_name else "latest"
    tag = f"deploygenie-{tools.repo_name(workspace).lower().replace(' ', '-')}:{tag_suffix}"
    ok, log = tools.run_docker_build(workspace, tag)

    if ok:
        return {"reply": f"{generated_note}Docker image built: {tag}\n\nPR: {pr_url}"}
    return {"reply": f"{generated_note}Docker build failed:\n\n{log}\n\nPR is still open: {pr_url}"}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("chat_reply", chat_reply)
    graph.add_node("plan", plan)
    graph.add_node("await_plan_approval", await_plan_approval)
    graph.add_node("declined_at_plan", declined_at_plan)
    graph.add_node("code", code)
    graph.add_node("await_code_approval", await_code_approval)
    graph.add_node("declined_at_code", declined_at_code)
    graph.add_node("test", test)
    graph.add_node("await_test_approval", await_test_approval)
    graph.add_node("declined_at_test", declined_at_test)
    graph.add_node("publish", publish)
    graph.add_node("review", review)
    graph.add_node("finish_unresolved", finish_unresolved)
    graph.add_node("release_readiness", release_readiness)
    graph.add_node("await_readiness_approval", await_readiness_approval)
    graph.add_node("declined_at_readiness", declined_at_readiness)
    graph.add_node("build_docker_image", build_docker_image)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent, {"chat": "chat_reply", "task": "plan"})
    graph.add_edge("chat_reply", END)

    graph.add_conditional_edges("plan", route_after_plan, {"await_plan_approval": "await_plan_approval", "end": END})
    graph.add_conditional_edges(
        "await_plan_approval", route_after_plan_approval, {"code": "code", "declined_at_plan": "declined_at_plan"}
    )
    graph.add_edge("declined_at_plan", END)

    graph.add_edge("code", "await_code_approval")
    graph.add_conditional_edges(
        "await_code_approval", route_after_code_approval, {"test": "test", "declined_at_code": "declined_at_code"}
    )
    graph.add_edge("declined_at_code", END)

    graph.add_edge("test", "await_test_approval")
    graph.add_conditional_edges(
        "await_test_approval",
        route_after_test_approval,
        {"publish": "publish", "declined_at_test": "declined_at_test"},
    )
    graph.add_edge("declined_at_test", END)

    graph.add_conditional_edges("publish", route_after_publish, {"review": "review", "end": END})
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {"code": "code", "release_readiness": "release_readiness", "finish_unresolved": "finish_unresolved"},
    )
    graph.add_edge("finish_unresolved", END)

    graph.add_edge("release_readiness", "await_readiness_approval")
    graph.add_conditional_edges(
        "await_readiness_approval",
        route_after_readiness_approval,
        {"build_docker_image": "build_docker_image", "declined_at_readiness": "declined_at_readiness"},
    )
    graph.add_edge("declined_at_readiness", END)
    graph.add_edge("build_docker_image", END)

    return graph.compile(checkpointer=MemorySaver())


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
