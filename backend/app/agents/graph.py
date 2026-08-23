import asyncio
import json
import logging
import re
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents import k8s_tools
from app.agents.state import AgentState
from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider
from app.mcp.factory import get_mcp_client
from app.memory import repository as memory
from app.rag.factory import get_graph_store, get_vector_store
from app.rag.retriever import hybrid_retrieve

# `k8s_tools` is still imported directly — but only for its pure, no-subprocess helpers
# (build_deployment_manifest, sanitize_dns_label, APP_LABEL, ...). Every function that
# actually touches git, Docker, or a real cluster now goes through the MCP client below,
# talking to the filesystem/docker/kubernetes MCP servers instead of a plain import.

logger = logging.getLogger(__name__)

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
        "outcome": "",
    }


def route_intent(state: AgentState) -> str:
    return state["intent"]


async def _retrieved_context(workspace_path: str, question: str) -> str:
    """Best-effort RAG grounding for chat: relevant chunks from indexed release history
    and project docs, plus known graph relationships for this repo. Returns "" on any
    failure (most likely: Ollama isn't running, or the embedding model isn't pulled) —
    chat must still work without it, just without the extra grounding."""
    try:
        llm = get_llm_provider()
        hits = await hybrid_retrieve(question, llm.embed, get_vector_store(), k=4)
    except Exception:
        logger.warning("RAG retrieval skipped for chat — embedding call failed", exc_info=True)
        return ""

    if not hits:
        return ""

    passages = "\n\n".join(f"[{h['metadata'].get('source', 'unknown')}] {h['text']}" for h in hits)
    facts = get_graph_store().recent_relationships(workspace_path, limit=6)
    facts_block = ""
    if facts:
        facts_block = "\n\nKnown relationships for this repo:\n" + "\n".join(
            f"- {f['from']} --{f['relation']}--> {f['to']}" for f in facts
        )

    return (
        "\n\nRelevant context retrieved from indexed release history and project docs "
        "(cite the bracketed source when you use one; ignore anything irrelevant to the "
        f"question):\n{passages}{facts_block}"
    )


async def chat_reply(state: AgentState) -> dict:
    llm = get_llm_provider()
    context = await _retrieved_context(state["workspace_path"], state["user_message"])
    system = (
        "You are ReleaseForge AI, an AI release engineering assistant. "
        f"The user's current repository is at: {state['workspace_path']}. "
        "This message doesn't look like a repo task, so respond conversationally. "
        "If they do want a change made, tell them to describe it directly and specifically."
        + context
    )
    response = await llm.chat(
        [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=state["user_message"]),
        ]
    )
    return {"reply": response.content}


def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _stem(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _resolve_target_file(target_file: str, files: list[str]) -> str:
    """If the model's target_file doesn't exist verbatim, try two safe, unambiguous
    corrections before trusting the guess:
      1. Same filename, different folder (a dropped or wrong path prefix).
      2. Same name and folder, different extension — e.g. asked to edit login.html
         when the project's real file is login.vue, a framework mismatch, not a
         typo. Only applied when exactly one file matches; genuinely new files or
         ambiguous cases (multiple matches) are left as the model chose."""
    if target_file in files:
        return target_file

    basename = target_file.rsplit("/", 1)[-1]
    same_name = [f for f in files if f.rsplit("/", 1)[-1] == basename]
    if len(same_name) == 1:
        return same_name[0]

    directory, stem = _dirname(target_file), _stem(target_file)
    same_stem_same_dir = [f for f in files if _dirname(f) == directory and _stem(f) == stem]
    if len(same_stem_same_dir) == 1:
        return same_stem_same_dir[0]

    return target_file


async def plan(state: AgentState) -> dict:
    llm = get_llm_provider()
    mcp = get_mcp_client()
    workspace = state["workspace_path"]
    files = await mcp.call_tool("coding", "list_files", {"workspace_path": workspace})

    cached_command = await memory.get_cached_test_command(workspace)
    project_context = f"This repo's test command is already known: `{cached_command}`.\n\n" if cached_command else ""

    prompt = (
        "You are the Planner in an AI release engineering pipeline. A user asked for a change "
        "in their repository. Pick exactly ONE existing or new file to modify to satisfy the "
        "request — keep this first change small and focused. If the file already appears in "
        "the list below, copy its path EXACTLY as shown there — do not shorten it, guess it, "
        "drop its folder, or guess a different extension than what similar files in this repo "
        "actually use (e.g. a Vue project's pages are .vue files, not .html).\n\n"
        + project_context
        + f"User request: {state['user_message']!r}\n\n"
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
            ),
            "outcome": "planning_failed",
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
    return {"reply": "Okay — I won't make this change.", "outcome": "declined_at_plan"}


async def code(state: AgentState) -> dict:
    llm = get_llm_provider()
    mcp = get_mcp_client()
    target = state["target_file"]

    # review_round > 0 means this is a retry (review already ran once and sent us back) —
    # original_content is the *pristine* pre-edit version, don't re-read from disk, that
    # would pick up our own last attempt instead of the true original.
    is_retry = state.get("review_round", 0) > 0
    if is_retry:
        original = state.get("original_content")
    else:
        read_result = await mcp.call_tool(
            "coding", "read_file", {"workspace_path": state["workspace_path"], "relative_path": target}
        )
        original = read_result["content"] if read_result["exists"] else None

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

    await mcp.call_tool(
        "coding",
        "write_file",
        {"workspace_path": state["workspace_path"], "relative_path": target, "content": new_content},
    )
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
    return {
        "reply": f"Okay — the change is still on disk, uncommitted. Nothing else happened.{note}",
        "outcome": "declined_at_code",
    }


async def test(state: AgentState) -> dict:
    workspace = state["workspace_path"]
    mcp = get_mcp_client()

    cached_command = await memory.get_cached_test_command(workspace)
    if cached_command:
        command = cached_command.split()
    else:
        detected = await mcp.call_tool("test", "detect_test_command", {"workspace_path": workspace})
        command = detected["command"]
        if command:
            await memory.remember_test_command(workspace, " ".join(command))

    # No asyncio.to_thread needed here anymore: the real subprocess (up to 90s) now runs
    # inside the filesystem MCP server's own process, not this one — awaiting the MCP
    # call is just waiting on a pipe read, which doesn't block this backend's event loop
    # the way calling subprocess.run() directly in-process would have.
    result = await mcp.call_tool("test", "run_tests", {"workspace_path": workspace, "command": command})
    return {"test_passed": result["passed"], "test_output": result["output"]}


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
            ),
            "outcome": "declined_at_test",
        }
    return {
        "reply": "Okay — I won't create a PR. The change is still on disk, uncommitted.",
        "outcome": "declined_at_test",
    }


async def publish(state: AgentState) -> dict:
    mcp = get_mcp_client()
    workspace = state["workspace_path"]
    target = state["target_file"]
    description = state.get("change_description", target)
    if len(description) > 72:
        description = description[:72].rsplit(" ", 1)[0] + "..."

    # Every call below can block on a real git/network subprocess for up to 60s (see
    # tools._run_git, on the filesystem MCP server side) — that blocking now happens in
    # a different OS process entirely (the MCP server), so awaiting it here never risks
    # freezing this backend's own event loop for other requests.
    if state.get("branch_name"):
        # Retry: push a fix commit to the already-open PR's branch instead of opening a new one.
        commit_message = f"ReleaseForge: fix per review — {description}"
        result = await mcp.call_tool(
            "publish",
            "push_fix_commit",
            {"workspace_path": workspace, "files": [target], "commit_message": commit_message},
        )
        if not result["ok"]:
            return {"reply": f"Approved, but pushing the fix failed:\n\n{result['log']}", "outcome": "publish_failed"}
        return {}

    base_branch = await mcp.call_tool("publish", "current_branch", {"workspace_path": workspace})
    slug = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")[:40]
    branch_name = f"releaseforge/{slug}-{uuid.uuid4().hex[:6]}"
    commit_message = f"ReleaseForge: {description}"

    push_result = await mcp.call_tool(
        "publish",
        "commit_and_push",
        {"workspace_path": workspace, "branch_name": branch_name, "files": [target], "commit_message": commit_message},
    )
    if not push_result["ok"]:
        return {"reply": f"Approved, but `git push` failed:\n\n{push_result['log']}", "outcome": "publish_failed"}

    pr_result = await mcp.call_tool(
        "publish",
        "open_pull_request",
        {
            "workspace_path": workspace,
            "branch_name": branch_name,
            "base_branch": base_branch,
            "title": commit_message,
            "body": f"{state.get('plan', '')}\n\nOpened by ReleaseForge, before independent review.",
        },
    )
    if pr_result["ok"]:
        return {"branch_name": branch_name, "base_branch": base_branch, "pr_url": pr_result["result"]}
    return {"branch_name": branch_name, "base_branch": base_branch, "pr_error": pr_result["result"]}


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
    return {"reply": reply, "outcome": "review_unresolved"}


def release_readiness(state: AgentState) -> dict:
    """Deterministic, not an LLM guess — the score is only ever built from signals
    ReleaseForge AI actually has (tests, review). No security/dependency/infra checks exist
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
    return {"reply": f"Okay — I won't build an image. The PR is still open: {pr_url}", "outcome": "declined_at_readiness"}


async def build_docker_image(state: AgentState) -> dict:
    mcp = get_mcp_client()
    workspace = state["workspace_path"]
    pr_url = state.get("pr_url") or state.get("branch_name", "")

    generated_note = ""
    has_dockerfile = await mcp.call_tool("build", "has_dockerfile", {"workspace_path": workspace})
    if not has_dockerfile:
        llm = get_llm_provider()
        files = await mcp.call_tool("build", "list_files", {"workspace_path": workspace})
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
        await mcp.call_tool(
            "build", "write_file", {"workspace_path": workspace, "relative_path": "Dockerfile", "content": content}
        )

        push_result = await mcp.call_tool(
            "build",
            "push_fix_commit",
            {"workspace_path": workspace, "files": ["Dockerfile"], "commit_message": "ReleaseForge: add Dockerfile"},
        )
        generated_note = (
            "Generated a Dockerfile and added it to the PR.\n\n"
            if push_result["ok"]
            else f"Generated a Dockerfile, but couldn't push it:\n{push_result['log']}\n\n"
        )

    branch_name = state.get("branch_name", "")
    tag_suffix = branch_name.split("/")[-1] if branch_name else "latest"
    repo = await mcp.call_tool("build", "repo_name", {"workspace_path": workspace})
    tag = f"releaseforge-{repo.lower().replace(' ', '-')}:{tag_suffix}"
    # No asyncio.to_thread here either — the real build (up to 300s) runs inside the
    # docker MCP server's own process now, a separate subprocess from this backend.
    build_result = await mcp.call_tool("build", "run_docker_build", {"workspace_path": workspace, "tag": tag})

    if build_result["ok"]:
        # Not terminal anymore — Phase 4 continues on into deployment. No "reply"/"outcome"
        # here on purpose: route_after_build_docker_image below uses the absence of
        # "reply" to mean "keep going," the same convention route_after_publish already
        # established for the review step.
        return {"docker_image_tag": tag, "deploy_log": generated_note}
    return {
        "reply": f"{generated_note}Docker build failed:\n\n{build_result['log']}\n\nPR is still open: {pr_url}",
        "outcome": "docker_build_failed",
    }


def route_after_build_docker_image(state: AgentState) -> str:
    return "end" if state.get("reply") else "select_deployment_strategy"


async def select_deployment_strategy(state: AgentState) -> dict:
    mcp = get_mcp_client()
    workspace = state["workspace_path"]
    repo = await mcp.call_tool("deployment", "repo_name", {"workspace_path": workspace})
    app_name = k8s_tools.sanitize_dns_label(repo)
    namespace = f"releaseforge-{app_name}"

    try:
        has_existing = await mcp.call_tool(
            "deployment", "deployment_exists", {"namespace": namespace, "deployment_name": app_name}
        )
    except RuntimeError:
        # No kubectl at all — the kubernetes MCP server raised, which the client surfaces
        # as a RuntimeError. decide_deployment_strategy still runs (Recreate, since
        # there's clearly no prior deployment either), and the actual missing-tool
        # failure surfaces cleanly later, in deploy(), the same graceful-failure pattern
        # already used for a missing `docker`/`gh` binary.
        has_existing = False

    strategy, reason = k8s_tools.decide_deployment_strategy(
        has_existing, state.get("review_risk", "Unknown"), state.get("readiness_score", 0)
    )
    return {
        "deployment_strategy": strategy,
        "deployment_strategy_reason": reason,
        "k8s_namespace": namespace,
        "k8s_app_name": app_name,
        "k8s_service_name": app_name,
    }


def await_deploy_approval(state: AgentState) -> dict:
    message = (
        f"Deployment strategy: {state.get('deployment_strategy', '')}\n"
        f"{state.get('deployment_strategy_reason', '')}\n\n"
        f"Target: local `kind` cluster, namespace `{state.get('k8s_namespace', '')}`\n\n"
        "This deploys to a local Kubernetes cluster on this machine, not anywhere public — "
        "still asking first, since infrastructure changes are a human-approval case regardless "
        "of environment.\n\n"
        "Reply 'approve' to deploy, or anything else to stop here — the built image and PR stay as they are."
    )
    response = interrupt({"message": message})
    return {"approval_response": response}


def route_after_deploy_approval(state: AgentState) -> str:
    return "deploy" if is_affirmative(state.get("approval_response", "")) else "declined_at_deploy"


def declined_at_deploy(state: AgentState) -> dict:
    pr_url = state.get("pr_url") or state.get("branch_name", "")
    return {
        "reply": f"Okay — I won't deploy. Image built as {state.get('docker_image_tag', '')}. PR: {pr_url}",
        "outcome": "declined_at_deploy",
    }


async def deploy(state: AgentState) -> dict:
    """Applies manifests appropriate to the chosen strategy. Deliberately does NOT decide
    success here — that's validate()'s job (a real `kubectl rollout status` wait), this
    node only reports whether the apply itself went through, so a cluster/image-load
    failure routes straight to recovery without wasting a rollout-status timeout on
    something that was never even applied."""
    mcp = get_mcp_client()
    namespace = state["k8s_namespace"]
    app_name = state["k8s_app_name"]
    service_name = state["k8s_service_name"]
    image = state["docker_image_tag"]
    strategy = state["deployment_strategy"]
    port = k8s_tools.detect_container_port(state["workspace_path"])
    log_lines = [state.get("deploy_log", "")]
    deploy_color = ""

    try:
        cluster_result = await mcp.call_tool("deployment", "ensure_kind_cluster", {})
        log_lines.append(f"kind cluster: {cluster_result['message']}")
        if not cluster_result["ok"]:
            return {"deploy_log": "\n".join(log_lines), "deploy_applied": False}

        ns_result = await mcp.call_tool("deployment", "ensure_namespace", {"namespace": namespace})
        log_lines.append(f"namespace: {ns_result['message']}")

        load_result = await mcp.call_tool("deployment", "kind_load_image", {"image_tag": image})
        log_lines.append(f"image load: {load_result['message']}")
        if not load_result["ok"]:
            return {"deploy_log": "\n".join(log_lines), "deploy_applied": False}

        if strategy in ("Recreate", "Rolling"):
            strategy_type = "Recreate" if strategy == "Recreate" else "RollingUpdate"
            deployment = k8s_tools.build_deployment_manifest(
                app_name, namespace, image, port, replicas=2, strategy_type=strategy_type
            )
            service = k8s_tools.build_service_manifest(
                service_name, namespace, port, {k8s_tools.APP_LABEL: app_name}
            )
            apply_result = await mcp.call_tool(
                "deployment", "apply_manifests", {"namespace": namespace, "manifests": [deployment, service]}
            )
            ok = apply_result["ok"]
            log_lines.append(f"apply: {apply_result['message']}")
            if ok:
                # Force the selector, don't rely on `kubectl apply`'s diff-against-
                # last-applied-configuration to fix it: if an earlier Blue-Green
                # promotion used `kubectl patch` to add a `version` key to this Service's
                # selector, apply's 3-way merge won't detect a change here (the manifest
                # text is identical to what was originally applied) and will report
                # "unchanged" while leaving that stale key in place — silently routing
                # to nothing once a Rolling/Recreate deploy happens right after.
                patch_result = await mcp.call_tool(
                    "deployment",
                    "patch_service_selector",
                    {"namespace": namespace, "service_name": service_name, "selector": {k8s_tools.APP_LABEL: app_name}},
                )
                log_lines.append(f"service selector reset: {patch_result['message']}")
            deploy_target = app_name

        elif strategy == "Blue-Green":
            selector_result = await mcp.call_tool(
                "deployment", "get_service_selector", {"namespace": namespace, "service_name": service_name}
            )
            current_selector = selector_result["selector"]
            current_color = (current_selector or {}).get(k8s_tools.VERSION_LABEL, "blue")
            deploy_color = "green" if current_color == "blue" else "blue"
            deploy_target = f"{app_name}-{deploy_color}"
            deployment = k8s_tools.build_deployment_manifest(
                deploy_target,
                namespace,
                image,
                port,
                replicas=2,
                strategy_type="RollingUpdate",
                extra_labels={k8s_tools.APP_LABEL: app_name, k8s_tools.VERSION_LABEL: deploy_color},
            )
            manifests = [deployment]
            if current_selector is None:
                # First-ever Service for this app: point it straight at the new color —
                # there's no live traffic on an old color to protect yet.
                manifests.append(
                    k8s_tools.build_service_manifest(
                        service_name, namespace, port, {k8s_tools.APP_LABEL: app_name, k8s_tools.VERSION_LABEL: deploy_color}
                    )
                )
            apply_result = await mcp.call_tool(
                "deployment", "apply_manifests", {"namespace": namespace, "manifests": manifests}
            )
            ok = apply_result["ok"]
            log_lines.append(f"apply ({deploy_color} environment): {apply_result['message']}")

        else:  # Canary
            deploy_target = f"{app_name}-canary"
            deployment = k8s_tools.build_deployment_manifest(
                deploy_target,
                namespace,
                image,
                port,
                replicas=1,
                strategy_type="RollingUpdate",
                extra_labels={k8s_tools.APP_LABEL: app_name, k8s_tools.VERSION_LABEL: "canary"},
            )
            manifests = [deployment]
            stable_exists = await mcp.call_tool(
                "deployment", "deployment_exists", {"namespace": namespace, "deployment_name": app_name}
            )
            if not stable_exists:
                # No stable version running yet — nothing for a canary to sit beside. The
                # Service still needs to exist so this "canary" is actually reachable.
                manifests.append(
                    k8s_tools.build_service_manifest(
                        service_name, namespace, port, {k8s_tools.APP_LABEL: app_name}
                    )
                )
            apply_result = await mcp.call_tool(
                "deployment", "apply_manifests", {"namespace": namespace, "manifests": manifests}
            )
            ok = apply_result["ok"]
            log_lines.append(f"apply (canary): {apply_result['message']}")
            if ok:
                # Same reasoning as the Recreate/Rolling branch above: force the selector
                # back to the plain, version-less form a canary needs (so it pools BOTH
                # the stable and canary pods for the replica-ratio traffic split) — a
                # prior Blue-Green promotion may have left a `version`-scoped selector
                # that a same-looking `apply` here would not detect as changed.
                patch_result = await mcp.call_tool(
                    "deployment",
                    "patch_service_selector",
                    {"namespace": namespace, "service_name": service_name, "selector": {k8s_tools.APP_LABEL: app_name}},
                )
                log_lines.append(f"service selector reset: {patch_result['message']}")

        return {
            "deploy_log": "\n".join(log_lines),
            "deploy_applied": ok,
            "k8s_deploy_target": deploy_target,
            "deploy_color": deploy_color,
        }
    except RuntimeError as exc:
        # A missing `kind`/`kubectl` binary on the kubernetes MCP server's side surfaces
        # here as a tool-call failure — same graceful-failure outcome as before, just
        # arriving through the MCP client instead of a local KubeToolMissingError.
        log_lines.append(str(exc))
        return {"deploy_log": "\n".join(log_lines), "deploy_applied": False}


def route_after_deploy(state: AgentState) -> str:
    return "validate" if state.get("deploy_applied") else "recovery"


async def validate(state: AgentState) -> dict:
    mcp = get_mcp_client()
    namespace = state["k8s_namespace"]
    target = state["k8s_deploy_target"]
    result = await mcp.call_tool("deployment", "rollout_status", {"namespace": namespace, "deployment_name": target})
    log = state.get("deploy_log", "") + f"\nrollout status ({target}): {result['message']}"
    return {"deploy_log": log, "validation_passed": result["ok"]}


def route_after_validate(state: AgentState) -> str:
    return "monitor" if state.get("validation_passed") else "recovery"


async def monitor(state: AgentState) -> dict:
    """A short, bounded poll for the failure mode rollout status alone can't see: a pod
    that reported Ready and then immediately started crash-looping. Real Kubernetes
    monitoring runs continuously — this is a deliberately small, honest slice of that
    (see the design doc's Section 3.10), not a claim of full production monitoring."""
    mcp = get_mcp_client()
    namespace = state["k8s_namespace"]
    app_name = state["k8s_app_name"]
    color_or_canary = state.get("deploy_color") or ("canary" if state["deployment_strategy"] == "Canary" else "")
    selector = f"{k8s_tools.APP_LABEL}={app_name}"
    if color_or_canary:
        selector += f",{k8s_tools.VERSION_LABEL}={color_or_canary}"

    before = await mcp.call_tool(
        "deployment", "get_pod_restart_total", {"namespace": namespace, "label_selector": selector}
    )
    await asyncio.sleep(15)
    after = await mcp.call_tool(
        "deployment", "get_pod_restart_total", {"namespace": namespace, "label_selector": selector}
    )

    crash_looping = after["total_restarts"] > before["total_restarts"]
    summary = (
        f"Monitored for 15s: {after['pod_statuses'] or 'no pods matched'} — "
        f"{after['total_restarts'] - before['total_restarts']} new restart(s)."
    )
    log = state.get("deploy_log", "") + f"\nmonitor: {summary}"
    return {"monitor_summary": summary, "deploy_log": log, "validation_passed": not crash_looping}


def route_after_monitor(state: AgentState) -> str:
    return "promote_and_finish" if state.get("validation_passed") else "recovery"


async def promote_and_finish(state: AgentState) -> dict:
    mcp = get_mcp_client()
    namespace = state["k8s_namespace"]
    app_name = state["k8s_app_name"]
    service_name = state["k8s_service_name"]
    strategy = state["deployment_strategy"]
    target = state["k8s_deploy_target"]
    image = state["docker_image_tag"]
    log_lines = [state.get("deploy_log", "")]

    if strategy == "Blue-Green":
        new_color = state.get("deploy_color", "")
        await mcp.call_tool(
            "deployment",
            "patch_service_selector",
            {
                "namespace": namespace,
                "service_name": service_name,
                "selector": {k8s_tools.APP_LABEL: app_name, k8s_tools.VERSION_LABEL: new_color},
            },
        )
        old_color = "blue" if new_color == "green" else "green"
        delete_result = await mcp.call_tool(
            "deployment", "delete_deployment", {"namespace": namespace, "deployment_name": f"{app_name}-{old_color}"}
        )
        log_lines.append(
            f"promoted {new_color}, switched traffic, removed old {old_color} environment ({delete_result['message']})"
        )
        # If this app's previous release used Rolling/Recreate (no color suffix) rather
        # than Blue-Green, that plain deployment is still running and now orphaned from
        # the Service (its pods carry no "version" label, so the freshly patched selector
        # no longer matches them). delete_deployment is a no-op if it doesn't exist, so
        # this is always safe to attempt.
        await mcp.call_tool("deployment", "delete_deployment", {"namespace": namespace, "deployment_name": app_name})

    elif strategy == "Canary":
        # target != app_name alone isn't enough to know a real, plain-named stable
        # Deployment exists to promote into — found by live-testing a Canary run right
        # after a Blue-Green promotion: the "stable" version was actually named
        # "<app>-green", so this used to try to `set image` on a Deployment named plain
        # "<app>" that didn't exist, then delete the canary regardless of whether that
        # succeeded — destroying real serving capacity (the canary's live pods) for a
        # promotion that never actually happened. Check for real instead.
        stable_exists = target != app_name and await mcp.call_tool(
            "deployment", "deployment_exists", {"namespace": namespace, "deployment_name": app_name}
        )
        if stable_exists:
            set_image_result = await mcp.call_tool(
                "deployment",
                "set_deployment_image",
                {"namespace": namespace, "deployment_name": app_name, "container_name": app_name, "image": image},
            )
            if set_image_result["ok"]:
                await mcp.call_tool(
                    "deployment", "rollout_status", {"namespace": namespace, "deployment_name": app_name}
                )
                delete_result = await mcp.call_tool(
                    "deployment", "delete_deployment", {"namespace": namespace, "deployment_name": target}
                )
                log_lines.append(f"promoted canary to stable, removed canary deployment ({delete_result['message']})")
            else:
                log_lines.append(
                    f"couldn't promote canary to the stable deployment ({set_image_result['message']}) — "
                    "leaving the canary running rather than deleting it for a promotion that didn't happen"
                )
        elif target == app_name:
            log_lines.append("canary deployment was this app's first-ever version — nothing further to promote")
        else:
            log_lines.append(
                "no plain-named stable deployment to promote into (this app's stable version is likely "
                "running under a Blue-Green name) — leaving the canary running as-is; a known gap in "
                "mixing strategies across releases, see the Phase 4 build-plan notes"
            )

    pr_url = state.get("pr_url") or state.get("branch_name", "")
    reply = (
        f"Deployed successfully using the {strategy} strategy.\n\n"
        f"{state.get('monitor_summary', '')}\n\n"
        f"Namespace: {namespace}\n"
        f"Reach it locally: kubectl port-forward -n {namespace} service/{service_name} 8080:{k8s_tools.detect_container_port(state['workspace_path'])}\n\n"
        f"PR: {pr_url}"
    )
    return {"deploy_log": "\n".join(log_lines), "reply": reply, "outcome": "deployed"}


async def recovery(state: AgentState) -> dict:
    mcp = get_mcp_client()
    namespace = state["k8s_namespace"]
    app_name = state["k8s_app_name"]
    selector = f"{k8s_tools.APP_LABEL}={app_name}"

    logs = await mcp.call_tool("recovery", "pod_logs", {"namespace": namespace, "label_selector": selector})
    describe = await mcp.call_tool("recovery", "describe_pods", {"namespace": namespace, "label_selector": selector})

    llm = get_llm_provider()
    prompt = (
        "A Kubernetes deployment just failed validation or started crash-looping shortly after. "
        "Read the evidence below and give a short, plain-English root-cause analysis (2-4 sentences) "
        "a developer can act on immediately. Don't pad it with generic advice.\n\n"
        f"Deployment log so far:\n{state.get('deploy_log', '')}\n\n"
        f"Pod logs (tail):\n{logs[-3000:] or '(no logs captured)'}\n\n"
        f"kubectl describe pods (tail):\n{describe[-3000:] or '(not available)'}"
    )
    response = await llm.chat([LLMMessage(role="user", content=prompt)])
    return {"recovery_analysis": response.content.strip()}


def await_rollback_approval(state: AgentState) -> dict:
    message = (
        f"Deployment failed validation.\n\n"
        f"Root cause: {state.get('recovery_analysis', '(analysis unavailable)')}\n\n"
        f"{state.get('monitor_summary', '')}\n\n"
        "Reply 'approve' to roll back, or anything else to leave the failed deployment as-is for you to inspect."
    )
    response = interrupt({"message": message})
    return {"approval_response": response}


def route_after_rollback_approval(state: AgentState) -> str:
    return "rollback" if is_affirmative(state.get("approval_response", "")) else "declined_at_rollback"


async def rollback(state: AgentState) -> dict:
    mcp = get_mcp_client()
    namespace = state["k8s_namespace"]
    app_name = state["k8s_app_name"]
    strategy = state["deployment_strategy"]
    target = state.get("k8s_deploy_target", "") or app_name

    if strategy in ("Recreate", "Rolling"):
        result = await mcp.call_tool("recovery", "rollout_undo", {"namespace": namespace, "deployment_name": app_name})
        action = f"Rolled back {app_name} to its previous version ({result['message']})"
    else:
        # Blue-Green and Canary never switched live traffic to the failed version in the
        # first place — rolling back just means deleting the failed side, the stable
        # version (or, for Blue-Green, the untouched Service) was never at risk.
        result = await mcp.call_tool(
            "recovery", "delete_deployment", {"namespace": namespace, "deployment_name": target}
        )
        action = f"Removed the failed {target} deployment; the previous stable version was never touched ({result['message']})"

    pr_url = state.get("pr_url") or state.get("branch_name", "")
    reply = f"{action}.\n\nRoot cause: {state.get('recovery_analysis', '')}\n\nPR: {pr_url}"
    return {"reply": reply, "outcome": "rolled_back"}


def declined_at_rollback(state: AgentState) -> dict:
    pr_url = state.get("pr_url") or state.get("branch_name", "")
    reply = (
        "Okay — leaving the failed deployment as it is for you to inspect directly.\n\n"
        f"Root cause: {state.get('recovery_analysis', '')}\n\n"
        f"Namespace: {state.get('k8s_namespace', '')}\n"
        f"PR: {pr_url}"
    )
    return {"reply": reply, "outcome": "declined_at_rollback"}


def build_graph(checkpointer=None):
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
    graph.add_node("select_deployment_strategy", select_deployment_strategy)
    graph.add_node("await_deploy_approval", await_deploy_approval)
    graph.add_node("declined_at_deploy", declined_at_deploy)
    graph.add_node("deploy", deploy)
    graph.add_node("validate", validate)
    graph.add_node("monitor", monitor)
    graph.add_node("promote_and_finish", promote_and_finish)
    graph.add_node("recovery", recovery)
    graph.add_node("await_rollback_approval", await_rollback_approval)
    graph.add_node("declined_at_rollback", declined_at_rollback)
    graph.add_node("rollback", rollback)

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

    graph.add_conditional_edges(
        "build_docker_image",
        route_after_build_docker_image,
        {"select_deployment_strategy": "select_deployment_strategy", "end": END},
    )
    graph.add_edge("select_deployment_strategy", "await_deploy_approval")
    graph.add_conditional_edges(
        "await_deploy_approval",
        route_after_deploy_approval,
        {"deploy": "deploy", "declined_at_deploy": "declined_at_deploy"},
    )
    graph.add_edge("declined_at_deploy", END)

    graph.add_conditional_edges("deploy", route_after_deploy, {"validate": "validate", "recovery": "recovery"})
    graph.add_conditional_edges("validate", route_after_validate, {"monitor": "monitor", "recovery": "recovery"})
    graph.add_conditional_edges(
        "monitor", route_after_monitor, {"promote_and_finish": "promote_and_finish", "recovery": "recovery"}
    )
    graph.add_edge("promote_and_finish", END)

    graph.add_edge("recovery", "await_rollback_approval")
    graph.add_conditional_edges(
        "await_rollback_approval",
        route_after_rollback_approval,
        {"rollback": "rollback", "declined_at_rollback": "declined_at_rollback"},
    )
    graph.add_edge("declined_at_rollback", END)
    graph.add_edge("rollback", END)

    # MemorySaver is a fallback for callers that don't pass a real checkpointer (tests,
    # ad-hoc scripts). The FastAPI app always passes a persistent one — see main.py's
    # lifespan — since MemorySaver's whole state is lost on every process restart.
    return graph.compile(checkpointer=checkpointer or MemorySaver())
