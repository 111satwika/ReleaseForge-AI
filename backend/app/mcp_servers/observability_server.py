"""MCP server exposing Resources and Prompts — the two MCP primitives this project's
own build plan named as "not yet" built (only Tools were ever actually needed for the
filesystem/docker/kubernetes servers' real workflows). Kept as its own server, same
least-privilege reasoning Phase 7 already used for splitting docker_server.py out from
filesystem_server.py: this is read-only observability data and prompt templates, not an
action on an external system, so it has no reason to share a process with anything that
touches git, Docker, or a cluster.

Resources here are genuinely backed by this project's real data (GET /eval/summary,
GET /audit/log, GET /incidents already expose the same information over HTTP — this is
the same data, reachable through the MCP protocol instead). Scope is honest, not
oversold: these are additive and independently live-verified, but graph.py's own
internal LLM calls still build their prompts directly as f-strings rather than being
rewired to fetch them through prompts/get — that would be a much larger, riskier change
to logic this project has already live-tested extensively, for no functional benefit
over what's built here."""

import json

from mcp.server.mcpserver import MCPServer

from app.eval.metrics import evaluate_platform
from app.memory import repository as memory
from app.security.audit import recent_audit_events

server = MCPServer(
    name="releaseforge-observability",
    instructions="Read-only platform evaluation data (resources) and reusable prompt templates (prompts).",
)


@server.resource("releaseforge://eval-summary")
async def eval_summary_resource() -> str:
    """Same computation GET /eval/summary exposes over HTTP — platform-wide (every
    workspace this backend has ever worked on), not scoped to one repo."""
    from dataclasses import asdict

    report = await evaluate_platform(None)
    return json.dumps(
        {
            "records_evaluated": report["records_evaluated"],
            "agent": asdict(report["agent"]),
            "coding": asdict(report["coding"]),
            "deployment": asdict(report["deployment"]),
        }
    )


@server.resource("releaseforge://audit-log")
async def audit_log_resource() -> str:
    """Same data GET /audit/log exposes — every MCP tool call this backend has attempted,
    successful or not, across every workspace."""
    events = await recent_audit_events(limit=100, workspace_path=None)
    return json.dumps(
        [
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
    )


@server.resource("releaseforge://incidents")
async def incidents_resource() -> str:
    """Same data GET /incidents exposes — every rollback's real root-cause analysis,
    across every workspace, not just the most recent handful for one repo."""
    incidents = await memory.all_incidents(None)
    return json.dumps(
        [
            {
                "id": i.id,
                "workspace_path": i.workspace_path,
                "release_id": i.release_id,
                "deployment_strategy": i.deployment_strategy,
                "action": i.action,
                "root_cause": i.root_cause,
                "created_at": i.created_at.isoformat(),
            }
            for i in incidents
        ]
    )


@server.prompt()
def classify_intent_prompt(user_message: str) -> str:
    """The real prompt graph.py's classify_intent() node sends to the LLM, exposed as a
    reusable MCP Prompt template rather than only ever existing inline as an f-string —
    kept in exact sync with the wording graph.py actually uses; if that prompt is ever
    edited, this docstring is where to remember to check."""
    return (
        "Classify the user's message as exactly one word: TASK or CHAT.\n"
        "TASK = they want you to inspect, add, change, fix, or test something in their repository.\n"
        "CHAT = anything else (questions, greetings, general conversation).\n\n"
        f"Message: {user_message!r}\n\n"
        "Reply with exactly one word: TASK or CHAT."
    )


@server.prompt()
def recovery_analysis_prompt(deploy_log: str, pod_logs: str, describe_pods: str) -> str:
    """The real prompt graph.py's recovery() node sends to the LLM after a deployment
    fails validation or starts crash-looping — same wording, exposed as a reusable
    template."""
    return (
        "A Kubernetes deployment just failed validation or started crash-looping shortly after. "
        "Read the evidence below and give a short, plain-English root-cause analysis (2-4 sentences) "
        "a developer can act on immediately. Don't pad it with generic advice.\n\n"
        f"Deployment log so far:\n{deploy_log}\n\n"
        f"Pod logs (tail):\n{pod_logs[-3000:] or '(no logs captured)'}\n\n"
        f"kubectl describe pods (tail):\n{describe_pods[-3000:] or '(not available)'}"
    )


if __name__ == "__main__":
    server.run()
