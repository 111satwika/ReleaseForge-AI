"""Per-agent tool scoping — the design doc's Section 14 "Agents should have only the
tools they need" table, made real. This project has no multi-user auth system yet (it's
a single local backend behind a VS Code extension, not a multi-tenant service), so
OAuth/OIDC/user-level RBAC would be security theater — there's no second user to scope
away from. What genuinely applies here, and is genuinely enforceable, is restricting
which of the 26 MCP tools each *agent role* can invoke, regardless of who's asking:
a Coding Agent has no legitimate reason to ever call `delete_deployment`, so it can't,
even if a prompt-injected file somehow tried to talk it into requesting that tool.

Enforced client-side, in MCPClientManager.call_tool(), not inside the MCP servers
themselves: these servers only ever talk to this one trusted local client over stdio
pipes, never over a network a different, untrusted client could reach — the "network
isolation" half of Section 14 is already true by construction here, so a second
server-side auth layer would just be checking the same thing twice.
"""

FILESYSTEM_READ_WRITE = {"list_files", "read_file", "write_file"}
FILESYSTEM_TEST = {"detect_test_command", "run_tests"}
FILESYSTEM_GIT = {"current_branch", "commit_and_push", "push_fix_commit", "open_pull_request"}
DOCKER_BUILD = {"run_docker_build"}
KUBERNETES_DEPLOY = {
    "ensure_kind_cluster",
    "kind_load_image",
    "ensure_namespace",
    "apply_manifests",
    "rollout_status",
    "deployment_exists",
    "get_pod_restart_total",
    "get_service_selector",
    "patch_service_selector",
    "set_deployment_image",
    # Also needed here, not just by Recovery: promote_and_finish() legitimately deletes
    # the old color (Blue-Green) or the now-promoted canary as ordinary cleanup after a
    # *successful* deploy — a different purpose than Recovery's "delete the failed
    # side," but the same underlying tool. Least privilege means each role gets what it
    # actually needs, not that a tool must belong to exactly one role.
    "delete_deployment",
}
KUBERNETES_RECOVER = {"pod_logs", "describe_pods", "rollout_undo", "delete_deployment"}

# Mirrors the design doc's table (Coding Agent -> GitHub + Filesystem + Terminal, etc.)
# adapted to this project's actual graph nodes rather than the doc's more abstract
# agent names — "PR Agent" here is the publish() node, "Deployment Agent" covers
# select_deployment_strategy/deploy/validate/monitor/promote_and_finish, and so on.
ROLE_SCOPES: dict[str, set[str]] = {
    "coding": FILESYSTEM_READ_WRITE,
    "test": FILESYSTEM_TEST,
    "publish": FILESYSTEM_GIT,
    "build": FILESYSTEM_READ_WRITE | {"has_dockerfile", "repo_name", "push_fix_commit"} | DOCKER_BUILD,
    "deployment": {"repo_name"} | KUBERNETES_DEPLOY,
    # Deliberately does NOT include KUBERNETES_DEPLOY — a Recovery Agent can inspect
    # (logs, describe) and roll back or delete a failed deployment, but has no business
    # applying new manifests or patching a Service's live selector; that's the
    # Deployment Agent's job, and giving Recovery the same power would defeat the point
    # of a separate, narrower role for the failure path.
    "recovery": KUBERNETES_RECOVER,
}


class ScopeDeniedError(PermissionError):
    def __init__(self, role: str, tool_name: str):
        self.role = role
        self.tool_name = tool_name
        super().__init__(f"Role '{role}' is not permitted to call tool '{tool_name}' (least-privilege scoping).")


def check_scope(role: str, tool_name: str) -> None:
    allowed = ROLE_SCOPES.get(role)
    if allowed is None:
        raise ScopeDeniedError(role, tool_name)  # unknown role — fail closed, not open
    if tool_name not in allowed:
        raise ScopeDeniedError(role, tool_name)
