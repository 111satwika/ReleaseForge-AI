"""Unit tests for app/security/scopes.py — the least-privilege tool scoping table."""

import pytest

from app.security.scopes import ROLE_SCOPES, ScopeDeniedError, check_scope


def test_coding_role_can_read_and_write_files_but_not_touch_git_or_docker():
    check_scope("coding", "read_file")
    check_scope("coding", "write_file")
    check_scope("coding", "list_files")
    with pytest.raises(ScopeDeniedError):
        check_scope("coding", "commit_and_push")
    with pytest.raises(ScopeDeniedError):
        check_scope("coding", "run_docker_build")


def test_publish_role_can_push_and_open_prs_but_not_edit_files():
    check_scope("publish", "commit_and_push")
    check_scope("publish", "open_pull_request")
    with pytest.raises(ScopeDeniedError):
        check_scope("publish", "write_file")


def test_deployment_role_can_deploy_but_not_delete_via_recovery_only_tools():
    check_scope("deployment", "apply_manifests")
    check_scope("deployment", "patch_service_selector")
    check_scope("deployment", "delete_deployment")  # needed for its own post-promotion cleanup
    with pytest.raises(ScopeDeniedError):
        check_scope("deployment", "pod_logs")  # a Recovery-only diagnostic tool


def test_recovery_role_can_inspect_and_roll_back_but_cannot_apply_new_manifests():
    check_scope("recovery", "pod_logs")
    check_scope("recovery", "describe_pods")
    check_scope("recovery", "rollout_undo")
    check_scope("recovery", "delete_deployment")
    with pytest.raises(ScopeDeniedError):
        check_scope("recovery", "apply_manifests")
    with pytest.raises(ScopeDeniedError):
        check_scope("recovery", "patch_service_selector")


def test_unknown_role_fails_closed_not_open():
    with pytest.raises(ScopeDeniedError):
        check_scope("some-made-up-role", "read_file")


def test_scope_denied_error_carries_role_and_tool_for_the_audit_log():
    with pytest.raises(ScopeDeniedError) as exc_info:
        check_scope("coding", "delete_deployment")
    assert exc_info.value.role == "coding"
    assert exc_info.value.tool_name == "delete_deployment"


def test_every_role_scope_is_a_non_empty_set():
    """A role with an empty scope would be indistinguishable from a typo — every real
    role defined should actually be able to do at least one thing."""
    for role, tools in ROLE_SCOPES.items():
        assert tools, f"role '{role}' has an empty scope"


def test_no_role_can_reach_every_tool():
    """A sanity check on the whole point of this module: least privilege means no single
    role should have blanket access to everything MCP exposes."""
    all_tools = set().union(*ROLE_SCOPES.values())
    for role, tools in ROLE_SCOPES.items():
        assert tools != all_tools, f"role '{role}' has unrestricted access to every tool"
