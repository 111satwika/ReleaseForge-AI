"""Tests for the pure, no-subprocess parts of k8s_tools: manifest generation, name
sanitizing, port detection, and strategy selection. The live cluster interaction
(ensure_kind_cluster, apply_manifests, rollout_status, ...) is exercised separately,
against a real `kind` cluster — see the manual verification in the Phase 4 build-plan
entry rather than here, since a unit test suite shouldn't depend on a real cluster
existing to pass."""

import json
import subprocess

import app.agents.k8s_tools as k8s_tools_module
from app.agents.k8s_tools import (
    APP_LABEL,
    VERSION_LABEL,
    build_deployment_manifest,
    build_service_manifest,
    decide_deployment_strategy,
    detect_container_port,
    patch_service_selector,
    manifests_to_yaml,
    sanitize_dns_label,
)


def test_sanitize_dns_label_lowercases_and_strips_invalid_chars():
    assert sanitize_dns_label("My Cool App!!") == "my-cool-app"


def test_sanitize_dns_label_strips_leading_trailing_hyphens():
    assert sanitize_dns_label("--weird__name--") == "weird-name" or sanitize_dns_label("--weird__name--").strip("-")


def test_sanitize_dns_label_truncates_to_63_chars():
    assert len(sanitize_dns_label("a" * 100)) <= 63


def test_sanitize_dns_label_never_empty():
    assert sanitize_dns_label("!!!") == "app"


def test_detect_container_port_reads_expose_line(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\nEXPOSE 5000\nCMD [\"run\"]\n")
    assert detect_container_port(str(tmp_path)) == 5000


def test_detect_container_port_falls_back_without_dockerfile(tmp_path):
    assert detect_container_port(str(tmp_path)) == 8080


def test_detect_container_port_falls_back_without_expose(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\nCMD [\"run\"]\n")
    assert detect_container_port(str(tmp_path)) == 8080


def test_build_deployment_manifest_sets_strategy_type():
    manifest = build_deployment_manifest("app", "ns", "img:tag", 8080, replicas=2, strategy_type="Recreate")
    assert manifest["spec"]["strategy"] == {"type": "Recreate"}
    assert manifest["metadata"]["namespace"] == "ns"
    assert manifest["spec"]["replicas"] == 2
    assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == "img:tag"


def test_build_deployment_manifest_includes_extra_labels_in_selector():
    manifest = build_deployment_manifest(
        "app-green", "ns", "img:tag", 8080, replicas=1, strategy_type="RollingUpdate",
        extra_labels={VERSION_LABEL: "green"},
    )
    labels = manifest["spec"]["selector"]["matchLabels"]
    assert labels == {APP_LABEL: "app-green", VERSION_LABEL: "green"}


def test_extra_labels_can_override_the_app_label_to_the_shared_app_name():
    """Regression test for a real bug caught by live verification against a `kind`
    cluster: graph.py's Blue-Green/Canary deploy() calls pass a resource name that
    differs from the shared app name (e.g. "myapp-green", "myapp-canary") as the
    manifest's `name`. If the `app` label silently followed that resource name instead
    of the shared app name, the Service selector (which always targets `app=<app_name>`)
    would match zero pods after promotion — deployed "successfully" while serving no
    traffic at all. Callers must be able to pass APP_LABEL inside extra_labels to force
    it back to the shared app name; this locks in that extra_labels always wins."""
    manifest = build_deployment_manifest(
        "myapp-green", "ns", "img:tag", 8080, replicas=1, strategy_type="RollingUpdate",
        extra_labels={APP_LABEL: "myapp", VERSION_LABEL: "green"},
    )
    labels = manifest["metadata"]["labels"]
    assert labels[APP_LABEL] == "myapp"
    assert manifest["spec"]["selector"]["matchLabels"][APP_LABEL] == "myapp"
    assert manifest["spec"]["template"]["metadata"]["labels"][APP_LABEL] == "myapp"


def test_build_service_manifest_targets_the_right_port():
    manifest = build_service_manifest("svc", "ns", 9000, {APP_LABEL: "app"})
    assert manifest["spec"]["ports"] == [{"port": 9000, "targetPort": 9000}]
    assert manifest["spec"]["selector"] == {APP_LABEL: "app"}


def test_manifests_to_yaml_produces_valid_multi_document_yaml():
    import yaml

    deployment = build_deployment_manifest("app", "ns", "img", 8080, 1, "Recreate")
    service = build_service_manifest("app", "ns", 8080, {APP_LABEL: "app"})
    text = manifests_to_yaml([deployment, service])
    docs = list(yaml.safe_load_all(text))
    assert len(docs) == 2
    assert docs[0]["kind"] == "Deployment"
    assert docs[1]["kind"] == "Service"


# ---------- strategy selection ----------


def test_first_deployment_always_recreate_regardless_of_risk():
    strategy, reason = decide_deployment_strategy(has_existing_deployment=False, risk="High", readiness_score=95)
    assert strategy == "Recreate"
    assert "first deployment" in reason.lower()


def test_high_risk_existing_app_gets_blue_green():
    strategy, _ = decide_deployment_strategy(has_existing_deployment=True, risk="High", readiness_score=80)
    assert strategy == "Blue-Green"


def test_medium_risk_existing_app_gets_canary():
    strategy, _ = decide_deployment_strategy(has_existing_deployment=True, risk="Medium", readiness_score=85)
    assert strategy == "Canary"


def test_low_risk_existing_app_gets_rolling():
    strategy, _ = decide_deployment_strategy(has_existing_deployment=True, risk="Low", readiness_score=100)
    assert strategy == "Rolling"


def test_unknown_risk_existing_app_defaults_to_rolling_not_a_crash():
    strategy, _ = decide_deployment_strategy(has_existing_deployment=True, risk="Unknown", readiness_score=70)
    assert strategy == "Rolling"


# ---------- Service selector patching ----------


def test_patch_service_selector_uses_json_replace_not_merge_patch(monkeypatch):
    """Regression test for a real bug caught by live verification: a `--type=merge`
    patch recursively merges nested objects, so patching selector {"app": "x"} onto a
    live selector of {"app": "x", "version": "green"} left "version" sitting there
    untouched — a Canary deploy right after a Blue-Green promotion kept routing only to
    the promoted color, silently, because the Service selector was never actually reset.
    A JSON Patch "replace" op swaps the field wholesale, which is the only way to
    guarantee the selector becomes exactly the given dict."""
    captured = {}

    def fake_run(args, timeout, input_text=None):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="service/x patched\n", stderr="")

    monkeypatch.setattr(k8s_tools_module, "_run", fake_run)
    monkeypatch.setattr(k8s_tools_module.shutil, "which", lambda name: "/usr/bin/kubectl")

    patch_service_selector("ns", "svc", {"app": "myapp"})

    args = captured["args"]
    assert "--type=json" in args
    assert "--type=merge" not in args
    patch_arg = args[args.index("-p") + 1]
    patch_body = json.loads(patch_arg)
    assert patch_body == [{"op": "replace", "path": "/spec/selector", "value": {"app": "myapp"}}]
