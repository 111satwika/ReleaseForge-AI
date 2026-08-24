"""Tests for the fast, deterministic paths of app/agents/tools.py's two newest
functions. The real tool invocations (pip-audit, bandit) are deliberately NOT run here —
pip-audit alone took ~90s live against this project's own requirements.txt (a real
network call to a vulnerability database per dependency), which would make this
otherwise-fast suite noticeably slower for every future test run. That real behavior was
verified live instead and is documented in RELEASEFORGE_BUILD_PLAN.md, the same "fast
unit tests for logic, live verification for the real tool" split run_docker_build's own
tests already use (only its missing-Dockerfile path is unit-tested, not a real build)."""

import app.agents.tools as tools_module
from app.agents.tools import check_dependencies, remove_docker_image, run_security_scan


def test_check_dependencies_with_no_manifest_reports_not_applicable(tmp_path):
    result = check_dependencies(str(tmp_path))
    assert result["ran"] is False
    assert result["tool"] is None
    assert result["vulnerabilities_found"] == 0
    assert "No requirements.txt or package.json" in result["detail"]


def test_run_security_scan_with_no_python_files_reports_not_applicable(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>")
    result = run_security_scan(str(tmp_path))
    assert result["ran"] is False
    assert result["tool"] == "bandit"
    assert result["findings"] == 0
    assert "No Python files found to scan" in result["detail"]


def test_remove_docker_image_reports_cleanly_when_docker_is_missing(monkeypatch):
    monkeypatch.setattr(tools_module.shutil, "which", lambda name: None)
    ok, detail = remove_docker_image("some-tag:test")
    assert ok is False
    assert "Docker isn't installed" in detail
