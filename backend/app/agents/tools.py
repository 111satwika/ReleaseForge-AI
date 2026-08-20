import json
import os
import re
import shutil
import subprocess
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "out", "dist", "build"}


class UnsafePathError(Exception):
    pass


def _resolve_safe_path(workspace_path: str, relative_path: str) -> Path:
    base = Path(workspace_path).resolve()
    target = (base / relative_path).resolve()
    if target != base and base not in target.parents:
        raise UnsafePathError(f"'{relative_path}' resolves outside the workspace")
    return target


def list_files(workspace_path: str, limit: int = 60) -> list[str]:
    """Top-level-ish repo listing so the Planner has something to look at."""
    base = Path(workspace_path)
    entries: list[str] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), base).replace("\\", "/")
            entries.append(rel)
            if len(entries) >= limit:
                return entries
    return entries


def read_file(workspace_path: str, relative_path: str) -> str | None:
    path = _resolve_safe_path(workspace_path, relative_path)
    if not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def write_file(workspace_path: str, relative_path: str, content: str) -> None:
    path = _resolve_safe_path(workspace_path, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def detect_test_command(workspace_path: str) -> list[str] | None:
    base = Path(workspace_path)
    if (base / "pytest.ini").exists() or (base / "pyproject.toml").exists() or (base / "setup.cfg").exists():
        return ["pytest", "-q"]
    if (base / "manage.py").exists():
        return ["python", "manage.py", "test"]
    package_json = base / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            if "test" in data.get("scripts", {}):
                return ["npm", "test", "--silent"]
        except (json.JSONDecodeError, OSError):
            pass
    return None


def run_tests(workspace_path: str, timeout: int = 90) -> tuple[bool | None, str]:
    command = detect_test_command(workspace_path)
    if command is None:
        return None, "No recognized test command found (checked pytest/pyproject.toml, manage.py, package.json)."

    try:
        result = subprocess.run(
            command,
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output[-4000:]
    except FileNotFoundError:
        return None, f"Test command not found on PATH: {' '.join(command)}"
    except subprocess.TimeoutExpired:
        return False, f"Tests timed out after {timeout}s: {' '.join(command)}"


def _run_git(workspace_path: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=workspace_path, capture_output=True, text=True)


def current_branch(workspace_path: str) -> str:
    result = _run_git(workspace_path, ["branch", "--show-current"])
    return result.stdout.strip() or "main"


def commit_and_push(
    workspace_path: str, branch_name: str, files: list[str], commit_message: str
) -> tuple[bool, str]:
    log: list[str] = []
    for step in (
        ["checkout", "-b", branch_name],
        ["add", *files],
        ["commit", "-m", commit_message],
        ["push", "-u", "origin", branch_name],
    ):
        result = _run_git(workspace_path, step)
        log.append(f"$ git {' '.join(step)}\n{result.stdout}{result.stderr}".strip())
        if result.returncode != 0:
            return False, "\n\n".join(log)
    return True, "\n\n".join(log)


def push_fix_commit(workspace_path: str, files: list[str], commit_message: str) -> tuple[bool, str]:
    """Adds a follow-up commit to whatever branch is already checked out (the one an
    earlier commit_and_push created) and pushes it — for updating an already-open PR."""
    log: list[str] = []
    for step in (["add", *files], ["commit", "-m", commit_message], ["push"]):
        result = _run_git(workspace_path, step)
        log.append(f"$ git {' '.join(step)}\n{result.stdout}{result.stderr}".strip())
        if result.returncode != 0:
            return False, "\n\n".join(log)
    return True, "\n\n".join(log)


def _parse_github_owner_repo(remote_url: str) -> str | None:
    match = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(\.git)?$", remote_url.strip())
    return match.group(1) if match else None


def open_pull_request(
    workspace_path: str, branch_name: str, base_branch: str, title: str, body: str
) -> tuple[bool, str]:
    """Prefers `gh pr create`; falls back to a GitHub compare URL if the CLI isn't installed."""
    if shutil.which("gh") is not None:
        result = subprocess.run(
            ["gh", "pr", "create", "--title", title, "--body", body, "--base", base_branch, "--head", branch_name],
            cwd=workspace_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stdout + result.stderr).strip()

    remote = _run_git(workspace_path, ["remote", "get-url", "origin"]).stdout.strip()
    owner_repo = _parse_github_owner_repo(remote)
    if owner_repo:
        compare_url = f"https://github.com/{owner_repo}/compare/{base_branch}...{branch_name}?expand=1"
        return True, compare_url

    return False, "Branch pushed, but no `gh` CLI and origin isn't a github.com remote — open the PR manually."
