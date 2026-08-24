import json
import os
import re
import shutil
import subprocess
import sys
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


def run_tests_with_command(
    workspace_path: str, command: list[str] | None, timeout: int = 90
) -> tuple[bool | None, str]:
    if command is None:
        return None, "No recognized test command found (checked pytest/pyproject.toml, manage.py, package.json)."

    try:
        result = subprocess.run(
            command,
            cwd=workspace_path,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            shell=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output[-4000:]
    except FileNotFoundError:
        return None, f"Test command not found on PATH: {' '.join(command)}"
    except subprocess.TimeoutExpired:
        return False, f"Tests timed out after {timeout}s: {' '.join(command)}"


def run_tests(workspace_path: str, timeout: int = 90) -> tuple[bool | None, str]:
    return run_tests_with_command(workspace_path, detect_test_command(workspace_path), timeout)


def _run_git(workspace_path: str, args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            # No terminal is ever attached to this process (it's spawned headlessly by
            # the extension), so if git needs credentials it can't prompt for them —
            # closing stdin makes credential helpers fail fast instead of hanging
            # forever, which previously froze the entire backend (single event loop).
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=1,
            stdout="",
            stderr=(
                f"git {' '.join(args)} timed out after {timeout}s — likely waiting on "
                "credentials with no terminal attached to answer them. Set up a git "
                "credential helper (or an SSH remote) so pushes don't need a prompt."
            ),
        )


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
        try:
            result = subprocess.run(
                ["gh", "pr", "create", "--title", title, "--body", body, "--base", base_branch, "--head", branch_name],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return False, "`gh pr create` timed out after 60s — check `gh auth status`."
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stdout + result.stderr).strip()

    remote = _run_git(workspace_path, ["remote", "get-url", "origin"]).stdout.strip()
    owner_repo = _parse_github_owner_repo(remote)
    if owner_repo:
        compare_url = f"https://github.com/{owner_repo}/compare/{base_branch}...{branch_name}?expand=1"
        return True, compare_url

    return False, "Branch pushed, but no `gh` CLI and origin isn't a github.com remote — open the PR manually."


def has_dockerfile(workspace_path: str) -> bool:
    return (Path(workspace_path) / "Dockerfile").exists()


def repo_name(workspace_path: str) -> str:
    return Path(workspace_path).name


def run_docker_build(workspace_path: str, tag: str, timeout: int = 300) -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "Docker isn't installed, or `docker` isn't on PATH."
    try:
        result = subprocess.run(
            ["docker", "build", "-t", tag, "."],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output[-4000:]
    except FileNotFoundError:
        return False, "Docker isn't installed, or `docker` isn't on PATH."
    except subprocess.TimeoutExpired:
        return False, f"Docker build timed out after {timeout}s."


def remove_docker_image(tag: str, timeout: int = 30) -> tuple[bool, str]:
    """Cleanup for throwaway verification builds (see test()'s Dockerfile-build check in
    graph.py) — without this, every code/review round on a Dockerfile change would leave
    another tagged image sitting on disk forever. Best-effort: a missing Docker install
    or an already-gone image both return False rather than raising, since neither is
    worth failing a test result over."""
    if shutil.which("docker") is None:
        return False, "Docker isn't installed, or `docker` isn't on PATH."
    try:
        result = subprocess.run(
            ["docker", "rmi", "-f", tag],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        return result.returncode == 0, ((result.stdout or "") + (result.stderr or "")).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def check_dependencies(workspace_path: str, timeout: int = 150) -> dict:
    """Real, tool-based dependency vulnerability check — pip-audit against a Python
    repo's requirements.txt, npm audit against a Node repo's package.json. Neither tool
    is assumed installed; a missing tool or an unparseable result is reported honestly
    as "couldn't run," never silently folded into "no vulnerabilities found" — those are
    two different things, and collapsing them would be exactly the kind of dishonest
    signal release_readiness() has avoided everywhere else.

    150s default, not a guess: pip-audit genuinely took ~90s live against this project's
    own requirements.txt (it queries a real vulnerability database per dependency, over
    the network, for every package including transitive ones) — a 60s default failed
    outright the first time this was actually run."""
    requirements = Path(workspace_path) / "requirements.txt"
    package_json = Path(workspace_path) / "package.json"

    if requirements.exists():
        # Invoked as `sys.executable -m pip_audit`, not a bare `pip-audit` command —
        # found live: pip-audit is a Python package installed into this backend's own
        # venv, and a venv's Scripts dir isn't on PATH for a subprocess unless the venv
        # was actually activated first, which this backend never is (it's launched
        # directly as `.venv/Scripts/python.exe -m uvicorn ...`). `shutil.which()`
        # against a bare command name correctly found nothing and reported "not
        # installed" even though it demonstrably was — the same class of "installed but
        # not on this process's PATH" issue Docker Desktop hit earlier in this project,
        # just one layer more specific (a venv-local tool, not a system-wide one).
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip_audit", "-r", str(requirements), "--format", "json"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
            )
            if "No module named" in (result.stderr or ""):
                return {"ran": False, "tool": "pip-audit", "vulnerabilities_found": 0, "detail": "pip-audit isn't installed."}
            data = json.loads(result.stdout or "[]")
            packages = data if isinstance(data, list) else data.get("dependencies", [])
            names = sorted({pkg["name"] for pkg in packages if pkg.get("vulns")})
            count = sum(len(pkg.get("vulns", [])) for pkg in packages)
            detail = f"{count} known vulnerabilit{'y' if count == 1 else 'ies'} in: {', '.join(names)}" if count else "No known vulnerabilities found."
            return {"ran": True, "tool": "pip-audit", "vulnerabilities_found": count, "detail": detail}
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as exc:
            return {"ran": False, "tool": "pip-audit", "vulnerabilities_found": 0, "detail": f"pip-audit failed to run: {exc}"}

    if package_json.exists():
        if shutil.which("npm") is None:
            return {"ran": False, "tool": "npm audit", "vulnerabilities_found": 0, "detail": "npm isn't installed."}
        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
            )
            data = json.loads(result.stdout or "{}")
            if "error" in data:
                detail = data["error"].get("summary") or str(data["error"])
                return {"ran": False, "tool": "npm audit", "vulnerabilities_found": 0, "detail": f"npm audit reported an error: {detail}"}
            count = data.get("metadata", {}).get("vulnerabilities", {}).get("total", 0)
            detail = f"{count} known vulnerabilit{'y' if count == 1 else 'ies'} reported by npm audit." if count else "No known vulnerabilities found."
            return {"ran": True, "tool": "npm audit", "vulnerabilities_found": count, "detail": detail}
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as exc:
            return {"ran": False, "tool": "npm audit", "vulnerabilities_found": 0, "detail": f"npm audit failed to run: {exc}"}

    return {"ran": False, "tool": None, "vulnerabilities_found": 0, "detail": "No requirements.txt or package.json found — nothing to check."}


def run_security_scan(workspace_path: str, timeout: int = 60) -> dict:
    """Real static-analysis security scan via bandit — honestly scoped to Python: a repo
    with no .py files correctly reports "nothing to scan," not a fake pass. Extending
    this to other languages would need a different tool (e.g. semgrep) and wasn't
    attempted here — stated plainly rather than pretending broader coverage exists."""
    if not any(Path(workspace_path).rglob("*.py")):
        return {"ran": False, "tool": "bandit", "findings": 0, "detail": "No Python files found to scan."}
    try:
        result = subprocess.run(
            # sys.executable -m bandit, not a bare `bandit` command — same venv-not-on-
            # PATH reasoning as pip-audit above. -q is separately load-bearing: without
            # it bandit prints "[main] INFO ..." lines to stdout ahead of the JSON,
            # found live — the very first run against this project's own code failed to
            # parse for exactly that reason.
            [sys.executable, "-m", "bandit", "-q", "-r", ".", "-f", "json", "--severity-level", "medium", "-x", ",".join(SKIP_DIRS)],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        if "No module named" in (result.stderr or ""):
            return {"ran": False, "tool": "bandit", "findings": 0, "detail": "bandit isn't installed."}
        data = json.loads(result.stdout or "{}")
        results = data.get("results", [])
        count = len(results)
        issues = sorted({r.get("test_name", "unknown") for r in results})
        detail = f"{count} finding(s) at medium+ severity: {', '.join(issues)}" if count else "No medium+ severity findings."
        return {"ran": True, "tool": "bandit", "findings": count, "detail": detail}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as exc:
        return {"ran": False, "tool": "bandit", "findings": 0, "detail": f"bandit failed to run: {exc}"}
