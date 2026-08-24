"""Real GitHub OAuth Device Flow authentication — built on request, even though
scopes.py's own module docstring argues OAuth is security theater for a single local
backend behind one VS Code extension with no second user to scope away from. That
reasoning doesn't change here: this doesn't add multi-user access control (there's still
only one real user), it adds a real "prove you're a logged-in human with a real GitHub
account" gate in front of the backend's HTTP surface, the same mechanism `gh auth login`
itself uses (this project already depends on `gh` being authenticated — see the build
plan's Round 3 CI addendum) rather than a device-flow implementation invented from
scratch.

No client secret needed or used: GitHub's Device Flow is designed for exactly this kind
of client (a local tool with no way to keep a secret confidential), so the OAuth App is
registered as public with only a Client ID. That ID isn't sensitive — it's meant to be
embedded in client code, the same way it's visible in every open-source CLI that uses
device flow (gh itself included)."""

import os

import httpx

GITHUB_CLIENT_ID = os.getenv("RELEASEFORGE_GITHUB_CLIENT_ID", "Ov23liNq51YDl8ey8RTU")
DEVICE_CODE_URL = "https://github.com/login/device/code"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"

# One shared token for the one real user this backend ever serves — an in-memory module
# global, the same "single local user" reasoning scopes.py already uses, not a database
# table implying multi-user accounts that don't exist. Lost on backend restart by
# design: a fresh process should not silently trust a stale session.
_current_token: str | None = None
_current_login: str | None = None


class DeviceFlowError(Exception):
    pass


async def start_device_flow() -> dict:
    """Kicks off the flow: GitHub returns a user_code to show the human and a
    device_code this backend polls with. No scope requested — this only needs to prove
    identity, not grant any GitHub API access on the user's behalf."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            DEVICE_CODE_URL,
            data={"client_id": GITHUB_CLIENT_ID, "scope": ""},
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    if "device_code" not in data:
        raise DeviceFlowError(f"GitHub didn't return a device code: {data}")
    return data


async def poll_once(device_code: str) -> dict:
    """One poll attempt — the caller (the /auth/device/poll endpoint) is responsible for
    calling this at the interval GitHub specified in start_device_flow()'s response, not
    this function busy-waiting itself; an HTTP endpoint blocking for a human to click
    "Authorize" on GitHub's site would tie up a request for an unbounded, human-timescale
    amount of time.

    Returns {"status": "authenticated", "login": ...} once approved,
    {"status": "pending"} while waiting, or {"status": "error", "detail": ...} on a real
    failure (expired, denied, or a genuinely unexpected response)."""
    global _current_token, _current_login

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            ACCESS_TOKEN_URL,
            data={
                "client_id": GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    error = data.get("error")
    if error == "authorization_pending":
        return {"status": "pending"}
    if error == "slow_down":
        return {"status": "pending"}
    if error:
        return {"status": "error", "detail": data.get("error_description", error)}

    access_token = data.get("access_token")
    if not access_token:
        return {"status": "error", "detail": f"Unexpected response from GitHub: {data}"}

    login = await _fetch_login(access_token)
    _current_token = access_token
    _current_login = login
    return {"status": "authenticated", "login": login}


async def _fetch_login(access_token: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        return resp.json().get("login", "unknown")


def is_authenticated() -> bool:
    return _current_token is not None


def current_user() -> str | None:
    return _current_login


def check_bearer_token(authorization_header: str | None) -> None:
    """Raises PermissionError if the given `Authorization` header doesn't carry the
    exact token this backend issued to itself via the device flow above. Deliberately a
    plain function, not a class/dependency-injection object — this project's other
    security checks (check_scope in scopes.py) are already plain functions, kept
    consistent rather than introducing a different pattern for this one."""
    if _current_token is None:
        raise PermissionError("Not authenticated — complete the GitHub device flow first (POST /auth/device/start).")
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise PermissionError("Missing or malformed Authorization header.")
    token = authorization_header.removeprefix("Bearer ").strip()
    if token != _current_token:
        raise PermissionError("Invalid token.")


def sign_out() -> None:
    global _current_token, _current_login
    _current_token = None
    _current_login = None


def issued_token() -> str | None:
    """The token to actually hand back to the extension once authenticated — read once,
    right after /auth/device/poll reports success. Separate from check_bearer_token()
    (which never reveals the token, only checks against it) to keep "read the secret"
    and "verify the secret" as two distinctly-named operations."""
    return _current_token
