"""Tests for app/security/oauth.py's real GitHub Device Flow implementation. Real
network calls are never made — httpx.AsyncClient is monkeypatched to a fake that returns
canned GitHub-shaped responses, the same "prove the logic, not the model/service"
approach every other external-dependency test in this suite already uses."""

import pytest

from app.security import oauth


@pytest.fixture(autouse=True)
def _reset_oauth_state():
    """oauth.py's authenticated state is a module-level global by design (one real user,
    no database row implying accounts that don't exist) — must be reset between tests so
    one test's sign-in doesn't leak into the next."""
    oauth.sign_out()
    yield
    oauth.sign_out()


class _FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Returns each queued response in order, one per post()/get() call — enough to
    script a full start -> poll(pending) -> poll(authenticated) -> fetch-login sequence
    without ever touching the network."""

    def __init__(self, responses):
        self._responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, data=None, headers=None):
        return self._responses.pop(0)

    async def get(self, url, headers=None):
        return self._responses.pop(0)


def _patch_httpx(monkeypatch, *responses):
    client = _FakeAsyncClient(responses)
    monkeypatch.setattr(oauth.httpx, "AsyncClient", lambda: client)


# ---------- check_bearer_token ----------


def test_check_bearer_token_rejects_when_nobody_has_authenticated():
    with pytest.raises(PermissionError, match="complete the GitHub device flow"):
        oauth.check_bearer_token("Bearer anything")


def test_check_bearer_token_rejects_missing_header(monkeypatch):
    monkeypatch.setattr(oauth, "_current_token", "real-token")
    with pytest.raises(PermissionError, match="Missing or malformed"):
        oauth.check_bearer_token(None)


def test_check_bearer_token_rejects_malformed_header(monkeypatch):
    monkeypatch.setattr(oauth, "_current_token", "real-token")
    with pytest.raises(PermissionError, match="Missing or malformed"):
        oauth.check_bearer_token("real-token")  # missing "Bearer " prefix


def test_check_bearer_token_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(oauth, "_current_token", "real-token")
    with pytest.raises(PermissionError, match="Invalid token"):
        oauth.check_bearer_token("Bearer wrong-token")


def test_check_bearer_token_accepts_the_real_token(monkeypatch):
    monkeypatch.setattr(oauth, "_current_token", "real-token")
    oauth.check_bearer_token("Bearer real-token")  # must not raise


# ---------- device flow ----------


@pytest.mark.asyncio
async def test_start_device_flow_returns_githubs_response(monkeypatch):
    _patch_httpx(
        monkeypatch,
        _FakeResponse(
            {"device_code": "dc123", "user_code": "ABCD-1234", "verification_uri": "https://github.com/login/device", "expires_in": 900, "interval": 5}
        ),
    )
    data = await oauth.start_device_flow()
    assert data["user_code"] == "ABCD-1234"


@pytest.mark.asyncio
async def test_start_device_flow_raises_on_a_malformed_response(monkeypatch):
    _patch_httpx(monkeypatch, _FakeResponse({"error": "something GitHub-shaped went wrong"}))
    with pytest.raises(oauth.DeviceFlowError):
        await oauth.start_device_flow()


@pytest.mark.asyncio
async def test_poll_once_reports_pending_while_waiting(monkeypatch):
    _patch_httpx(monkeypatch, _FakeResponse({"error": "authorization_pending"}))
    result = await oauth.poll_once("dc123")
    assert result == {"status": "pending"}
    assert oauth.is_authenticated() is False


@pytest.mark.asyncio
async def test_poll_once_reports_a_real_error_distinctly_from_pending(monkeypatch):
    _patch_httpx(monkeypatch, _FakeResponse({"error": "expired_token", "error_description": "The device code has expired."}))
    result = await oauth.poll_once("dc123")
    assert result["status"] == "error"
    assert "expired" in result["detail"]


@pytest.mark.asyncio
async def test_poll_once_authenticates_and_fetches_the_real_login(monkeypatch):
    _patch_httpx(
        monkeypatch,
        _FakeResponse({"access_token": "gho_realtoken", "token_type": "bearer"}),
        _FakeResponse({"login": "111satwika"}),
    )
    result = await oauth.poll_once("dc123")
    assert result == {"status": "authenticated", "login": "111satwika"}
    assert oauth.is_authenticated() is True
    assert oauth.current_user() == "111satwika"
    assert oauth.issued_token() == "gho_realtoken"


@pytest.mark.asyncio
async def test_sign_out_clears_the_authenticated_state(monkeypatch):
    _patch_httpx(
        monkeypatch,
        _FakeResponse({"access_token": "gho_realtoken", "token_type": "bearer"}),
        _FakeResponse({"login": "111satwika"}),
    )
    await oauth.poll_once("dc123")
    assert oauth.is_authenticated() is True

    oauth.sign_out()
    assert oauth.is_authenticated() is False
    assert oauth.current_user() is None
    with pytest.raises(PermissionError):
        oauth.check_bearer_token("Bearer gho_realtoken")
