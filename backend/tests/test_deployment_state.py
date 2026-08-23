"""Tests for the DeploymentState repository functions — the real state tracking that
closes Phase 4's known N×N deployment-strategy-transition gap (promote_and_finish()'s
Canary branch used to guess the stable Deployment's name from a naming convention
instead of asking what it actually is). conftest.py already points this whole test
session at an isolated temp database."""

import pytest
import pytest_asyncio

from app.memory import repository as memory
from app.memory.db import init_db


@pytest_asyncio.fixture(autouse=True)
async def _ensure_test_db_ready():
    await init_db()


@pytest.mark.asyncio
async def test_get_deployment_state_is_none_before_anything_is_set():
    state = await memory.get_deployment_state("/repo-a", "myapp")
    assert state is None


@pytest.mark.asyncio
async def test_set_then_get_deployment_state_round_trips():
    await memory.set_deployment_state("/repo-b", "myapp", "myapp-green", "Blue-Green")
    state = await memory.get_deployment_state("/repo-b", "myapp")
    assert state is not None
    assert state.current_deployment_name == "myapp-green"
    assert state.current_strategy == "Blue-Green"


@pytest.mark.asyncio
async def test_set_deployment_state_overwrites_not_duplicates():
    """Exactly the scenario the real bug happened in: a Canary release right after a
    Blue-Green promotion needs the SECOND set_deployment_state call to correctly replace
    the tracked deployment name, not add a second row that would make get_deployment_state
    ambiguous."""
    await memory.set_deployment_state("/repo-c", "myapp", "myapp-green", "Blue-Green")
    await memory.set_deployment_state("/repo-c", "myapp", "myapp", "Canary")

    state = await memory.get_deployment_state("/repo-c", "myapp")
    assert state.current_deployment_name == "myapp"
    assert state.current_strategy == "Canary"


@pytest.mark.asyncio
async def test_deployment_state_is_scoped_per_workspace_and_app():
    await memory.set_deployment_state("/repo-d", "app-one", "app-one", "Rolling")
    await memory.set_deployment_state("/repo-d", "app-two", "app-two-blue", "Blue-Green")

    one = await memory.get_deployment_state("/repo-d", "app-one")
    two = await memory.get_deployment_state("/repo-d", "app-two")
    assert one.current_deployment_name == "app-one"
    assert two.current_deployment_name == "app-two-blue"
