"""Unit tests for builtin hooks."""

from __future__ import annotations

import pytest

from src.framework.harness.hooks import HookChain
from src.framework.harness.hooks.builtins import (
    PayloadSizeHook,
    RequiredMetadataKeysHook,
    SecretScanHook,
)
from src.framework.harness.protocols import HookCost
from src.framework.harness.state import (
    ContextPayload,
    HarnessState,
    Observation,
    Task,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_payload_size_hook_passes_when_small() -> None:
    """A small context payload passes the size hook."""
    state = HarnessState(last_context=ContextPayload(system_prompt="hi", task_brief="t"))
    hook = PayloadSizeHook(max_chars=10_000)
    assert await hook.check(state) is None


@pytest.mark.asyncio
async def test_payload_size_hook_violates_when_too_big() -> None:
    """An overlarge payload triggers a violation."""
    big = "x" * 100
    state = HarnessState(last_context=ContextPayload(system_prompt=big, task_brief="t"))
    hook = PayloadSizeHook(max_chars=50)
    violation = await hook.check(state)
    assert violation is not None
    assert "exceeds limit" in violation.detail


@pytest.mark.asyncio
async def test_payload_size_hook_no_op_without_context() -> None:
    """No context payload yet → no violation possible."""
    state = HarnessState()
    hook = PayloadSizeHook(max_chars=10)
    assert await hook.check(state) is None


@pytest.mark.asyncio
async def test_required_metadata_keys_hook_passes_when_present() -> None:
    """All required keys present → no violation."""
    state = HarnessState(task=Task(id="t", goal="g", metadata={"owner": "alice", "ticket": "T-1"}))
    hook = RequiredMetadataKeysHook(required=["owner", "ticket"])
    assert await hook.check(state) is None


@pytest.mark.asyncio
async def test_required_metadata_keys_hook_violates_when_missing() -> None:
    """Missing keys produce a violation listing them."""
    state = HarnessState(task=Task(id="t", goal="g", metadata={"owner": "alice"}))
    hook = RequiredMetadataKeysHook(required=["owner", "ticket"])
    violation = await hook.check(state)
    assert violation is not None
    assert "ticket" in violation.detail


@pytest.mark.asyncio
async def test_secret_scan_detects_openai_key() -> None:
    """An OpenAI-style key in the response text trips the hook."""
    state = HarnessState(last_response_text="found my key: sk-" + "a" * 30)
    hook = SecretScanHook()
    violation = await hook.check(state)
    assert violation is not None
    assert "openai" in violation.detail


@pytest.mark.asyncio
async def test_secret_scan_clean_passes() -> None:
    """No secret patterns → no violation."""
    state = HarnessState(last_response_text="all clean")
    hook = SecretScanHook()
    assert await hook.check(state) is None


@pytest.mark.asyncio
async def test_secret_scan_inspects_observations() -> None:
    """Secrets in tool observations are also caught."""
    obs = Observation(
        invocation_id="i",
        tool_name="shell",
        success=True,
        payload="-----BEGIN RSA PRIVATE KEY-----\nMIIE",
    )
    state = HarnessState(last_observations=(obs,))
    hook = SecretScanHook()
    violation = await hook.check(state)
    assert violation is not None
    assert "private_key" in violation.detail


@pytest.mark.asyncio
async def test_hook_chain_orders_builtins_cheap_first() -> None:
    """Builtins ordered by their declared cost class."""
    chain = HookChain.of(
        SecretScanHook(),
        RequiredMetadataKeysHook(required=["owner"]),
        PayloadSizeHook(max_chars=10_000),
    )
    # Cheap (payload_size, required_keys) ahead of medium (secret_scan).
    assert chain.names()[0] in {"payload_size", "required_metadata_keys"}
    assert chain.names()[-1] == "secret_scan"


def test_secret_scan_cost_class_is_medium() -> None:
    """``SecretScanHook`` declares the medium cost class."""
    hook = SecretScanHook()
    assert hook.cost_class is HookCost.MEDIUM
