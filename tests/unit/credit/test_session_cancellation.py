# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session cancellation must not cancel other sessions sharing a worker."""

from unittest.mock import AsyncMock

import pytest

from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.credit.messages import CancelCredits
from aiperf.credit.sticky_router import StickyCreditRouter
from tests.unit.timing.conftest import make_credit


@pytest.fixture
def registered_router(benchmark_run: BenchmarkRun) -> StickyCreditRouter:
    """Create a real one-worker router with a mock wire transport."""
    router = StickyCreditRouter(run=benchmark_run, service_id="test-session-cancel")
    router._router_client.send_to = AsyncMock()
    router._register_worker("worker-1")
    return router


@pytest.mark.asyncio
async def test_cancel_fork_child_preserves_parent_sibling_and_unrelated_session(
    registered_router: StickyCreditRouter,
) -> None:
    """Keep other sessions alive when their worker is shared with the target."""
    router = registered_router
    await router.send_credit(make_credit(id=1, corr_id="parent", num_turns=3))
    await router.send_credit(
        make_credit(id=2, corr_id="target", parent_correlation_id="parent")
    )
    await router.send_credit(
        make_credit(id=3, corr_id="sibling", parent_correlation_id="parent")
    )
    await router.send_credit(make_credit(id=4, corr_id="unrelated"))
    worker = router._workers["worker-1"]
    router._router_client.send_to.reset_mock()

    await router.cancel_session_credits("target")

    router._router_client.send_to.assert_awaited_once_with(
        "worker-1", CancelCredits(credit_ids={2})
    )
    assert worker.active_credit_ids == {1, 2, 3, 4}
    assert worker.in_flight_credits == 4
    assert "parent" in router._sticky_sessions
    assert not router._cancellation_pending


@pytest.mark.asyncio
async def test_cancel_session_is_idempotent_until_credit_returns(
    registered_router: StickyCreditRouter,
) -> None:
    """Coalesce repeated aborts until their credit returns."""
    router = registered_router
    await router.send_credit(make_credit(id=1, corr_id="target"))
    router._router_client.send_to.reset_mock()

    await router.cancel_session_credits("target")
    await router.cancel_session_credits("target")

    router._router_client.send_to.assert_awaited_once_with(
        "worker-1", CancelCredits(credit_ids={1})
    )


@pytest.mark.asyncio
async def test_phase_cancellation_does_not_repeat_session_cancellation(
    registered_router: StickyCreditRouter,
) -> None:
    """A phase boundary must not interrupt cancellation cleanup twice."""
    router = registered_router
    await router.send_credit(make_credit(id=1, corr_id="target"))
    await router.send_credit(make_credit(id=2, corr_id="neighbor"))
    router._router_client.send_to.reset_mock()

    await router.cancel_session_credits("target")
    await router.cancel_all_credits()

    sent = [
        call.args[1].credit_ids
        for call in router._router_client.send_to.await_args_list
    ]
    assert sent == [{1}, {2}]
    assert router._workers["worker-1"].in_flight_credits == 2


@pytest.mark.asyncio
async def test_completed_credit_cannot_be_cancelled(
    registered_router: StickyCreditRouter,
) -> None:
    """Forget completed credits before selecting cancellations."""
    router = registered_router
    await router.send_credit(make_credit(id=1, corr_id="target"))
    router._track_credit_returned("worker-1", 1, False, False)
    router._router_client.send_to.reset_mock()

    await router.cancel_session_credits("target")

    router._router_client.send_to.assert_not_awaited()
    assert router._workers["worker-1"].in_flight_credits == 0


@pytest.mark.asyncio
async def test_cancel_unknown_session_does_not_broadcast(
    registered_router: StickyCreditRouter,
) -> None:
    """Unknown session IDs must not cancel neighboring work."""
    router = registered_router
    await router.send_credit(make_credit(id=1, corr_id="neighbor"))
    router._router_client.send_to.reset_mock()

    await router.cancel_session_credits("missing")

    router._router_client.send_to.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_cancel_send_can_be_retried(
    registered_router: StickyCreditRouter,
) -> None:
    """A failed send must not make later cancellation attempts inert."""
    router = registered_router
    await router.send_credit(make_credit(id=1, corr_id="target"))
    router._router_client.send_to.reset_mock()
    router._router_client.send_to.side_effect = RuntimeError("transport unavailable")

    with pytest.raises(RuntimeError, match="transport unavailable"):
        await router.cancel_session_credits("target")
    router._router_client.send_to.side_effect = None
    await router.cancel_session_credits("target")

    assert router._router_client.send_to.await_count == 2
    assert router._workers["worker-1"].in_flight_credits == 1


@pytest.mark.asyncio
async def test_credit_return_retires_session_and_cancellation_tracking(
    registered_router: StickyCreditRouter,
) -> None:
    """Retire ownership and pending cancellation when the worker acknowledges it."""
    router = registered_router
    await router.send_credit(make_credit(id=1, corr_id="target"))
    await router.cancel_session_credits("target")

    router._track_credit_returned("worker-1", 1, True, False)

    worker = router._workers["worker-1"]
    assert not worker.active_credit_sessions
    assert not worker.cancellation_requested_ids
    assert not worker.active_credit_ids
    assert worker.in_flight_credits == 0
    assert worker.total_cancelled_credits == 1
