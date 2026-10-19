# SPDX-License-Identifier: Apache-2.0
"""Worker/control arrival permutations must not change logical callback order."""

import asyncio
from contextlib import nullcontext
from typing import Any
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.credit.messages import CreditReturn
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit
from aiperf.simulation.identity import request_id
from aiperf.simulation.protocol import Cancel, Grant, Ready, SimulationError, Terminal
from aiperf.simulation.router import SimulatorCreditRouter
from tests.unit.timing.conftest import make_credit


class Clock:
    """Only the router boundary is under test; runtime draining has separate tests."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.started.set()
        self.finished.set()
        self.next_deadline_ns = None
        self.now = 0

    def now_ns(self) -> int:
        return self.now

    def advance_to(self, now: int) -> None:
        assert now >= self.now
        self.now = now

    async def drain_current(self) -> None:
        pass

    def assert_drained(self) -> None:
        assert self.next_deadline_ns is None

    def activity_scope(self, *, label: str) -> Any:
        return nullcontext()


def credit(number: int) -> Credit:
    return Credit(
        id=number,
        phase=CreditPhase.PROFILING,
        phase_index=0,
        conversation_id=f"conversation-{number}",
        x_correlation_id=f"session-{number}",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
    )


@pytest.fixture
def setup_router(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    client = Mock()
    client.request = AsyncMock()
    monkeypatch.setattr("aiperf.simulation.router.CoordinatorClient", lambda _: client)
    native = Mock()
    native.send_credit = AsyncMock()
    native.cancel_all_credits = AsyncMock()
    runtime = Clock()
    router = SimulatorCreditRouter(
        native,
        benchmark_id="run",
        endpoint="tcp://127.0.0.1:12345",
        runtime=runtime,
        timeout_s=0.1,
    )
    return router, native, client, runtime


@pytest.fixture
def setup_virtual_router(setup_router: tuple[Any, ...]) -> tuple[Any, ...]:
    router, native, client, _ = setup_router
    runtime = VirtualTimingRuntime()
    router.runtime = runtime
    return router, native, client, runtime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "return_first",
    [param(True, id="http-first"), param(False, id="grant-first")],
)  # fmt: skip
async def test_router_pairs_both_arrival_orders_and_preserves_grant_order(
    setup_router: tuple[Any, ...], return_first: bool
) -> None:
    router, native, client, runtime = setup_router
    credits = [credit(1), credit(2)]
    rids = [request_id("run", 0, item.id) for item in credits]
    callbacks = []

    async def returned(worker: str, message: CreditReturn) -> None:
        assert not message.first_token_sent
        callbacks.append((message.credit.id, runtime.now_ns()))

    async def deliver_http() -> None:
        for item in reversed(credits):
            await native.set_return_callback.call_args.args[0](
                "worker", CreditReturn(credit=item, first_token_sent=True)
            )

    async def exchange(ready: Ready) -> Grant:
        if ready.round == 0:
            if not return_first:
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.create_task(deliver_http())
                )
            return Grant("run", 0, 10, [Terminal(rid, "COMPLETED", 10) for rid in rids])
        assert ready.finished
        return Grant("run", 1, 10, [], status="FINISHED")

    client.request.side_effect = exchange
    router.set_return_callback(returned)
    for item in credits:
        await router.send_credit(item)
    if return_first:
        await deliver_http()
        assert callbacks == []
    await router.drive()
    assert callbacks == [(1, 10), (2, 10)]
    assert [
        op.rid for op in client.request.call_args_list[0].args[0].operations
    ] == rids
    assert client.request.await_count == 2
    client.close.assert_called_once()


@pytest.mark.asyncio
async def test_router_cancel_precedes_native_disconnect_and_waits_for_cleanup(
    setup_router: tuple[Any, ...],
) -> None:
    router, native, client, runtime = setup_router
    item = credit(1)
    rid = request_id("run", 0, 1)
    callback = AsyncMock()
    router.set_return_callback(callback)
    await router.send_credit(item)

    async def cancel_native() -> None:
        assert type(router._operations[-1]).__name__ == "Cancel"
        await native.set_return_callback.call_args.args[0](
            "worker", CreditReturn(credit=item, cancelled=True)
        )

    native.cancel_all_credits.side_effect = cancel_native
    await router.cancel_all_credits()
    client.request.side_effect = [
        Grant("run", 0, 0, [Terminal(rid, "CANCELLED", 0)]),
        Grant("run", 1, 10, [], status="FINISHED"),
    ]
    await router.drive()
    assert runtime.now_ns() == 10
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_drain_deduplicates_and_resets_for_next_batch(
    setup_virtual_router: tuple[Any, ...],
) -> None:
    router, native, _, runtime = setup_virtual_router
    callback = AsyncMock()
    router.set_return_callback(callback)

    with runtime.activity_scope():
        await router.wait_for_cancellation_drain(timeout=1)

    for number in (1, 2):
        item = credit(number)
        rid = request_id("run", 0, number)
        await router.send_credit(item)
        # Dispatch must remain usable without a running driver or activity scope.
        await asyncio.wait_for(router.cancel_all_credits(), 1)
        await asyncio.wait_for(router.cancel_all_credits(), 1)
        assert (
            sum(isinstance(op, Cancel) and op.rid == rid for op in router._operations)
            == 1
        )

        waiter = runtime.create_task(router.wait_for_cancellation_drain(timeout=1))
        try:
            await asyncio.wait_for(runtime.drain_current(), 1)
            assert not waiter.done()
            assert runtime.next_deadline_ns is None
            await native.set_return_callback.call_args.args[0](
                "worker", CreditReturn(credit=item, cancelled=True)
            )
            await asyncio.wait_for(runtime.drain_current(), 1)
            assert not waiter.done()
            await router._apply_grant(
                Grant("run", number, 0, [Terminal(rid, "CANCELLED", 0)])
            )
            await asyncio.wait_for(runtime.drain_current(), 1)
            await waiter
            assert callback.await_count == number
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    assert runtime.now_ns() == 0
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_cancellation_drain_waits_for_callback_after_pending_entry_removed(
    setup_virtual_router: tuple[Any, ...],
) -> None:
    router, native, _, runtime = setup_virtual_router
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    callback_completed = asyncio.Event()
    item = credit(1)
    rid = request_id("run", 0, 1)

    async def returned(worker: str, message: CreditReturn) -> None:
        callback_started.set()
        await runtime.wait_event(release_callback)
        callback_completed.set()

    router.set_return_callback(returned)
    await router.send_credit(item)
    await router.cancel_all_credits()
    await native.set_return_callback.call_args.args[0](
        "worker", CreditReturn(credit=item, cancelled=True)
    )
    grant = asyncio.create_task(
        router._apply_grant(Grant("run", 0, 0, [Terminal(rid, "CANCELLED", 0)]))
    )
    waiter = None
    try:
        await asyncio.wait_for(callback_started.wait(), 1)
        assert rid not in router._pending
        waiter = runtime.create_task(router.wait_for_cancellation_drain(timeout=1))
        await asyncio.wait_for(runtime.drain_current(), 1)
        assert not waiter.done()
        assert not callback_completed.is_set()
        release_callback.set()
        await asyncio.wait_for(grant, 1)
        await asyncio.wait_for(runtime.drain_current(), 1)
        await waiter
        assert callback_completed.is_set()
    finally:
        release_callback.set()
        grant.cancel()
        tasks = [grant]
        if waiter is not None:
            waiter.cancel()
            tasks.append(waiter)
        await asyncio.gather(*tasks, return_exceptions=True)
    assert runtime.now_ns() == 0
    runtime.assert_drained()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [param("mismatch", id="terminal-mismatch"), param("callback", id="callback-error")],
)  # fmt: skip
async def test_cancellation_drain_does_not_report_success_after_return_failure(
    setup_virtual_router: tuple[Any, ...], failure: str
) -> None:
    router, native, _, runtime = setup_virtual_router
    callback = AsyncMock(
        side_effect=ValueError("callback failed") if failure == "callback" else None
    )
    router.set_return_callback(callback)
    item = credit(1)
    await router.send_credit(item)
    await router.cancel_all_credits()
    await native.set_return_callback.call_args.args[0](
        "worker", CreditReturn(credit=item, cancelled=True)
    )

    async def wait_for_failure() -> None:
        with pytest.raises(TimeoutError):
            await router.wait_for_cancellation_drain(timeout=0.05)

    waiter = runtime.create_task(wait_for_failure())
    try:
        await asyncio.wait_for(runtime.drain_current(), 1)
        terminal = Terminal(
            request_id("run", 0, 1),
            "COMPLETED" if failure == "mismatch" else "CANCELLED",
            0,
        )
        error = SimulationError if failure == "mismatch" else ValueError
        message = "terminal mismatch" if failure == "mismatch" else "callback failed"
        with pytest.raises(error, match=message):
            await router._apply_grant(Grant("run", 0, 0, [terminal]))
        await asyncio.wait_for(waiter, 1)
        if failure == "mismatch":
            callback.assert_not_awaited()
        else:
            callback.assert_awaited_once()
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    assert runtime.now_ns() == 0


@pytest.mark.asyncio
async def test_cancellation_drain_timeout_does_not_release_unmatched_request(
    setup_virtual_router: tuple[Any, ...],
) -> None:
    router, native, _, runtime = setup_virtual_router
    callback = AsyncMock()
    router.set_return_callback(callback)
    item = credit(1)
    rid = request_id("run", 0, 1)
    await router.send_credit(item)
    await router.cancel_all_credits()
    await native.set_return_callback.call_args.args[0](
        "worker", CreditReturn(credit=item, cancelled=True)
    )
    with runtime.activity_scope(), pytest.raises(TimeoutError):
        await router.wait_for_cancellation_drain(timeout=0.001)
    assert rid in router._pending
    callback.assert_not_awaited()
    assert runtime.now_ns() == 0
    assert runtime.next_deadline_ns is None
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_worker_failure_interrupts_control_wait(
    setup_router: tuple[Any, ...],
) -> None:
    router, native, client, _ = setup_router
    item = credit(1)
    await router.send_credit(item)

    async def exchange(ready: Ready) -> Grant:
        await native.set_return_callback.call_args.args[0](
            "worker", CreditReturn(credit=item, error="payload construction failed")
        )
        await asyncio.Event().wait()
        pytest.fail("Control must be cancelled after the worker fails")

    client.request.side_effect = exchange
    with pytest.raises(SimulationError, match="payload construction failed"):
        await router.drive()
    assert client.request.await_count == 1
    client.close.assert_called_once()


@pytest.mark.asyncio
async def test_grant_without_http_terminal_fails_instead_of_releasing_callback(
    setup_router: tuple[Any, ...],
) -> None:
    router, _, client, _ = setup_router
    await router.send_credit(credit(1))
    callback = AsyncMock()
    router.set_return_callback(callback)
    client.request.return_value = Grant(
        "run", 0, 10, [Terminal(request_id("run", 0, 1), "COMPLETED", 10)]
    )
    with pytest.raises(SimulationError, match="did not settle"):
        await router.drive()
    callback.assert_not_awaited()


def test_request_identity_separates_warmup_phases_and_credit_prefixes() -> None:
    rids = {
        request_id("run", phase, number)
        for phase in (None, 0, 1)
        for number in (1, 10, 100)
    }
    assert len(rids) == 9
    assert len({len(rid) for rid in rids}) == 1
    assert request_id("other", 0, 1) not in rids


@pytest.mark.asyncio
async def test_child_keeps_parent_worker_after_parent_final_turn(
    setup_router: tuple[Any, ...], benchmark_run: Any
) -> None:
    _, _, client, runtime = setup_router
    native = StickyCreditRouter(run=benchmark_run, service_id="test-router")
    native._router_client.send_to = AsyncMock()
    native._register_worker("worker-A")
    native._register_worker("worker-B")
    router = SimulatorCreditRouter(
        native, benchmark_id="run", endpoint="tcp://127.0.0.1:12345", runtime=runtime
    )
    await router.send_credit(make_credit(id=1, corr_id="root", turn=0, num_turns=2))
    parent_worker = native._router_client.send_to.call_args.args[0]
    branch_router = getattr(router, "sticky_router", None)
    assert branch_router is not None
    branch_router.register_child_routing("root")
    await router.send_credit(make_credit(id=2, corr_id="root", turn=1, num_turns=2))
    child = msgspec.structs.replace(
        make_credit(id=3, corr_id="child", turn=0, num_turns=1),
        parent_correlation_id="root",
    )
    await router.send_credit(child)
    assert native._router_client.send_to.call_args.args[0] == parent_worker
    branch_router.release_child_routing("root")
    assert "root" not in native._sticky_sessions
    router.close()
    client.close.assert_called_once()
