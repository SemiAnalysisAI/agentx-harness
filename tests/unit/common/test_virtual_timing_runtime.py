# SPDX-License-Identifier: Apache-2.0
"""Causal virtual time without polling asyncio's runnable queue."""

import asyncio
import inspect

import pytest
from pytest import param

from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.timing.concurrency import (
    DynamicConcurrencyLimit,
    GlobalPhaseConcurrencyLimiter,
)


@pytest.mark.asyncio
async def test_run_scope_owns_lifecycle_across_nested_activity() -> None:
    runtime = VirtualTimingRuntime()
    with runtime.activity_scope():
        assert not runtime.started.is_set()
    assert not runtime.finished.is_set()

    with runtime.run_scope():
        assert runtime.started.is_set()
        with runtime.activity_scope():
            assert not runtime.activity.idle
        assert not runtime.finished.is_set()
        assert not runtime.activity.idle
    assert runtime.finished.is_set()
    runtime.assert_drained()

    with pytest.raises(RuntimeError, match="only run once"), runtime.run_scope():
        pytest.fail("A completed virtual run must not restart")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [param(ValueError, id="failure"), param(asyncio.CancelledError, id="cancelled")],
)  # fmt: skip
async def test_run_scope_closes_on_failure_or_cancellation(
    failure: type[BaseException],
) -> None:
    runtime = VirtualTimingRuntime()
    error = failure("stopped")
    with pytest.raises(failure) as caught, runtime.run_scope():
        raise error
    assert caught.value is error
    assert runtime.started.is_set()
    assert runtime.finished.is_set()
    assert runtime.activity.idle


@pytest.mark.asyncio
async def test_clock_advance_does_not_release_callbacks() -> None:
    runtime = VirtualTimingRuntime(timestamp_origin_ns=123)
    seen = []
    runtime.call_later(3, lambda: seen.append(runtime.now_ns()))
    assert runtime.next_deadline_ns == 3_000_000_000
    runtime.advance_to(3_000_000_000)
    assert seen == []
    assert runtime.timestamp_ns() == 3_000_000_123
    await runtime.drain_current()
    assert seen == [3_000_000_000]
    assert runtime.next_deadline_ns is None


@pytest.mark.asyncio
async def test_task_registration_and_event_ready_before_resume() -> None:
    runtime = VirtualTimingRuntime()
    ready = asyncio.Event()
    seen = []

    async def work() -> None:
        seen.append("started")
        await runtime.wait_event(ready)
        seen.append("resumed")

    task = runtime.create_task(work())
    assert not runtime.activity.idle
    await runtime.drain_current()
    assert seen == ["started"]
    assert runtime.activity.idle
    ready.set()
    assert not runtime.activity.idle
    await runtime.drain_current()
    assert seen == ["started", "resumed"]
    assert task.done()


@pytest.mark.asyncio
async def test_final_drain_rejects_blocked_activity_and_future_timer() -> None:
    runtime = VirtualTimingRuntime()
    release = asyncio.Event()
    task = runtime.create_task(runtime.wait_event(release))
    await runtime.drain_current()
    assert runtime.activity.idle
    with pytest.raises(RuntimeError, match="Unfinished timing activities"):
        runtime.assert_drained()
    release.set()
    await runtime.drain_current()
    assert task.done()
    timer = runtime.call_later(1, lambda: None)
    with pytest.raises(RuntimeError, match="Unfinished timing deadline"):
        runtime.assert_drained()
    timer.cancel()
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_gather_tracks_capacity_wait_and_all_lanes() -> None:
    runtime = VirtualTimingRuntime()
    capacity = DynamicConcurrencyLimit(1, runtime=runtime)
    release = asyncio.Event()
    seen = []

    async def lane(index: int) -> None:
        await capacity.acquire()
        seen.append(index)
        if index == 0:
            await runtime.wait_event(release)
        capacity.release()

    async def phases() -> None:
        with runtime.run_scope():
            await runtime.gather(lane(0), lane(1), lane(2))

    task = runtime.create_task(phases())
    await runtime.started.wait()
    await runtime.drain_current()
    assert seen == [0]
    assert not runtime.finished.is_set()
    release.set()
    await runtime.drain_current()
    assert seen == [0, 1, 2]
    assert runtime.finished.is_set()
    assert task.done()
    assert capacity.effective_slots == 1


@pytest.mark.asyncio
async def test_capacity_wake_generation_keeps_unresumed_waiter_active() -> None:
    runtime = VirtualTimingRuntime()
    capacity = DynamicConcurrencyLimit(0, runtime=runtime)
    seen = []

    async def acquire(index: int) -> None:
        await capacity.acquire()
        seen.append(index)

    tasks = [runtime.create_task(acquire(i)) for i in range(2)]
    await runtime.drain_current()
    capacity.set_limit(1)
    assert not runtime.activity.idle
    await runtime.drain_current()
    assert seen == [0]
    capacity.release()
    await runtime.drain_current()
    assert seen == [0, 1]
    assert all(task.done() for task in tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_at_deadline", [False, True])
async def test_phase_timeout_precedes_work_timer_but_completed_result_wins(
    complete_at_deadline: bool,
) -> None:
    runtime = VirtualTimingRuntime()
    event = asyncio.Event()
    seen = []
    runtime.call_later(1, lambda: seen.append("work"))

    async def phase() -> None:
        try:
            await runtime.wait_event(event, timeout=1)
            seen.append("complete")
        except TimeoutError:
            seen.append("timeout")

    runtime.create_task(phase())
    await runtime.drain_current()
    runtime.advance_to(1_000_000_000)
    if complete_at_deadline:
        event.set()
    await runtime.drain_current()
    assert seen == ["complete" if complete_at_deadline else "timeout", "work"]


@pytest.mark.asyncio
async def test_done_observer_and_nested_tasks_settle_before_grant() -> None:
    runtime = VirtualTimingRuntime()
    scheduler = LoopScheduler(runtime=runtime)
    seen = []

    async def work(index: int) -> None:
        seen.append(index)

    def drained() -> None:
        if len(seen) == 1:
            scheduler.schedule_later(0, work(2))

    scheduler.set_drain_observer(drained)
    scheduler.execute_async(work(1))
    await runtime.drain_current()
    assert seen == [1, 2]
    assert scheduler.is_idle()


@pytest.mark.asyncio
async def test_cancel_blocked_task_settles_cleanup_before_idle() -> None:
    runtime = VirtualTimingRuntime()
    cleaned = []

    async def work() -> None:
        try:
            await runtime.sleep(10)
        finally:
            cleaned.append(True)

    task = runtime.create_task(work())
    await runtime.drain_current()
    task.cancel()
    assert not runtime.activity.idle
    await runtime.drain_current()
    assert cleaned == [True]
    assert runtime.next_deadline_ns is None


@pytest.mark.asyncio
async def test_cancel_before_start_closes_coroutine() -> None:
    runtime = VirtualTimingRuntime()

    async def work() -> None:
        pytest.fail("Cancelled task must not start")

    coro = work()
    task = runtime.create_task(coro)
    task.cancel()
    await runtime.drain_current()
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_cancel_waiting_phase_capacity_releases_global_slot() -> None:
    runtime = VirtualTimingRuntime()
    limiter = GlobalPhaseConcurrencyLimiter(runtime=runtime)
    limiter.configure_for_phase(0, 1)
    limiter._phase_limits[0].set_limit(0)
    task = runtime.create_task(limiter.acquire(0, lambda: True))
    await runtime.drain_current()
    assert limiter._global_limit.effective_slots == 0
    task.cancel()
    await runtime.drain_current()
    assert limiter._global_limit.effective_slots == 1


@pytest.mark.asyncio
async def test_tracked_failure_is_not_successful_quiescence() -> None:
    runtime = VirtualTimingRuntime()

    async def work() -> None:
        raise ValueError("broken workflow")

    runtime.create_task(work())
    with pytest.raises(RuntimeError, match="Tracked timing work failed") as error:
        await runtime.drain_current()
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_wall_liveness_timeout_does_not_advance_virtual_clock() -> None:
    runtime = VirtualTimingRuntime()
    timed_out = asyncio.Event()

    async def work() -> None:
        with pytest.raises(TimeoutError):
            await runtime.wait_wall_event(asyncio.Event(), 0.001)
        timed_out.set()

    runtime.create_task(work())
    await runtime.drain_current()
    assert runtime.next_deadline_ns is None
    await asyncio.wait_for(timed_out.wait(), 1)
    await runtime.drain_current()
    assert runtime.now_ns() == 0


@pytest.mark.asyncio
async def test_untracked_semantic_wait_rejected() -> None:
    runtime = VirtualTimingRuntime()
    with pytest.raises(RuntimeError, match="tracked activity"):
        await runtime.wait_event(asyncio.Event())


@pytest.mark.asyncio
async def test_deadline_validation_and_cancelled_head() -> None:
    runtime = VirtualTimingRuntime()
    cancelled = runtime.call_later(1, lambda: None)
    runtime.call_later(2, lambda: None)
    cancelled.cancel()
    assert runtime.next_deadline_ns == 2_000_000_000
    with pytest.raises(ValueError, match="past pending"):
        runtime.advance_to(3_000_000_000)
    for value in (-1, 1.1, True):
        with pytest.raises(ValueError, match="monotonic integer"):
            runtime.advance_to(value)
    with pytest.raises(ValueError, match="finite"):
        runtime.call_later(float("nan"), lambda: None)
