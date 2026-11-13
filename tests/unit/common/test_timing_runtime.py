# SPDX-License-Identifier: Apache-2.0
"""Clock injection without replacing asyncio task execution."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
from pytest import param

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.timing_runtime import (
    CallbackHandle,
    RealTimingRuntime,
    TimerHandle,
    TimingRuntime,
)
from aiperf.common.utils import yield_to_event_loop


@pytest.fixture
def runtime() -> Mock:
    runtime = Mock(spec=TimingRuntime)
    runtime.now_ns.return_value = 10 * NANOS_PER_SECOND
    runtime.timer_time.return_value = 100.0

    def call_at(when: float, callback: Callable[..., Any], *args: Any) -> Mock:
        handle = Mock(spec=TimerHandle)
        handle.when.return_value = when
        return handle

    runtime.call_at.side_effect = call_at
    runtime.call_later.side_effect = lambda delay, callback, *args: runtime.call_at(
        runtime.timer_time() + delay, callback, *args
    )
    runtime.call_soon.side_effect = lambda callback, *args: Mock(spec=CallbackHandle)
    return runtime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,target",
    [
        param("schedule_at_perf_ns", 12 * NANOS_PER_SECOND, id="nanoseconds"),
        param("schedule_at_perf_sec", 12.0, id="seconds"),
        param("schedule_later", 2.0, id="relative"),
        param("schedule_at", 102.0, id="timer-clock"),
    ],
)  # fmt: skip
async def test_scheduler_injected_clock_controls_deadline_and_release(
    runtime: Mock, method: str, target: int | float
) -> None:
    scheduler = LoopScheduler(runtime=runtime)
    completed = asyncio.Event()

    async def work() -> None:
        completed.set()

    getattr(scheduler, method)(target, work())
    when, callback, *args = runtime.call_at.call_args.args
    assert when == 102.0
    await yield_to_event_loop()
    assert not completed.is_set()
    assert scheduler.pending_count == 1

    callback(*args)
    await completed.wait()
    await yield_to_event_loop()
    assert scheduler.is_idle()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "group,expected",
    [
        param(None, [101.0, 104.0], id="all-timers"),
        param("a", [101.0, 108.0], id="one-tree"),
    ],
)  # fmt: skip
async def test_scheduler_injected_clock_cap_preserves_other_deadlines(
    runtime: Mock, group: str | None, expected: list[float]
) -> None:
    scheduler = LoopScheduler(runtime=runtime)

    async def work() -> None:
        pytest.fail("Pending work must not run before the runtime releases it")

    coroutines = [work(), work()]
    scheduler.schedule_later(5.0, coroutines[0], group_id="a")
    scheduler.schedule_later(8.0, coroutines[1], group_id="b")
    if group is None:
        shifted = scheduler.cap_pending_delay(1.0)
    else:
        shifted = scheduler.cap_pending_delay_for_group(group, 1.0)
    assert shifted == 4.0
    assert sorted(h.when() for h, _ in scheduler._handles.values()) == expected
    scheduler.cancel_all_pending()
    assert scheduler.is_idle()
    assert all(inspect.getcoroutinestate(c) == inspect.CORO_CLOSED for c in coroutines)


@pytest.mark.asyncio
async def test_scheduler_zero_cap_enqueues_once_in_injected_runtime(
    runtime: Mock,
) -> None:
    scheduler = LoopScheduler(runtime=runtime)

    async def work() -> None:
        pytest.fail("Capped work must still wait for a runtime callback")

    scheduler.schedule_later(5.0, work())
    assert scheduler.cap_pending_delay(0.0) == 5.0
    assert scheduler.cap_pending_delay(0.0) == 0.0
    runtime.call_soon.assert_called_once()
    scheduler.cancel_all_pending()


def test_real_runtime_clock_reads_do_not_require_an_event_loop() -> None:
    runtime = RealTimingRuntime()
    before = runtime.now_ns()
    assert runtime.timestamp_ns() > 0
    assert runtime.now_ns() >= before
