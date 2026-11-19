# SPDX-License-Identifier: Apache-2.0
"""Request-count phases wait semantically while still observing strategy errors."""

import asyncio

import pytest

from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.timing.phase.runner import PhaseRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_finishes_first", [False, True])
async def test_count_phase_waits_without_holding_the_virtual_clock(
    strategy_finishes_first: bool,
) -> None:
    runtime = VirtualTimingRuntime()
    runner = object.__new__(PhaseRunner)
    runner._runtime = runtime
    event = asyncio.Event()
    strategy = asyncio.get_running_loop().create_future()
    if strategy_finishes_first:
        strategy.set_result(None)
    task = runtime.create_task(
        runner._wait_for_event_or_task_exception(event, strategy, "sending")
    )
    await runtime.drain_current()
    assert not task.done()
    assert runtime.activity.idle
    runtime.call_later(2, event.set)
    runtime.advance_to(2_000_000_000)
    await runtime.drain_current()
    await task
    strategy.cancel()
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_strategy_failure_wins_over_a_same_timestamp_event() -> None:
    runtime = VirtualTimingRuntime()
    runner = object.__new__(PhaseRunner)
    runner._runtime = runtime
    event = asyncio.Event()
    strategy = asyncio.get_running_loop().create_future()
    strategy.set_exception(ValueError("strategy failed"))
    event.set()
    with (
        runtime.activity_scope(),
        pytest.raises(ValueError, match="strategy failed"),
    ):
        await runner._wait_for_event_or_task_exception(event, strategy, "sending")
