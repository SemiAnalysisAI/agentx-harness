# SPDX-License-Identifier: Apache-2.0
"""Native rate and concurrency work must wait for logical time or completions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.request_rate import RequestRateStrategy


def make_strategy(
    runtime: VirtualTimingRuntime,
    *,
    pattern: ArrivalPattern = ArrivalPattern.CONSTANT,
    rate: float | None = 10,
) -> tuple[RequestRateStrategy, MagicMock]:
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        arrival_pattern=pattern,
        request_rate=rate,
        total_expected_requests=3,
    )
    source = MagicMock()
    source.next.return_value.build_first_turn.return_value = TurnToSend(
        "conversation", "session", 0, 1
    )
    issuer = MagicMock(try_issue_credit=AsyncMock(), issue_credit=AsyncMock())
    checker = MagicMock()
    checker.can_start_new_session.return_value = True
    strategy = RequestRateStrategy(
        config=config,
        conversation_source=source,
        scheduler=MagicMock(),
        stop_checker=checker,
        credit_issuer=issuer,
        lifecycle=SimpleNamespace(started_at_perf_ns=0),
        runtime=runtime,
    )
    return strategy, issuer


@pytest.mark.asyncio
async def test_constant_arrivals_advance_only_with_virtual_time() -> None:
    runtime = VirtualTimingRuntime()
    strategy, issuer = make_strategy(runtime)
    observed = []

    async def issue(turn: TurnToSend) -> bool:
        observed.append(runtime.now_ns())
        return len(observed) < 3

    issuer.try_issue_credit.side_effect = issue
    task = runtime.create_task(strategy.execute_phase())
    await runtime.drain_current()
    assert observed == []
    for index in range(1, 4):
        deadline = index * 100_000_000
        assert runtime.next_deadline_ns == deadline
        runtime.advance_to(deadline)
        await runtime.drain_current()
        assert observed == [i * 100_000_000 for i in range(1, index + 1)]
    await task
    runtime.assert_drained()
    assert strategy._create_high_res_pacer() is None


@pytest.mark.asyncio
async def test_saturated_concurrency_waits_for_a_completion_without_busy_loop() -> None:
    runtime = VirtualTimingRuntime()
    strategy, issuer = make_strategy(
        runtime, pattern=ArrivalPattern.CONCURRENCY_BURST, rate=None
    )
    issuer.try_issue_credit.side_effect = [None, False]
    task = runtime.create_task(strategy.execute_phase())
    await runtime.drain_current()
    issuer.try_issue_credit.assert_awaited_once()
    assert not task.done()
    assert runtime.activity.idle
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conversation",
        x_correlation_id="session",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
    )
    await strategy.handle_credit_return(credit)
    await runtime.drain_current()
    await task
    assert issuer.try_issue_credit.await_count == 2
    runtime.assert_drained()


@pytest.mark.asyncio
async def test_delayed_continuation_wakes_the_waiting_virtual_strategy() -> None:
    runtime = VirtualTimingRuntime()
    strategy, issuer = make_strategy(
        runtime, pattern=ArrivalPattern.CONCURRENCY_BURST, rate=None
    )
    strategy._stop_checker.can_start_new_session.return_value = False
    strategy._stop_checker.can_send_any_turn.return_value = True
    issuer.issue_credit.return_value = False
    task = runtime.create_task(strategy.execute_phase())
    await runtime.drain_current()
    assert runtime.activity.idle
    turn = TurnToSend("conversation", "session", 1, 2)
    runtime.call_later(
        2, lambda: runtime.create_task(strategy._queue_continuation(turn))
    )
    runtime.advance_to(2_000_000_000)
    await runtime.drain_current()
    await task
    issuer.issue_credit.assert_awaited_once_with(turn)
    runtime.assert_drained()
