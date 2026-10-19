# SPDX-License-Identifier: Apache-2.0
"""The production issuer observes the virtual phase's duration boundary."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_index,depth", [(0, 0), (1, 0), (0, 1), (1, 1)])
async def test_no_root_or_child_credit_at_duration_deadline(
    turn_index: int, depth: int
) -> None:
    runtime = VirtualTimingRuntime(timestamp_origin_ns=100)
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        expected_duration_sec=1,
        grace_period_sec=5,
    )
    lifecycle = PhaseLifecycle(config, runtime=runtime)
    progress = PhaseProgressTracker(config)
    stop_checker = StopConditionChecker(
        config=config, lifecycle=lifecycle, counter=progress.counter
    )
    router = MagicMock(send_credit=AsyncMock())
    issuer = CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=stop_checker,
        progress=progress,
        concurrency_manager=ConcurrencyManager(runtime=runtime),
        credit_router=router,
        cancellation_policy=MagicMock(
            next_cancellation_delay_ns=MagicMock(return_value=None)
        ),
        lifecycle=lifecycle,
        runtime=runtime,
    )
    lifecycle.start()
    turn = TurnToSend("trace", "session", turn_index, 3, agent_depth=depth)
    runtime.advance_to(999_999_999)
    assert await issuer.issue_credit(turn)
    credit = router.send_credit.call_args.kwargs["credit"]
    assert credit.issued_at_ns == 1_000_000_099
    runtime.advance_to(1_000_000_000)
    assert not await issuer.issue_credit(turn)
    router.send_credit.assert_awaited_once()
    assert lifecycle.time_left_in_seconds(include_grace_period=True) == 5
