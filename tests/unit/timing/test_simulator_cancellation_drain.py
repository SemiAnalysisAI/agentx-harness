# SPDX-License-Identifier: Apache-2.0
"""Grace cancellation must drain native callbacks before virtual completion."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models.branch_stats import BranchStats
from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.simulation.identity import request_id
from aiperf.simulation.protocol import Cancel, Grant, Ready, Submit, Terminal
from aiperf.simulation.router import SimulatorCreditRouter
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker_returns_first",
    [param(True, id="worker-first"), param(False, id="terminal-first")],
)  # fmt: skip
async def test_grace_cancellation_drains_callback_with_pending_dag(
    monkeypatch: pytest.MonkeyPatch, worker_returns_first: bool
) -> None:
    runtime = VirtualTimingRuntime()
    client = Mock(request=AsyncMock())
    monkeypatch.setattr("aiperf.simulation.router.CoordinatorClient", lambda _: client)
    native = Mock(send_credit=AsyncMock(), cancel_all_credits=AsyncMock())
    router = SimulatorCreditRouter(
        native,
        benchmark_id="drain",
        endpoint="tcp://127.0.0.1:1",
        runtime=runtime,
        timeout_s=1.0,
    )
    publisher = Mock(publish_progress=AsyncMock(), publish_phase_complete=AsyncMock())
    concurrency = Mock()
    # Duration cutoff can leave a join pending after its last wire request ends.
    branches = Mock(
        has_pending_branch_work=Mock(return_value=True),
        snapshot_branch_stats=Mock(return_value=BranchStats()),
    )
    callback = CreditCallbackHandler(concurrency, branch_orchestrator=branches)
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        phase_index=0,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        expected_duration_sec=1.0,
        grace_period_sec=1.0,
    )
    runner = PhaseRunner(
        config=config,
        conversation_source=Mock(dataset_metadata=None),
        phase_publisher=publisher,
        credit_router=router,
        concurrency_manager=concurrency,
        callback_handler=callback,
        branch_orchestrator=branches,
        runtime=runtime,
    )
    callback.register_phase(
        phase=config.phase,
        phase_index=config.phase_index,
        progress=runner._progress,
        lifecycle=runner._lifecycle,
        stop_checker=runner._stop_checker,
        strategy=SimpleNamespace(),
    )
    router.set_return_callback(callback.on_credit_return)
    credit = Credit(
        id=0,
        phase=config.phase,
        phase_index=config.phase_index,
        conversation_id="trace",
        x_correlation_id="session",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
    )
    rid = request_id("drain", credit.phase_index, credit.id)
    returned = CreditReturn(credit=credit, cancelled=True)
    receive_return = native.set_return_callback.call_args.args[0]
    terminal_granted = asyncio.Event()
    callback_entered = asyncio.Event()
    release_callback = asyncio.Event()
    callback_finished = asyncio.Event()

    async def intercept(_: Credit) -> bool:
        callback_entered.set()
        await runtime.wait_wall_event(release_callback, timeout=1.0)
        callback_finished.set()
        return True

    branches.intercept = AsyncMock(side_effect=intercept)

    async def exchange(ready: Ready) -> Grant:
        if ready.round == 0:
            assert ready.operations == [Submit(rid)]
            assert ready.next_wakeup_ns == 2_000_000_000
            return Grant("drain", ready.round, ready.next_wakeup_ns, [])
        if ready.round == 1:
            assert ready.operations == [Cancel(rid)]
            assert runner._progress.in_flight == 1
            assert not runner._lifecycle.is_complete
            native.cancel_all_credits.assert_awaited_once()
            if worker_returns_first:
                await receive_return("worker", returned)
                assert not callback_entered.is_set()
            terminal_granted.set()
            return Grant(
                "drain",
                ready.round,
                runtime.now_ns(),
                [Terminal(rid, "CANCELLED", runtime.now_ns())],
            )
        assert ready.round == 2
        assert ready.finished
        assert callback_finished.is_set()
        assert runner._lifecycle.is_complete
        return Grant("drain", ready.round, runtime.now_ns(), [], status="FINISHED")

    client.request.side_effect = exchange

    async def run_phase() -> None:
        with runtime.run_scope():
            runner._lifecycle.start()
            runner._progress.increment_sent(TurnToSend("trace", "session", 0, 2))
            await router.send_credit(credit)
            runner._progress.freeze_sent_counts()
            runner._progress.all_credits_sent_event.set()
            runner._lifecycle.mark_sending_complete(timeout_triggered=True)
            await runner._wait_for_returning_complete()

    phase_task = asyncio.create_task(run_phase())
    driver_task = asyncio.create_task(router.drive())
    try:
        await asyncio.wait_for(terminal_granted.wait(), timeout=1.0)
        if not worker_returns_first:
            assert not callback_entered.is_set()
            assert runner._progress.in_flight == 1
            await receive_return("worker", returned)
        await asyncio.wait_for(callback_entered.wait(), timeout=1.0)
        await asyncio.wait_for(runtime.activity.wait_idle(), timeout=1.0)
        assert runner._progress.in_flight == 0
        assert not runner._progress.all_credits_returned_event.is_set()
        assert not runner._lifecycle.is_complete
        assert not phase_task.done()
        publisher.publish_phase_complete.assert_not_awaited()
        release_callback.set()
        await asyncio.wait_for(asyncio.gather(phase_task, driver_task), timeout=2.0)
    finally:
        for task in (phase_task, driver_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(phase_task, driver_task, return_exceptions=True)
        router.close()

    stats = publisher.publish_phase_complete.call_args.args[0]
    assert stats.final_requests_sent == 1
    assert stats.final_requests_completed == 0
    assert stats.final_requests_cancelled == 1
    assert stats.grace_period_timeout_triggered
    assert runtime.now_ns() == 2_000_000_000
    assert runtime.finished.is_set()
    assert client.request.await_args_list[-1].args[0].finished
    runtime.assert_drained()
