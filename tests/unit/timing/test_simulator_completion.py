# SPDX-License-Identifier: Apache-2.0
"""Final result publication cannot race simulator cleanup or hide its failure."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.common.models.branch_stats import BranchStats
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditsCompleteMessage,
)
from aiperf.plugin.enums import TimingMode
from aiperf.simulation.completion import SimulationPhasePublisher
from aiperf.simulation.protocol import SimulationError
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.manager import TimingManager


def publisher(final_index: int = 0) -> tuple[SimulationPhasePublisher, Mock]:
    """Create the opt-in publisher with a recording message bus."""
    client = Mock(publish=AsyncMock())
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        phase_index=final_index,
        timing_mode=TimingMode.AGENTIC_REPLAY,
    )
    return SimulationPhasePublisher(
        pub_client=client, service_id="timing", final_phase_config=config
    ), client


@pytest.mark.asyncio
async def test_warmup_and_progress_publish_but_final_success_waits_for_flush() -> None:
    gate, client = publisher()
    warmup = CreditPhaseStats(phase=CreditPhase.WARMUP, phase_index=None, start_ns=1)
    profile = CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=0, start_ns=2)
    branches = BranchStats()
    await gate.publish_phase_complete(warmup)
    await gate.publish_phase_sending_complete(profile)
    await gate.publish_progress(profile)
    await gate.publish_phase_complete(profile, branches)
    await gate.publish_credits_complete()
    messages = [call.args[0] for call in client.publish.await_args_list]
    assert [type(message) for message in messages] == [
        CreditPhaseCompleteMessage,
        CreditPhaseSendingCompleteMessage,
        CreditPhaseProgressMessage,
    ]
    assert messages[0].stats.phase == CreditPhase.WARMUP
    await gate.flush()
    final, complete = [call.args[0] for call in client.publish.await_args_list[-2:]]
    assert isinstance(final, CreditPhaseCompleteMessage)
    assert final.stats is profile
    assert final.branch_stats is branches
    assert isinstance(complete, CreditsCompleteMessage)


@pytest.mark.asyncio
async def test_only_last_profile_is_held_and_phase_indices_are_preserved() -> None:
    gate, client = publisher(final_index=1)
    first = CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=0)
    final = CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=1)
    await gate.publish_phase_complete(first)
    await gate.publish_phase_complete(final)
    await gate.publish_credits_complete()
    client.publish.assert_awaited_once()
    assert client.publish.call_args.args[0].stats is first
    await gate.flush()
    assert client.publish.await_args_list[1].args[0].stats is final


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["phase", "credits"])
async def test_incomplete_lifecycle_cannot_publish_success(missing: str) -> None:
    gate, client = publisher()
    if missing != "phase":
        await gate.publish_phase_complete(
            CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=0)
        )
    if missing != "credits":
        await gate.publish_credits_complete()
    with pytest.raises(SimulationError, match="completion is missing"):
        await gate.flush()
    client.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_publication_does_not_send_credits_complete_or_retry() -> None:
    gate, client = publisher()
    await gate.publish_phase_complete(
        CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=0)
    )
    await gate.publish_credits_complete()
    client.publish.side_effect = RuntimeError("message bus stopped")
    with pytest.raises(RuntimeError, match="message bus stopped"):
        await gate.flush()
    client.publish.assert_awaited_once()
    assert isinstance(client.publish.call_args.args[0], CreditPhaseCompleteMessage)
    with pytest.raises(SimulationError, match="already consumed"):
        await gate.flush()
    client.publish.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("driver_fails", [False, True])
async def test_manager_only_flushes_after_driver_finishes_successfully(
    driver_fails: bool,
) -> None:
    gate, client = publisher()
    phases_done, driver_waiting, finish_driver = (asyncio.Event() for _ in range(3))

    async def phases() -> None:
        await gate.publish_phase_complete(
            CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=0)
        )
        await gate.publish_credits_complete()
        phases_done.set()

    async def drive() -> None:
        driver_waiting.set()
        await finish_driver.wait()
        if driver_fails:
            raise SimulationError("native cleanup failed")

    manager = SimpleNamespace(
        _simulator=SimpleNamespace(drive=drive),
        _phase_orchestrator=SimpleNamespace(start=phases),
        _simulation_publisher=gate,
    )
    task = asyncio.create_task(TimingManager._run_profiling(manager))
    await phases_done.wait()
    await driver_waiting.wait()
    client.publish.assert_not_awaited()
    assert not task.done()
    finish_driver.set()
    if driver_fails:
        with pytest.raises(SimulationError, match="native cleanup failed"):
            await task
        client.publish.assert_not_awaited()
    else:
        await task
        assert [type(call.args[0]) for call in client.publish.await_args_list] == [
            CreditPhaseCompleteMessage,
            CreditsCompleteMessage,
        ]
