# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final result publication cannot race simulator cleanup or hide its failure."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import orjson
import pytest
from pytest import param

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
from aiperf.simulation.identity import VIRTUAL_TIMESTAMP_ORIGIN_NS
from aiperf.simulation.protocol import SimulationError
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.manager import TimingManager


def publisher(
    final_index: int = 0, *, artifact_dir: Path | None = None
) -> tuple[SimulationPhasePublisher, Mock]:
    """Create the opt-in publisher with a recording message bus."""
    client = Mock(publish=AsyncMock())
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        phase_index=final_index,
        timing_mode=TimingMode.AGENTIC_REPLAY,
    )
    return SimulationPhasePublisher(
        pub_client=client,
        service_id="timing",
        final_phase_config=config,
        artifact_dir=artifact_dir,
        benchmark_id="test-benchmark",
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


def measurement_config() -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        phase_index=0,
        phase_name="measurement",
        timing_mode=TimingMode.AGENTIC_REPLAY,
        expected_duration_sec=3600,
        grace_period_sec=30,
    )


async def read_phase_windows(artifact_dir: Path) -> dict:
    payload = await asyncio.to_thread((artifact_dir / "phase_windows.json").read_bytes)
    return orjson.loads(payload)


@pytest.mark.asyncio
async def test_phase_windows_preserve_exact_measurement_and_early_grace_end(
    tmp_path: Path,
) -> None:
    gate, _ = publisher(artifact_dir=tmp_path)
    config = measurement_config()
    start_ns = 2**53 + 1
    end_ns = start_ns + 3600 * 10**9
    start = CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        phase_index=0,
        start_ns=VIRTUAL_TIMESTAMP_ORIGIN_NS + start_ns,
    )
    await gate.publish_phase_start(config, start)
    snapshot = await read_phase_windows(tmp_path)
    assert snapshot["version"] == 1
    assert snapshot["clock"] == "virtual_ns"
    assert snapshot["timestamp_origin_ns"] == VIRTUAL_TIMESTAMP_ORIGIN_NS
    assert snapshot["benchmark_id"] == "test-benchmark"
    window = snapshot["phases"][0]
    assert type(window["start_ns"]) is int
    assert window["start_ns"] == start_ns
    assert window["measurement_end_ns"] == end_ns
    assert window["configured_duration_ns"] == 3600 * 10**9
    assert window["configured_grace_ns"] == 30 * 10**9
    assert window["grace_limit_ns"] == end_ns + 30 * 10**9
    assert window["sent_end_ns"] is window["requests_end_ns"] is None
    assert window["incomplete_measurement"] is None

    sending = start.model_copy(
        update={
            "sent_end_ns": VIRTUAL_TIMESTAMP_ORIGIN_NS + end_ns + 1,
            "timeout_triggered": True,
        }
    )
    await gate.publish_phase_sending_complete(sending)
    snapshot = await read_phase_windows(tmp_path)
    assert snapshot["phases"][0]["sent_end_ns"] == end_ns + 1
    assert snapshot["phases"][0]["measurement_end_ns"] == end_ns
    assert snapshot["phases"][0]["complete"] is False

    complete = sending.model_copy(
        update={"requests_end_ns": VIRTUAL_TIMESTAMP_ORIGIN_NS + end_ns + 7 * 10**9}
    )
    await gate.publish_phase_complete(complete)
    await gate.publish_credits_complete()
    before_flush = await read_phase_windows(tmp_path)
    window = before_flush["phases"][0]
    assert window["requests_end_ns"] == end_ns + 7 * 10**9
    assert window["grace_limit_ns"] == end_ns + 30 * 10**9
    assert window["complete"] is True
    assert window["incomplete_measurement"] is False
    assert window["grace_period_timeout_triggered"] is False
    assert before_flush["protocol_finished"] is False
    await gate.flush()
    after_flush = await read_phase_windows(tmp_path)
    assert after_flush["protocol_finished"] is True
    assert after_flush["phases"] == before_flush["phases"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sent_offset_ns,requests_offset_ns,cancelled,incomplete",
    [
        param(-1, 5 * 10**9, False, True, id="early-send-stop-grace-crosses-end"),
        param(-1, -1, False, True, id="early-phase-completion"),
        param(0, 30 * 10**9, True, True, id="cancelled-phase"),
        param(0, 0, False, False, id="complete-at-exact-duration"),
    ],
)  # fmt: skip
async def test_phase_windows_classify_incomplete_measurement_from_lifecycle(
    tmp_path: Path,
    sent_offset_ns: int,
    requests_offset_ns: int,
    cancelled: bool,
    incomplete: bool,
) -> None:
    gate, _ = publisher(artifact_dir=tmp_path)
    start_ns = VIRTUAL_TIMESTAMP_ORIGIN_NS + 100
    end_ns = start_ns + 3600 * 10**9
    start = CreditPhaseStats(
        phase=CreditPhase.PROFILING, phase_index=0, start_ns=start_ns
    )
    await gate.publish_phase_start(measurement_config(), start)
    complete = start.model_copy(
        update={
            "sent_end_ns": end_ns + sent_offset_ns,
            "requests_end_ns": end_ns + requests_offset_ns,
            "was_cancelled": cancelled,
        }
    )
    await gate.publish_phase_sending_complete(complete)
    await gate.publish_phase_complete(complete)
    window = (await read_phase_windows(tmp_path))["phases"][0]
    assert window["measurement_end_ns"] == end_ns - VIRTUAL_TIMESTAMP_ORIGIN_NS
    assert (
        window["sent_end_ns"] == end_ns + sent_offset_ns - VIRTUAL_TIMESTAMP_ORIGIN_NS
    )
    assert window["complete"] is True
    assert window["incomplete_measurement"] is incomplete


@pytest.mark.asyncio
async def test_phase_windows_warmup_has_no_synthetic_measurement_duration(
    tmp_path: Path,
) -> None:
    gate, _ = publisher(artifact_dir=tmp_path)
    config = CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        expected_duration_sec=None,
        grace_period_sec=float("inf"),
    )
    start = CreditPhaseStats(
        phase=CreditPhase.WARMUP,
        phase_index=None,
        start_ns=VIRTUAL_TIMESTAMP_ORIGIN_NS + 12,
    )
    await gate.publish_phase_start(config, start)
    complete = start.model_copy(
        update={
            "sent_end_ns": VIRTUAL_TIMESTAMP_ORIGIN_NS + 13,
            "requests_end_ns": VIRTUAL_TIMESTAMP_ORIGIN_NS + 15,
        }
    )
    await gate.publish_phase_sending_complete(complete)
    await gate.publish_phase_complete(complete)
    window = (await read_phase_windows(tmp_path))["phases"][0]
    assert window["phase"] == CreditPhase.WARMUP
    assert window["configured_duration_ns"] is None
    assert window["measurement_end_ns"] is None
    assert window["configured_grace_ns"] is None
    assert window["grace_limit_ns"] is None
    assert window["grace_unbounded"] is True
    assert window["start_ns"] == 12
    assert window["requests_end_ns"] == 15
    assert window["complete"] is True
    assert window["incomplete_measurement"] is False


@pytest.mark.asyncio
async def test_phase_windows_require_start_before_completion(tmp_path: Path) -> None:
    gate, _ = publisher(artifact_dir=tmp_path)
    with pytest.raises(SimulationError, match="boundary has no start"):
        await gate.publish_phase_complete(
            CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=0)
        )
    assert not await asyncio.to_thread((tmp_path / "phase_windows.json").exists)


@pytest.mark.asyncio
async def test_phase_windows_default_disabled_performs_no_file_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unexpected_io = Mock(side_effect=AssertionError("default publisher performed I/O"))
    for name in ("mkdir", "write_bytes", "replace"):
        monkeypatch.setattr(Path, name, unexpected_io)
    gate, _ = publisher()
    stats = CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=0)
    await gate.publish_phase_start(measurement_config(), stats)
    await gate.publish_phase_sending_complete(stats)
    await gate.publish_phase_complete(stats)
    await gate.publish_credits_complete()
    await gate.flush()
    unexpected_io.assert_not_called()
