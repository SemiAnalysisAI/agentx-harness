# SPDX-License-Identifier: Apache-2.0
"""Cancelled virtual runs must never invent physical phase boundaries."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.enums import CreditPhase, ProfileCancelReason
from aiperf.common.messages import ProfileCancelCommand
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import CreditPhaseStats, MetricRecordMetadata
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.plugin.enums import AccumulatorType
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_tracker import RecordsTracker
from aiperf.simulation.identity import VIRTUAL_TIMESTAMP_ORIGIN_NS


@pytest.mark.asyncio
@pytest.mark.parametrize("simulator", [False, True])
@pytest.mark.parametrize(
    ("started", "complete"),
    [pytest.param(False, False, id="warmup-failure"),
     pytest.param(True, False, id="profile-failure"),
     pytest.param(True, True, id="committed-profile-boundary")],
)  # fmt: skip
async def test_cancel_preserves_virtual_records_without_wall_clock_fallback(
    benchmark_run: BenchmarkRun,
    monkeypatch: pytest.MonkeyPatch,
    simulator: bool,
    started: bool,
    complete: bool,
) -> None:
    benchmark_run.cfg.endpoint.transport = "sglang_simulator" if simulator else "http"
    benchmark_run.cfg.gpu_telemetry.enabled = False
    benchmark_run.cfg.server_metrics.enabled = False
    benchmark_run.cfg.scenario = None
    manager = RecordsManager.__new__(RecordsManager)
    manager.run = benchmark_run
    manager.service_id = "records"
    manager.debug = manager.info = manager.error = manager.warning = Mock()
    manager.publish = AsyncMock()
    manager._records_tracker = RecordsTracker()
    manager._error_tracker = ErrorTracker()
    manager._latest_branch_stats = None
    manager._skipped_context_overflow_counts_by_phase = {}
    manager._process_results_lock = asyncio.Lock()
    manager._processed_results = {}
    manager._stream_exporters = {}
    manager._analyzers = []
    accumulator = MetricsAccumulator(run=benchmark_run)
    manager._accumulators = {AccumulatorType.METRIC_RESULTS: accumulator}
    manager._metric_record_accumulators = [accumulator]

    origin = VIRTUAL_TIMESTAMP_ORIGIN_NS
    phase = CreditPhase.PROFILING if started else CreditPhase.WARMUP
    manager._records_tracker.update_phase_info(
        CreditPhaseStats(
            phase=phase,
            phase_index=0 if started else None,
            start_ns=origin,
            requests_end_ns=origin + 50_000_000 if complete else None,
        )
    )
    metadata = MetricRecordMetadata(
        session_num=0,
        request_start_ns=origin + 10_000_000,
        request_end_ns=origin + 40_000_000,
        worker_id="worker",
        record_processor_id="processor",
        benchmark_phase=phase,
        phase_index=0 if started else None,
    )
    await accumulator.process_record(
        MetricRecordsData(
            metadata=metadata,
            metrics={
                "request_latency": 30_000_000,
                "min_request_timestamp": origin + 10_000_000,
                "max_response_timestamp": origin + 40_000_000,
                "input_sequence_length": 7,
                "output_sequence_length": 3,
            },
        )
    )
    manager._records_tracker.update_from_request(metadata, None)

    physical_ns = origin + 10**16
    physical_clock = Mock(return_value=physical_ns)
    monkeypatch.setattr(time, "time_ns", physical_clock)
    result = await manager._on_profile_cancel_command(
        ProfileCancelCommand(
            service_id="controller", reason=ProfileCancelReason.SERVICE_ERROR
        )
    )

    assert result.results.was_cancelled
    missing_boundary = 0 if simulator else physical_ns
    assert result.results.start_ns == (origin if started else missing_boundary)
    assert result.results.end_ns == (
        origin + 50_000_000 if complete else missing_boundary
    )
    assert physical_clock.call_count == (
        0 if simulator else int(not started) + int(not complete)
    )
    records = result.results.records if started else result.results.warmup_records
    assert records is not None
    latency = next(record for record in records if record.tag == "request_latency")
    assert latency.avg == 30  # Display milliseconds, from the virtual record.
    duration = next(record for record in records if record.tag == "benchmark_duration")
    # The existing column store represents epoch nanoseconds as float64.
    assert duration.avg == pytest.approx(0.03, abs=256e-9)
    assert accumulator.record_count == 1
    assert accumulator.column_store.end_ns[0] == origin + 40_000_000
