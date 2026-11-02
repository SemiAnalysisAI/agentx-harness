# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``RecordsManager._process_results()``.

The ``accumulator`` plugin category drives metric summarization:
``MetricsAccumulator`` returns :class:`AccumulatorMetricsSummary`
(``results: dict[tag, MetricResult]``, ``timeslices``); GPU telemetry /
server metrics accumulators return list-shaped results.

The pipeline:

1. ``_deliver_network_rtt_to_accumulators`` wires optional RTT calibration.
2. ``_summarize_metric_record_accumulators`` exports metric-record accumulators.
3. ``_finalize_stream_exporters`` flushes JSONL writers concurrently.
4. ``ProcessRecordsResultMessage`` is published.
5. ``ProcessAllResultsMessage`` is published for the SystemController fan-in.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    ProcessAllResultsMessage,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProfileCancelCommand,
)
from aiperf.common.models import (
    MetricResult,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProfileMetricDurationCoverage,
    TimesliceResult,
)
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.plugin.enums import AccumulatorType, StreamExporterType
from aiperf.records.records_manager import RecordsManager

# ---------------------------------------------------------------------------
# Stub fixtures
# ---------------------------------------------------------------------------


_STUB_METRIC_RESULT = MetricResult(
    tag="request_latency",
    header="Request Latency",
    unit="ms",
    avg=100.0,
    count=10,
)


def _make_summary_accumulator(
    results: list[MetricResult] | None = None,
    *,
    timeslices: list[TimesliceResult] | None = None,
    summarize_exc: BaseException | None = None,
) -> MagicMock:
    """Stub for an ``AccumulatorProtocol`` returning :class:`AccumulatorMetricsSummary`."""
    acc = MagicMock()
    acc.__class__.__name__ = "StubMetricsAccumulator"
    if summarize_exc is not None:
        acc.summarize = AsyncMock(side_effect=summarize_exc)
    else:
        results_dict = {
            r.tag: r
            for r in (results if results is not None else [_STUB_METRIC_RESULT])
        }
        acc.summarize = AsyncMock(
            return_value=AccumulatorMetricsSummary(
                results=results_dict,
                timeslices=timeslices,
            )
        )
    return acc


def _make_list_accumulator(
    results: list[MetricResult] | None = None,
    summarize_exc: BaseException | None = None,
) -> MagicMock:
    """Stub for an accumulator returning ``list[MetricResult]``."""
    acc = MagicMock()
    acc.__class__.__name__ = "StubListAccumulator"
    if summarize_exc is not None:
        acc.summarize = AsyncMock(side_effect=summarize_exc)
    else:
        acc.summarize = AsyncMock(
            return_value=results if results is not None else [_STUB_METRIC_RESULT]
        )
    return acc


def _make_stub_stream_exporter() -> MagicMock:
    exp = MagicMock()
    exp.finalize = AsyncMock()
    return exp


def _make_manager_mock(
    *,
    accumulators: dict[AccumulatorType, MagicMock] | None = None,
    stream_exporters: dict[StreamExporterType, MagicMock] | None = None,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
    user_config_telemetry_disabled: bool = True,
    user_config_server_metrics_disabled: bool = True,
) -> MagicMock:
    """Build a mock ``RecordsManager`` with the unified pipeline methods bound.

    GPU telemetry / server metrics accumulators are absent by default and
    the user_config flags disable both side-channel publishes — those
    paths are exercised by separate target-side tests, not here.
    """
    mgr = MagicMock()
    mgr._accumulators = accumulators or {}
    # The byte-exact summary engine only summarizes accumulators that are
    # registered as metric_record-typed; mirror the production gate by treating
    # every supplied accumulator as a metric_record accumulator here.
    mgr._metric_record_accumulators = list((accumulators or {}).values())
    mgr._stream_exporters = stream_exporters or {}
    mgr._gpu_telemetry_accumulator = None
    mgr._server_metrics_accumulator = None
    # Branch-stats snapshot (read by _process_results).
    mgr._latest_branch_stats = None

    # Records tracker — drives the time window via PROFILING phase stats.
    phase_stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        start_ns=start_ns,
        requests_end_ns=end_ns,
    )
    mgr._records_tracker.create_stats_for_phase.return_value = phase_stats

    # Error tracker — empty errors keep the success path.
    mgr._error_tracker.get_error_summary_for_phase.return_value = []

    # v2 config — disable telemetry / server-metrics side channels via run.cfg.
    mgr.run = MagicMock()
    mgr.run.cfg.gpu_telemetry_disabled = user_config_telemetry_disabled
    mgr.run.cfg.server_metrics_disabled = user_config_server_metrics_disabled
    mgr.run.cfg.scenario = None
    mgr.run.cfg.get_profiling_phases.return_value = []

    # Logging
    mgr.debug = MagicMock()
    mgr.info = MagicMock()
    mgr.error = MagicMock()
    mgr.warning = MagicMock()
    mgr.exception = MagicMock()

    # Service identity + publish
    mgr.service_id = "test_records_manager"
    mgr.publish = AsyncMock()

    # Single-flight guard state read by the real _process_results wrapper.
    mgr._process_results_lock = asyncio.Lock()
    mgr._processed_results = {}

    # Bind real methods
    mgr._process_results = RecordsManager._process_results.__get__(mgr)
    mgr._process_results_impl = RecordsManager._process_results_impl.__get__(mgr)
    mgr._summarize_metric_record_accumulators = (
        RecordsManager._summarize_metric_record_accumulators.__get__(mgr)
    )
    mgr._has_records_for_phase = RecordsManager._has_records_for_phase.__get__(mgr)
    mgr._summarize_warmup_metric_records = (
        RecordsManager._summarize_warmup_metric_records.__get__(mgr)
    )
    mgr._validate_profile_metric_duration_coverage = (
        RecordsManager._validate_profile_metric_duration_coverage.__get__(mgr)
    )
    mgr._iter_concrete_phase_stats = RecordsManager._iter_concrete_phase_stats.__get__(
        mgr
    )
    mgr._summarize_one_accumulator = RecordsManager._summarize_one_accumulator.__get__(
        mgr
    )
    mgr._bucket_accumulator_summary = (
        RecordsManager._bucket_accumulator_summary.__get__(mgr)
    )
    mgr._analyzers = []
    mgr._run_analyzers = RecordsManager._run_analyzers.__get__(mgr)
    mgr._finalize_stream_exporters = RecordsManager._finalize_stream_exporters.__get__(
        mgr
    )
    mgr._publish_all_results = RecordsManager._publish_all_results.__get__(mgr)
    mgr._publish_telemetry_results = RecordsManager._publish_telemetry_results.__get__(
        mgr
    )
    mgr._publish_server_metrics_results = (
        RecordsManager._publish_server_metrics_results.__get__(mgr)
    )

    return mgr


# ---------------------------------------------------------------------------
# Tests: accumulator summarize fan-out
# ---------------------------------------------------------------------------


class TestProcessResultsAccumulatorPath:
    """``_process_results`` runs ``summarize`` on every accumulator and bridges
    both the typed :class:`AccumulatorMetricsSummary` shape and the legacy
    ``list[MetricResult]`` shape into the published
    :class:`ProcessRecordsResultMessage`."""

    @pytest.mark.asyncio
    async def test_calls_summarize_on_all_accumulators(self) -> None:
        acc1 = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc2 = _make_list_accumulator([])

        mgr = _make_manager_mock(
            accumulators={
                AccumulatorType.METRIC_RESULTS: acc1,
                AccumulatorType.GPU_TELEMETRY: acc2,
            }
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        acc1.summarize.assert_awaited_once()
        acc2.summarize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publishes_process_records_result_message(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessRecordsResultMessage) for m in published)

    @pytest.mark.asyncio
    async def test_returns_process_records_result(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert isinstance(result, ProcessRecordsResult)
        assert result.results.records is not None
        assert _STUB_METRIC_RESULT in result.results.records

    @pytest.mark.asyncio
    async def test_legacy_list_shape_accumulator_results_extended(self) -> None:
        """``list[MetricResult]`` accumulator output is appended to records."""
        acc_list = _make_list_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.GPU_TELEMETRY: acc_list})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert _STUB_METRIC_RESULT in (result.results.records or [])

    @pytest.mark.asyncio
    async def test_accumulator_summarize_failure_does_not_abort(self) -> None:
        """A failing summarize is wrapped into ``result.errors`` but the
        unified pipeline still runs."""
        failing = _make_summary_accumulator(
            summarize_exc=RuntimeError("summarize boom")
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: failing})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        # Errors logged + included in result.errors
        mgr.error.assert_called()
        assert any("summarize boom" in str(err.message or err) for err in result.errors)

    @pytest.mark.asyncio
    async def test_agentx_metric_coverage_failure_is_fatal(self) -> None:
        """A duration-based AgentX phase below 95% remains exportable but fatal."""
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc.profile_metric_duration_coverage.return_value = (
            ProfileMetricDurationCoverage(
                phase_name="profiling",
                expected_duration_seconds=3600.0,
                required_ratio=0.98,
                ttft_ratio=0.861194,
                inter_token_latency_ratio=0.861569,
            )
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 3600.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert len(result.fatal_errors) == 1
        assert result.fatal_errors[0].type == "ProfileMetricCoverageError"
        assert "required 95.0%" in result.fatal_errors[0].message
        assert "check inference server logs" in result.fatal_errors[0].message
        assert result.results.runtime_submission_invalid_reasons == [
            "insufficient_profile_metric_coverage"
        ]
        assert result.results.metric_duration_coverage[0].ttft_ratio == pytest.approx(
            0.861194
        )

    @pytest.mark.asyncio
    async def test_agentx_metric_coverage_passes_at_threshold(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc.profile_metric_duration_coverage.return_value = (
            ProfileMetricDurationCoverage(
                phase_name="profiling",
                expected_duration_seconds=3600.0,
                required_ratio=0.98,
                ttft_ratio=0.98,
                inter_token_latency_ratio=0.999,
            )
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 3600.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.fatal_errors == []
        assert result.results.runtime_submission_invalid_reasons == []

    @pytest.mark.asyncio
    async def test_agentx_metric_coverage_accepts_late_streaming_activity(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        acc.profile_metric_duration_coverage.return_value = (
            ProfileMetricDurationCoverage(
                phase_name="profiling",
                expected_duration_seconds=3600.0,
                required_ratio=0.98,
                ttft_ratio=0.979504,
                inter_token_latency_ratio=1.0,
            )
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 3600.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.fatal_errors == []
        assert result.results.runtime_submission_invalid_reasons == []

    @pytest.mark.asyncio
    async def test_agentx_short_unsafe_smoke_skips_metric_coverage(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 30.0
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        acc.profile_metric_duration_coverage.assert_not_called()
        assert result.fatal_errors == []

    @pytest.mark.asyncio
    async def test_empty_accumulators_produces_empty_records(self) -> None:
        mgr = _make_manager_mock(accumulators={})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert isinstance(result, ProcessRecordsResult)
        assert result.results.records == []

    @pytest.mark.asyncio
    async def test_timeslices_propagated_to_profile_results(self) -> None:
        """``timeslices`` from AccumulatorMetricsSummary populates
        ``ProfileResults.timeslices``."""
        slice_metrics = {
            "request_latency": MetricResult(
                tag="request_latency",
                header="Latency",
                unit="ms",
                avg=100.0,
                count=5,
            )
        }
        timeslices = [
            TimesliceResult(
                start_ns=1_000_000_000,
                end_ns=2_000_000_000,
                metric_results=slice_metrics,
            )
        ]
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT], timeslices=timeslices)
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.results.timeslices is not None
        assert len(result.results.timeslices) == 1
        assert result.results.timeslices[0].start_ns == 1_000_000_000
        assert result.results.timeslices[0].metric_results == slice_metrics


# ---------------------------------------------------------------------------
# Tests: cancelled flag propagation
# ---------------------------------------------------------------------------


class TestProcessResultsCancelled:
    @pytest.mark.asyncio
    async def test_cancelled_true_propagated_to_profile_results(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=True)

        assert result.results.was_cancelled is True

    @pytest.mark.asyncio
    async def test_cancelled_false_propagated_to_profile_results(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.results.was_cancelled is False


class TestProcessResultsMeasurementStart:
    """A measurement start moves the root summary, its start time and the
    coverage check to the measured span."""

    _START_NS = 1_000_000_000
    _CUT_NS = 721_000_000_000

    @staticmethod
    def _phase_scoped_accumulator(request_count: float) -> MagicMock:
        acc = MagicMock()
        acc.__class__.__name__ = "StubMetricsAccumulator"
        acc.supports_phase_scoped_export = True
        acc.export_results = AsyncMock(
            return_value=AccumulatorMetricsSummary(
                results={
                    "request_count": MetricResult(
                        tag="request_count",
                        header="Request Count",
                        unit="requests",
                        avg=request_count,
                    )
                },
                timeslices=None,
            )
        )
        return acc

    def _manager(self, acc: MagicMock, *, measurement_start: object) -> MagicMock:
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            start_ns=self._START_NS,
            end_ns=4_000_000_000_000,
        )
        mgr._has_multiple_phase_instances.return_value = False
        phase_config = MagicMock()
        phase_config.name = "profiling"
        phase_config.duration = 3600.0
        phase_config.measurement_start = measurement_start
        mgr.run.cfg.get_profiling_phases.return_value = [phase_config]
        return mgr

    @pytest.mark.asyncio
    async def test_root_summary_starts_at_measurement_start(self) -> None:
        acc = self._phase_scoped_accumulator(3.0)
        mgr = self._manager(acc, measurement_start=720.0)

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        ctx = acc.export_results.await_args.args[0]
        assert ctx.start_ns == self._CUT_NS
        assert ctx.min_request_start_ns == self._CUT_NS
        assert result.results.start_ns == self._CUT_NS
        assert result.results.measurement_start_sec == 720.0
        mgr.warning.assert_not_called()

    @pytest.mark.parametrize("measurement_start", [None, 0.0])
    @pytest.mark.asyncio
    async def test_no_measurement_start_keeps_phase_start(
        self, measurement_start: float | None
    ) -> None:
        acc = self._phase_scoped_accumulator(3.0)
        mgr = self._manager(acc, measurement_start=measurement_start)

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        ctx = acc.export_results.await_args.args[0]
        assert ctx.start_ns == self._START_NS
        assert ctx.min_request_start_ns is None
        assert result.results.start_ns == self._START_NS
        assert result.results.measurement_start_sec is None

    @pytest.mark.asyncio
    async def test_empty_measured_span_warns(self) -> None:
        acc = self._phase_scoped_accumulator(0.0)
        mgr = self._manager(acc, measurement_start=720.0)

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=True)

        assert "No request started after the measurement start" in str(
            mgr.warning.call_args
        )

    @pytest.mark.asyncio
    async def test_coverage_measured_from_measurement_start(self) -> None:
        acc = self._phase_scoped_accumulator(3.0)
        acc.profile_metric_duration_coverage.return_value = (
            ProfileMetricDurationCoverage(
                phase_name="profiling",
                expected_duration_seconds=3600.0,
                required_ratio=0.95,
                ttft_ratio=1.0,
                inter_token_latency_ratio=1.0,
            )
        )
        mgr = self._manager(acc, measurement_start=720.0)
        mgr.run.cfg.scenario = "inferencex-agentx-mvp"

        result = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert result.fatal_errors == []
        call = acc.profile_metric_duration_coverage.call_args
        assert call.args[0].start_ns == self._CUT_NS
        assert call.args[0].min_request_start_ns == self._CUT_NS
        assert call.kwargs["expected_duration_seconds"] == 3600.0


# ---------------------------------------------------------------------------
# Tests: _finalize_stream_exporters integration
# ---------------------------------------------------------------------------


class TestProcessResultsStreamExporters:
    @pytest.mark.asyncio
    async def test_stream_exporters_finalized(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        exp = _make_stub_stream_exporter()
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        exp.finalize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_stream_exporters_is_noop(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={},
        )

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessAllResultsMessage) for m in published)


# ---------------------------------------------------------------------------
# Tests: analyzer execution + ProcessAllResultsMessage publish
# ---------------------------------------------------------------------------


def _get_published_all_results(mgr: MagicMock) -> ProcessAllResultsMessage | None:
    """Return the published ``ProcessAllResultsMessage`` if any."""
    for call in mgr.publish.await_args_list:
        msg = call.args[0]
        if isinstance(msg, ProcessAllResultsMessage):
            return msg
    return None


class TestProcessResultsAllResultsPublish:
    """``_process_results`` publishes :class:`ProcessAllResultsMessage` for the
    SystemController fan-in."""

    @pytest.mark.asyncio
    async def test_publishes_process_all_results_message(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})

        await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        msg = _get_published_all_results(mgr)
        assert msg is not None


class TestProcessResultsSingleFlight:
    """``_process_results`` is single-flight per phase: the natural finalize task
    and the PROCESS_RECORDS / PROFILE_CANCEL commands can all reach it, but only
    the first does the work; later calls return the cached result without
    re-publishing or re-finalizing the stream exporters."""

    @pytest.mark.asyncio
    async def test_second_call_returns_cached_and_does_not_republish(self) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        exp = _make_stub_stream_exporter()
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            stream_exporters={StreamExporterType.RECORD_EXPORT: exp},
        )

        first = await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)
        publish_count = mgr.publish.await_count

        second = await mgr._process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert second is first
        # No additional publishes and the exporter is finalized exactly once.
        assert mgr.publish.await_count == publish_count
        exp.finalize.assert_awaited_once()


class TestProcessResultsServerMetricsFailure:
    """A server-metrics processing failure must not abort the fan-in publish."""

    @pytest.mark.asyncio
    async def test_server_metrics_processing_failure_still_publishes_fanin(
        self,
    ) -> None:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        mgr = _make_manager_mock(
            accumulators={AccumulatorType.METRIC_RESULTS: acc},
            user_config_server_metrics_disabled=False,
        )
        mgr._process_server_metrics_results = AsyncMock(
            side_effect=ValueError("start_ns must be less than end_ns")
        )

        await mgr._process_results(phase=CreditPhase.WARMUP, cancelled=True)

        published = [c.args[0] for c in mgr.publish.await_args_list]
        assert any(isinstance(m, ProcessRecordsResultMessage) for m in published)
        server_metrics_messages = [
            m for m in published if isinstance(m, ProcessServerMetricsResultMessage)
        ]
        assert len(server_metrics_messages) == 1
        assert server_metrics_messages[0].server_metrics_result.results is None
        assert any(
            "Failed to process server metrics results" in str(call.args[0])
            for call in mgr.exception.call_args_list
        )


class TestProcessResultsWindowArtifacts:
    """``window_start_interval`` writes profiling suffix windows before the root publish."""

    @staticmethod
    def _windowed_manager(
        interval: float | None, *, export_exc: BaseException | None = None
    ) -> tuple[MagicMock, MagicMock, list[str]]:
        acc = _make_summary_accumulator([_STUB_METRIC_RESULT])
        count = MetricResult(
            tag="request_count", header="Request Count", unit="requests", avg=3
        )
        acc.export_window_results = AsyncMock(
            side_effect=export_exc,
            return_value=AccumulatorMetricsSummary(results={count.tag: count}),
        )
        mgr = _make_manager_mock(accumulators={AccumulatorType.METRIC_RESULTS: acc})
        mgr._window_export_cancelled = False
        mgr.run.cfg.artifacts.window_start_interval = interval
        mgr._records_tracker.create_aggregate_stats_for_phase.return_value = (
            PhaseRecordsStats(
                phase=CreditPhase.PROFILING,
                start_ns=1_000_000_000,
                requests_end_ns=2_000_000_000,
            )
        )
        events: list[str] = []
        exporter = MagicMock()
        exporter.export_window_artifacts = AsyncMock(
            side_effect=lambda window: events.append(window.phase_name) or {}
        )
        exporter.write_window_manifest = AsyncMock(
            side_effect=lambda entries: events.append("manifest")
        )

        async def publish(message) -> None:  # noqa: ANN001
            events.append(type(message).__name__)

        mgr.publish = AsyncMock(side_effect=publish)
        return mgr, exporter, events

    @pytest.mark.asyncio
    async def test_windows_are_written_before_root_result_publish(self) -> None:
        mgr, exporter, events = self._windowed_manager(0.5)

        with patch(
            "aiperf.records.records_manager.ExporterManager", return_value=exporter
        ):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        assert events[:4] == [
            "start_0s",
            "start_0.5s",
            "manifest",
            "ProcessRecordsResultMessage",
        ]

    @pytest.mark.asyncio
    async def test_profile_cancel_before_window_export_skips_windows(self) -> None:
        mgr, exporter, events = self._windowed_manager(0.5)
        mgr._window_export_cancelled = True

        with patch(
            "aiperf.records.records_manager.ExporterManager", return_value=exporter
        ):
            await mgr._process_results(phase=CreditPhase.PROFILING, cancelled=False)

        exporter.export_window_artifacts.assert_not_awaited()
        assert events[0] == "ProcessRecordsResultMessage"

    @pytest.mark.asyncio
    async def test_profile_cancel_during_window_export_returns_staged_root(
        self,
    ) -> None:
        mgr, exporter, events = self._windowed_manager(0.5)
        mgr._records_tracker.mark_phase_cancelled = MagicMock()
        cancel_responses: list = []

        async def export_and_cancel(window):  # noqa: ANN001, ANN202
            events.append(window.phase_name)
            # Runs inside natural finalization, which holds the results lock:
            # the cancel must answer without waiting for it.
            cancel_responses.append(
                await asyncio.wait_for(
                    RecordsManager._on_profile_cancel_command(
                        mgr, ProfileCancelCommand(service_id="controller")
                    ),
                    timeout=1.0,
                )
            )
            return {}

        exporter.export_window_artifacts = AsyncMock(side_effect=export_and_cancel)

        with patch(
            "aiperf.records.records_manager.ExporterManager", return_value=exporter
        ):
            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        assert cancel_responses == [result]
        assert events[:3] == ["start_0s", "manifest", "ProcessRecordsResultMessage"]

    @pytest.mark.asyncio
    async def test_window_failure_keeps_root_result(self) -> None:
        mgr, exporter, events = self._windowed_manager(
            60, export_exc=RuntimeError("boom")
        )

        with patch(
            "aiperf.records.records_manager.ExporterManager", return_value=exporter
        ):
            result = await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=False
            )

        assert _STUB_METRIC_RESULT in result.results.records
        assert "ProcessRecordsResultMessage" in events
        exporter.write_window_manifest.assert_not_awaited()
        assert any(
            "Failed to export profiling windows" in str(call.args[0])
            for call in mgr.exception.call_args_list
        )

    @pytest.mark.parametrize("interval", [None, 60])
    @pytest.mark.asyncio
    async def test_no_windows_without_interval_or_on_cancelled_processing(
        self, interval: float | None
    ) -> None:
        mgr, exporter, events = self._windowed_manager(interval)

        with patch(
            "aiperf.records.records_manager.ExporterManager", return_value=exporter
        ):
            await mgr._process_results(
                phase=CreditPhase.PROFILING, cancelled=interval is not None
            )

        exporter.export_window_artifacts.assert_not_awaited()
