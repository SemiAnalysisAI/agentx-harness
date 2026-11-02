# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Suffix-window summaries of the profiling phase."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.models import MetricResult, PhaseRecordsStats
from aiperf.common.models.server_metrics_models import (
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
    ServerMetricsResults,
)
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.records.window_results import iter_window_profile_results, window_label

_START_NS = 10 * NANOS_PER_SECOND
_END_NS = _START_NS + 150 * NANOS_PER_SECOND
_GRACE_NS = 7


def _server_results(ctx: ExportContext) -> ServerMetricsResults:
    return ServerMetricsResults(
        endpoint_summaries={
            "node1:8081": ServerMetricsEndpointSummary(
                endpoint_url="http://node1:8081/metrics",
                info=ServerMetricsEndpointInfo(
                    total_fetches=1,
                    first_fetch_ns=ctx.start_ns,
                    last_fetch_ns=ctx.start_ns,
                    avg_fetch_latency_ms=1.0,
                    unique_updates=1,
                    first_update_ns=ctx.start_ns,
                    last_update_ns=ctx.start_ns,
                    duration_seconds=0.0,
                    avg_update_interval_ms=0.0,
                ),
                metrics={},
            )
        },
        start_ns=ctx.start_ns,
        end_ns=ctx.end_ns,
        endpoints_configured=["http://node1:8081/metrics"],
        endpoints_successful=["http://node1:8081/metrics"],
    )


class _FakeMetricsAccumulator:
    """Window k has 10*(k+1) successes and k+1 errors until ``empty_from``."""

    def __init__(self, empty_from: int | None = None) -> None:
        self.contexts: list[ExportContext] = []
        self._empty_from = empty_from

    async def export_window_results(
        self, ctx: ExportContext
    ) -> AccumulatorMetricsSummary:
        self.contexts.append(ctx)
        k = len(self.contexts)
        if self._empty_from is not None and k > self._empty_from:
            return AccumulatorMetricsSummary(results={})
        return AccumulatorMetricsSummary(
            results={
                "request_count": MetricResult(
                    tag="request_count",
                    header="Request Count",
                    unit="requests",
                    avg=10 * k,
                ),
                "error_request_count": MetricResult(
                    tag="error_request_count",
                    header="Error Count",
                    unit="requests",
                    avg=k,
                ),
            }
        )


class _FakeServerMetricsAccumulator:
    def __init__(self) -> None:
        self.contexts: list[ExportContext] = []

    async def export_window_results(self, ctx: ExportContext) -> ServerMetricsResults:
        self.contexts.append(ctx)
        return _server_results(ctx)


class _FailingTelemetryAccumulator:
    def __init__(self) -> None:
        self.contexts: list[ExportContext] = []

    async def export_results(self, ctx: ExportContext) -> None:
        self.contexts.append(ctx)
        raise RuntimeError("dcgm gone")


def _stats(
    start_ns: int | None = _START_NS, end_ns: int | None = _END_NS
) -> PhaseRecordsStats:
    return PhaseRecordsStats(
        phase=CreditPhase.PROFILING, start_ns=start_ns, requests_end_ns=end_ns
    )


async def _collect(**overrides) -> list:
    kwargs = {
        "metrics_accumulator": _FakeMetricsAccumulator(),
        "server_metrics_accumulator": None,
        "gpu_telemetry_accumulator": None,
        "stats": _stats(),
        "interval_sec": 60,
        "max_windows": 256,
        "final_scrape_grace_ns": _GRACE_NS,
        **overrides,
    }
    return [window async for window in iter_window_profile_results(**kwargs)]


@pytest.mark.parametrize(
    "offset_ns,expected",
    [
        param(0, "start_0s", id="zero"),
        param(600 * NANOS_PER_SECOND, "start_600s", id="integral"),
        param(1_500_000_000, "start_1.5s", id="fractional"),
        param(1, "start_0.000000001s", id="one-nanosecond"),
    ],
)  # fmt: skip
def test_window_label_formats_offset(offset_ns: int, expected: str) -> None:
    assert window_label(offset_ns) == expected


def test_window_label_is_unique_per_nanosecond() -> None:
    offsets = [0, 1, 999, 1_000, NANOS_PER_SECOND - 1, NANOS_PER_SECOND]
    assert len({window_label(offset) for offset in offsets}) == len(offsets)


@pytest.mark.asyncio
async def test_iter_window_profile_results_covers_start_grid_to_phase_end() -> None:
    metrics = _FakeMetricsAccumulator()
    server = _FakeServerMetricsAccumulator()
    telemetry = _FailingTelemetryAccumulator()

    windows = await _collect(
        metrics_accumulator=metrics,
        server_metrics_accumulator=server,
        gpu_telemetry_accumulator=telemetry,
    )

    assert [w.phase_name for w in windows] == ["start_0s", "start_60s", "start_120s"]
    assert [w.start_offset_sec for w in windows] == [0, 60, 120]
    expected_starts = [_START_NS + k * 60 * NANOS_PER_SECOND for k in range(3)]
    assert [w.start_ns for w in windows] == expected_starts
    assert {w.end_ns for w in windows} == {_END_NS}
    assert [ctx.min_request_start_ns for ctx in metrics.contexts] == expected_starts
    assert {ctx.phase for ctx in metrics.contexts} == {CreditPhase.PROFILING}
    assert [ctx.start_ns for ctx in server.contexts] == expected_starts
    assert {ctx.end_ns for ctx in telemetry.contexts} == {_END_NS + _GRACE_NS}
    assert [w.successful_request_count for w in windows] == [10, 20, 30]
    assert [w.error_request_count for w in windows] == [1, 2, 3]
    assert [w.server_metrics_results.start_ns for w in windows] == expected_starts
    assert all(w.telemetry_results is None for w in windows)
    assert all("dcgm gone" in w.telemetry_warnings[0] for w in windows)


@pytest.mark.asyncio
async def test_iter_window_profile_results_stops_at_first_empty_window() -> None:
    server = _FakeServerMetricsAccumulator()
    windows = await _collect(
        metrics_accumulator=_FakeMetricsAccumulator(empty_from=1),
        server_metrics_accumulator=server,
    )
    assert [w.phase_name for w in windows] == ["start_0s"]
    assert len(server.contexts) == 1


@pytest.mark.asyncio
async def test_iter_window_profile_results_honors_max_windows() -> None:
    metrics = _FakeMetricsAccumulator()
    windows = await _collect(metrics_accumulator=metrics, interval_sec=1, max_windows=4)
    assert [w.phase_name for w in windows] == [
        "start_0s",
        "start_1s",
        "start_2s",
        "start_3s",
    ]
    assert len(metrics.contexts) == 4


@pytest.mark.parametrize(
    "start_ns,end_ns",
    [
        param(None, _END_NS, id="no-start"),
        param(_START_NS, None, id="no-end"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_iter_window_profile_results_without_phase_bounds_yields_nothing(
    start_ns: int | None, end_ns: int | None
) -> None:
    metrics = _FakeMetricsAccumulator()
    assert (
        await _collect(metrics_accumulator=metrics, stats=_stats(start_ns, end_ns))
        == []
    )
    assert metrics.contexts == []
