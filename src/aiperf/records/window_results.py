# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Suffix-window summaries of the profiling phase.

Window k spans ``[profiling start + k * interval, profiling end]``. Exporting
every window at run end lets the measured window be chosen after the run, so
the benchmark itself never changes behavior to exclude a startup transient.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.models import MetricResult, PhaseRecordsStats, WindowProfileResults
from aiperf.common.types import MetricTagT


def window_label(offset_ns: int) -> str:
    """Directory-safe window name, unique per nanosecond, e.g. ``start_600s``."""
    seconds, nanos = divmod(offset_ns, NANOS_PER_SECOND)
    if nanos == 0:
        return f"start_{seconds}s"
    return f"start_{seconds}.{nanos:09d}".rstrip("0") + "s"


def _metric_count(results: dict[MetricTagT, MetricResult], tag: str) -> int:
    result = results.get(tag)
    if result is None or result.avg is None:
        return 0
    return int(result.avg)


async def _window_telemetry_results(
    accumulator: Any | None, ctx: ExportContext, final_scrape_grace_ns: int
) -> tuple[Any | None, list[str]]:
    if accumulator is None:
        return None, []
    try:
        # Like the whole-run export, keep the trailing scrape that closes GPU
        # counter deltas.
        results = await accumulator.export_results(
            replace(ctx, end_ns=ctx.end_ns + final_scrape_grace_ns)
        )
    except Exception as e:  # noqa: BLE001 - window artifact remains best-effort
        return None, [f"GPU telemetry window export failed: {type(e).__name__}: {e}"]
    if results is None or not getattr(results, "endpoints", None):
        return None, []
    return results, []


async def _window_server_metrics_results(
    accumulator: Any | None, ctx: ExportContext
) -> tuple[Any | None, list[str]]:
    if accumulator is None:
        return None, []
    try:
        results = await accumulator.export_window_results(ctx)
    except Exception as e:  # noqa: BLE001 - window artifact remains best-effort
        return None, [f"Server metrics window export failed: {type(e).__name__}: {e}"]
    if results is None or not getattr(results, "endpoint_summaries", None):
        return None, []
    return results, []


async def iter_window_profile_results(
    *,
    metrics_accumulator: Any,
    server_metrics_accumulator: Any | None,
    gpu_telemetry_accumulator: Any | None,
    stats: PhaseRecordsStats,
    interval_sec: float,
    max_windows: int,
    final_scrape_grace_ns: int,
    should_stop: Callable[[], bool] = lambda: False,
) -> AsyncIterator[WindowProfileResults]:
    """Yield the profiling suffix windows on the start grid, earliest first.

    Request metrics keep the profiling phase-tag selection and add a
    request-start lower bound, so throughput divides by the window length.
    Server metrics and GPU telemetry are cut by time, with the last scrape
    before the window start as the counter delta reference. Error details are
    not time-indexed, so a window carries only its error count.

    The request cohort only shrinks as the start moves later, so the grid stops
    at the first window without requests, and in any case after
    ``max_windows`` windows, or as soon as ``should_stop()`` is true before a
    window is computed. Yields nothing when the profiling phase has no start or
    end time.
    """
    if stats.start_ns is None or stats.requests_end_ns is None:
        return
    interval_ns = max(1, round(interval_sec * NANOS_PER_SECOND))
    for k in range(max_windows):
        if should_stop():
            return
        offset_ns = k * interval_ns
        start_ns = stats.start_ns + offset_ns
        if start_ns >= stats.requests_end_ns:
            return
        ctx = ExportContext(
            start_ns=start_ns,
            end_ns=stats.requests_end_ns,
            phase=CreditPhase.PROFILING,
            min_request_start_ns=start_ns,
        )
        summary = await metrics_accumulator.export_window_results(ctx)
        successful = _metric_count(summary.results, "request_count")
        errors = _metric_count(summary.results, "error_request_count")
        if successful + errors == 0:
            return
        telemetry, telemetry_warnings = await _window_telemetry_results(
            gpu_telemetry_accumulator, ctx, final_scrape_grace_ns
        )
        server_metrics, server_metrics_warnings = await _window_server_metrics_results(
            server_metrics_accumulator, ctx
        )
        yield WindowProfileResults(
            start_offset_sec=offset_ns / NANOS_PER_SECOND,
            phase_name=window_label(offset_ns),
            phase_kind="profiling",
            records=list(summary.results.values()),
            start_ns=start_ns,
            end_ns=stats.requests_end_ns,
            was_cancelled=stats.was_cancelled,
            successful_request_count=successful,
            error_request_count=errors,
            telemetry_results=telemetry,
            server_metrics_results=server_metrics,
            telemetry_warnings=telemetry_warnings,
            server_metrics_warnings=server_metrics_warnings,
        )
