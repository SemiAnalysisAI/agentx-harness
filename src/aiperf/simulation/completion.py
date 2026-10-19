# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep result finalization behind the simulator's FINISHED acknowledgment."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from aiperf.simulation.identity import VIRTUAL_TIMESTAMP_ORIGIN_NS, request_id
from aiperf.simulation.protocol import SimulationError
from aiperf.timing.phase.publisher import PhasePublisher

if TYPE_CHECKING:
    from aiperf.common.models import CreditPhaseStats
    from aiperf.common.models.branch_stats import BranchStats
    from aiperf.common.protocols import PubClientProtocol
    from aiperf.timing.config import CreditPhaseConfig


class SimulationPhasePublisher(PhasePublisher):
    """Publish ordinary phase events immediately; hold final success messages.

    The timing manager calls ``flush`` only after both AgentX phases and the
    simulator driver have succeeded. Waiting inside a phase would prevent the
    driver from observing that workload activity has finished.
    """

    def __init__(
        self,
        *,
        pub_client: PubClientProtocol,
        service_id: str,
        final_phase_config: CreditPhaseConfig,
        artifact_dir: Path | None = None,
        benchmark_id: str | None = None,
    ) -> None:
        super().__init__(pub_client=pub_client, service_id=service_id)
        self._final_phase = (final_phase_config.phase, final_phase_config.phase_index)
        self._completion: tuple[CreditPhaseStats, BranchStats | None] | None = None
        self._credits_complete = False
        self._artifact_dir = artifact_dir
        self._benchmark_id = benchmark_id
        self._phase_windows: dict[tuple[str, int | None], dict] = {}
        self._protocol_finished = False

    async def publish_phase_start(
        self, config: CreditPhaseConfig, phase_stats: CreditPhaseStats
    ) -> None:
        """Capture exact virtual boundaries independently of result timestamps."""
        if self._artifact_dir is not None:
            duration = config.expected_duration_sec
            grace = config.grace_period_sec
            key = (phase_stats.phase, phase_stats.phase_index)
            self._phase_windows[key] = {
                "phase": phase_stats.phase,
                "phase_index": phase_stats.phase_index,
                "phase_name": config.phase_name,
                "rid_prefix": (
                    request_id(self._benchmark_id, phase_stats.phase_index, 0).rsplit(
                        ".", 1
                    )[0]
                    + "."
                    if self._benchmark_id is not None
                    else None
                ),
                "configured_duration_ns": (
                    round(duration * 1e9) if duration is not None else None
                ),
                "configured_grace_ns": (
                    round(grace * 1e9)
                    if grace is not None and math.isfinite(grace)
                    else None
                ),
                "grace_unbounded": grace is None or not math.isfinite(grace),
            }
            await self._record_window(phase_stats, complete=False)
        await super().publish_phase_start(config, phase_stats)

    async def publish_phase_sending_complete(
        self, phase_stats: CreditPhaseStats
    ) -> None:
        """Preserve the issuance stop time, which may precede the duration limit."""
        await self._record_window(phase_stats, complete=False)
        await super().publish_phase_sending_complete(phase_stats)

    async def _record_window(self, stats: CreditPhaseStats, *, complete: bool) -> None:
        if self._artifact_dir is None:
            return
        key = (stats.phase, stats.phase_index)
        if key not in self._phase_windows:
            raise SimulationError(f"simulator phase boundary has no start: {key}")
        window = self._phase_windows[key]
        for name in ("start_ns", "sent_end_ns", "requests_end_ns"):
            timestamp = getattr(stats, name)
            window[name] = (
                timestamp - VIRTUAL_TIMESTAMP_ORIGIN_NS
                if timestamp is not None
                else None
            )
        start = window["start_ns"]
        duration = window["configured_duration_ns"]
        end = start + duration if start is not None and duration is not None else None
        window["measurement_end_ns"] = end
        grace = window["configured_grace_ns"]
        window["grace_limit_ns"] = (
            end + grace if end is not None and grace is not None else None
        )
        window["complete"] = complete
        window["was_cancelled"] = stats.was_cancelled
        window["timeout_triggered"] = stats.timeout_triggered
        window["grace_period_timeout_triggered"] = stats.grace_period_timeout_triggered
        window["incomplete_measurement"] = (
            end is not None
            and (
                stats.was_cancelled
                or window["sent_end_ns"] is None
                or window["sent_end_ns"] < end
                or window["requests_end_ns"] is None
                or window["requests_end_ns"] < end
            )
            if complete
            else None
        )
        await self._save_windows()

    async def _save_windows(self) -> None:
        if self._artifact_dir is None:
            return
        payload = orjson.dumps(
            {
                "version": 1,
                "clock": "virtual_ns",
                "timestamp_origin_ns": VIRTUAL_TIMESTAMP_ORIGIN_NS,
                "benchmark_id": self._benchmark_id,
                "protocol_finished": self._protocol_finished,
                "phases": list(self._phase_windows.values()),
            },
            option=orjson.OPT_INDENT_2,
        )
        await asyncio.to_thread(self._replace_windows, payload)

    def _replace_windows(self, payload: bytes) -> None:
        """Replace a small snapshot atomically from the asynchronous I/O worker."""
        assert self._artifact_dir is not None
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self._artifact_dir / "phase_windows.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(payload + b"\n")
        temporary.replace(path)

    async def publish_phase_complete(
        self,
        phase_stats: CreditPhaseStats,
        branch_stats: BranchStats | None = None,
    ) -> None:
        """Hold only the final configured phase, identified by kind and index."""
        await self._record_window(phase_stats, complete=True)
        if (phase_stats.phase, phase_stats.phase_index) == self._final_phase:
            if self._completion is not None:
                raise SimulationError(
                    f"duplicate final phase completion: {self._final_phase}"
                )
            self._completion = (phase_stats, branch_stats)
        else:
            await super().publish_phase_complete(phase_stats, branch_stats)

    async def publish_credits_complete(self) -> None:
        """Record phase orchestration completion without starting result export."""
        self._credits_complete = True

    async def flush(self) -> None:
        """Publish final success once, after the driver's FINISHED acknowledgment."""
        if self._completion is None or not self._credits_complete:
            raise SimulationError(
                "simulator final completion is missing or already consumed"
            )
        completion, self._completion = self._completion, None
        self._credits_complete = False
        # Consume before I/O: an ambiguous partial publication must not be retried.
        self._protocol_finished = True
        await self._save_windows()
        await super().publish_phase_complete(*completion)
        await super().publish_credits_complete()
