# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
import time
from typing import TYPE_CHECKING

import orjson

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.constants import BYTES_PER_MIB, NANOS_PER_SECOND
from aiperf.common.enums import CommandType, MessageType, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_message, on_start
from aiperf.common.messages import WorkerHealthMessage
from aiperf.plugin.enums import ServiceType
from aiperf.workers.worker_group_state import (
    WorkerStatusInfo,
    build_worker_status_summary,
    mark_stale_workers,
    update_worker_status,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


MEMORY_LOG_INTERVAL_S = 300.0
"""Seconds between worker memory summary log lines."""


class WorkerManager(BaseComponentService):
    """
    The WorkerManager service is primary responsibility to manage the worker processes.
    It will spawn the workers, monitor their health, and stop them when the service is stopped.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        self.trace("WorkerManager.__init__")
        self.worker_infos: dict[str, WorkerStatusInfo] = {}
        self._last_memory_log_ns = time.time_ns()

        self.cpu_count = multiprocessing.cpu_count()
        self.debug(lambda: f"Detected {self.cpu_count} CPU cores/threads")

        self.max_concurrency = self._max_concurrency_from_run()
        runtime = self.run.cfg.runtime
        self.max_workers = runtime.workers
        if self.max_workers is None:
            # Default to 75% of the CPU cores - 1, with a cap of Environment.WORKER.MAX_WORKERS_CAP, and a minimum of 1
            self.max_workers = max(
                1,
                min(
                    int(self.cpu_count * Environment.WORKER.CPU_UTILIZATION_FACTOR) - 1,
                    Environment.WORKER.MAX_WORKERS_CAP,
                ),
            )
            self.debug(
                lambda: f"Auto-setting max workers to {self.max_workers} due to no max workers specified."
            )

        # Cap the worker count to the max concurrency, but only if the user is in concurrency mode.
        if self.max_concurrency and self.max_concurrency < self.max_workers:
            self.max_workers = self.max_concurrency
            self.debug(
                lambda: f"Capping max workers to {self.max_workers} due to concurrency."
            )

        # Ensure we have at least the min workers
        workers_min = runtime.workers_min
        self.max_workers = max(
            self.max_workers,
            workers_min or 1,
        )
        self.initial_workers = self.max_workers

    def _max_concurrency_from_run(self) -> int | None:
        """Return the maximum profiling-phase concurrency from the run.

        Worker capacity is bounded by concurrency: there is no point spawning
        more workers than there are in-flight credit slots. Each profiling
        phase declares its own concurrency on the BenchmarkConfig
        (``run.cfg.phases[i].concurrency``), so take the max across them.
        Returns ``None`` for non-concurrency phases (request-rate, fixed
        schedule, etc.) so the workers/CPU cap below applies unchanged.
        """
        concurrencies = [
            phase.concurrency
            for phase in self.run.cfg.get_profiling_phases()
            if getattr(phase, "concurrency", None) is not None
        ]
        return max(concurrencies) if concurrencies else None

    @on_start
    async def _start(self) -> None:
        """Start worker manager-specific components."""
        self.debug("WorkerManager starting")

        await self.send_command_to_controller(
            CommandType.SPAWN_WORKERS,
            payload=orjson.dumps({"num_workers": self.initial_workers}),
        )
        self.debug("WorkerManager started")

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        worker_id = message.service_id
        info = self.worker_infos.get(worker_id)
        if not info:
            info = WorkerStatusInfo(
                worker_id=worker_id,
                last_update_ns=time.time_ns(),
                status=WorkerStatus.HEALTHY,
                health=message.health,
                task_stats=message.task_stats,
            )
            self.worker_infos[worker_id] = info
        self._update_worker_status(info, message)

    def _update_worker_status(
        self, info: WorkerStatusInfo, message: WorkerHealthMessage
    ) -> None:
        """Check the status of a worker."""
        update_worker_status(info, message, warning=self.warning)

    @background_task(immediate=False, interval=Environment.WORKER.CHECK_INTERVAL)
    async def _worker_status_loop(self) -> None:
        """Check the status of all workers."""
        self.debug("Checking worker status")
        mark_stale_workers(self.worker_infos)
        self._log_memory_summary()

    def _log_memory_summary(self) -> None:
        """Log worker RSS (and anonymous RSS on Linux) every few minutes so a
        long run's memory trend is in aiperf.log without a profiler."""
        now_ns = time.time_ns()
        if (
            now_ns - self._last_memory_log_ns
        ) / NANOS_PER_SECOND < MEMORY_LOG_INTERVAL_S:
            return
        self._last_memory_log_ns = now_ns
        healths = [info.health for info in self.worker_infos.values()]
        if not healths:
            return
        rss = sorted(h.memory_usage for h in healths)
        anon = sorted(h.memory_anon for h in healths if h.memory_anon is not None)
        mib = 1 / BYTES_PER_MIB
        line = (
            f"Worker memory ({len(rss)} workers): rss min/median/max "
            f"{rss[0] * mib:.0f}/{rss[len(rss) // 2] * mib:.0f}/{rss[-1] * mib:.0f} MiB"
        )
        if anon:
            line += (
                f", anon min/median/max "
                f"{anon[0] * mib:.0f}/{anon[len(anon) // 2] * mib:.0f}/{anon[-1] * mib:.0f} MiB"
            )
        self.info(line)

    @background_task(
        immediate=False, interval=Environment.WORKER.STATUS_SUMMARY_INTERVAL
    )
    async def _worker_summary_loop(self) -> None:
        """Generate a summary of the worker status."""
        await self.publish(
            build_worker_status_summary(
                service_id=self.service_id, worker_infos=self.worker_infos
            )
        )


def main() -> None:
    """Main entry point for the worker manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(ServiceType.WORKER_MANAGER)


if __name__ == "__main__":
    main()
