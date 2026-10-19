# SPDX-License-Identifier: Apache-2.0
"""Keep result finalization behind the simulator's FINISHED acknowledgment."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    ) -> None:
        super().__init__(pub_client=pub_client, service_id=service_id)
        self._final_phase = (final_phase_config.phase, final_phase_config.phase_index)
        self._completion: tuple[CreditPhaseStats, BranchStats | None] | None = None
        self._credits_complete = False

    async def publish_phase_complete(
        self,
        phase_stats: CreditPhaseStats,
        branch_stats: BranchStats | None = None,
    ) -> None:
        """Hold only the final configured phase, identified by kind and index."""
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
        await super().publish_phase_complete(*completion)
        await super().publish_credits_complete()
