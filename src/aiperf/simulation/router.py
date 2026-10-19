# SPDX-License-Identifier: Apache-2.0
"""Pair native worker returns with simulator terminals before advancing AgentX."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import aiofiles
import msgspec
import orjson

from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit
from aiperf.simulation.identity import request_id
from aiperf.simulation.protocol import (
    Cancel,
    CoordinatorClient,
    Grant,
    Ready,
    SimulationError,
    Submit,
)

if TYPE_CHECKING:
    from aiperf.common.virtual_timing_runtime import VirtualTimingRuntime
    from aiperf.credit.sticky_router import StickyCreditRouter

T = TypeVar("T")
ReturnCallback = Callable[[str, CreditReturn], Awaitable[None]]


@dataclass(slots=True)
class _PendingCredit:
    """A submitted credit awaiting both its HTTP result and virtual terminal."""

    response: asyncio.Future[tuple[str, CreditReturn]]
    """Resolved only after the worker has closed HTTP and committed session state."""
    cancelled: bool = False
    """Whether the driver has registered cancellation for this request."""


class SimulatorCreditRouter:
    """Wrap the existing router without changing its worker message protocol.

    Worker receive callbacks only store results. A single driver releases the
    original callbacks in Grant order, so a slow worker cannot reorder a round.
    """

    def __init__(
        self,
        router: StickyCreditRouter,
        *,
        benchmark_id: str,
        endpoint: str,
        runtime: VirtualTimingRuntime,
        artifact_dir: Path | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._router = router
        self._benchmark_id = benchmark_id
        self._client = CoordinatorClient(endpoint)
        self.runtime = runtime
        self._artifact_dir = artifact_dir
        self._timeout_s = timeout_s
        self._pending: dict[str, _PendingCredit] = {}
        self._cancelling: set[str] = set()
        self._cancellations_drained = asyncio.Event()
        self._cancellations_drained.set()
        self._issued: set[str] = set()
        self._operations: list[Submit | Cancel] = []
        self._return_callback: ReturnCallback | None = None
        self._failure: BaseException | None = None
        self._failed = asyncio.Event()
        self._trace: Any = None
        router.set_return_callback(self._receive_return)
        router.set_first_token_callback(self._receive_first_token)

    def _fail(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
            self._failed.set()

    async def send_credit(self, credit: Credit) -> None:
        """Register dispatch before any worker can start its HTTP request."""
        if credit.cancel_after_ns is not None:
            raise SimulationError("simulator does not support randomized cancellation")
        rid = request_id(self._benchmark_id, credit.phase_index, credit.id)
        if rid in self._issued:
            raise SimulationError(f"duplicate simulated credit: {rid}")
        self._issued.add(rid)
        self._pending[rid] = _PendingCredit(asyncio.get_running_loop().create_future())
        self._operations.append(Submit(rid))
        try:
            await self._router.send_credit(credit)
        except BaseException as exc:
            self._fail(
                SimulationError(f"credit dispatch interrupted for {rid}: {exc!r}")
            )
            raise

    async def wait_for_workers(self, timeout: float) -> None:
        """Preserve the native wall-clock startup gate."""
        await self._router.wait_for_workers(timeout)

    async def cancel_all_credits(self) -> None:
        """Declare cancellations before native workers close their HTTP requests."""
        for rid, pending in self._pending.items():
            if not pending.cancelled:
                pending.cancelled = True
                self._cancelling.add(rid)
                self._cancellations_drained.clear()
                self._operations.append(Cancel(rid))
        await self._router.cancel_all_credits()

    async def wait_for_cancellation_drain(self, timeout: float) -> None:
        """Wait for cancellation terminals, worker returns, and native callbacks."""
        await self.runtime.wait_wall_event(self._cancellations_drained, timeout)

    def mark_credits_complete(self) -> None:
        """Keep the native router's session cleanup accounting."""
        self._router.mark_credits_complete()

    @property
    def sticky_router(self) -> StickyCreditRouter:
        """Let native branch orchestration retain routing across parent completion."""
        return self._router

    def set_return_callback(self, callback: ReturnCallback) -> None:
        """Register the original AgentX credit callback."""
        self._return_callback = callback

    def set_first_token_callback(
        self, callback: Callable[[FirstToken], Awaitable[None]]
    ) -> None:
        """First-token scheduling is rejected by configuration validation."""

    async def _receive_first_token(self, message: FirstToken) -> None:
        # Transport token arrival is wall time and cannot release virtual credits.
        pass

    async def _receive_return(self, worker_id: str, message: CreditReturn) -> None:
        rid = request_id(
            self._benchmark_id, message.credit.phase_index, message.credit.id
        )
        pending = self._pending.get(rid)
        if pending is None or pending.response.done():
            self._fail(SimulationError(f"unexpected or duplicate CreditReturn: {rid}"))
            return
        if message.error is not None and not message.cancelled:
            self._fail(SimulationError(f"worker failed rid={rid}: {message.error}"))
        pending.response.set_result((worker_id, message))

    async def _guard(self, work: Coroutine[Any, Any, T]) -> T:
        """Bound physical work and surface worker errors while control is in flight."""
        task = asyncio.create_task(work)
        failed = asyncio.create_task(self._failed.wait())
        try:
            done, _ = await asyncio.wait(
                (task, failed),
                timeout=self._timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._failure is not None:
                raise self._failure
            if task not in done:
                raise SimulationError(
                    f"simulator driver did not settle within {self._timeout_s}s; "
                    f"pending={tuple(self._pending)}"
                )
            return task.result()
        finally:
            for item in (task, failed):
                if not item.done():
                    item.cancel()
            await asyncio.gather(task, failed, return_exceptions=True)

    async def _wait_return(self, rid: str) -> tuple[str, CreditReturn]:
        return await self._pending[rid].response

    async def _apply_return(self, worker_id: str, returned: CreditReturn) -> None:
        if self._return_callback is None:
            raise SimulationError("missing AgentX return callback")
        # FirstToken was withheld, so the terminal must release prefill accounting.
        if returned.first_token_sent:
            returned = msgspec.structs.replace(returned, first_token_sent=False)
        with self.runtime.activity_scope(label="callback"):
            await self._return_callback(worker_id, returned)

    async def _write_event(self, event: Ready | Grant) -> None:
        if self._trace is not None:
            await self._trace.write(
                orjson.dumps(
                    {
                        "event": type(event).__name__,
                        "now_ns": self.runtime.now_ns(),
                        **msgspec.to_builtins(event),
                    }
                )
                + b"\n"
            )

    async def _apply_grant(self, grant: Grant) -> None:
        """Apply every terminal before allowing workload timers to run."""
        self.runtime.advance_to(grant.now_ns)
        for terminal in grant.terminals:
            if terminal.rid not in self._pending:
                raise SimulationError(f"unexpected terminal: {terminal.rid}")
            worker_id, returned = await self._guard(self._wait_return(terminal.rid))
            cancelled = self._pending[terminal.rid].cancelled
            if terminal.kind == "ERROR" or (
                (terminal.kind == "CANCELLED") != returned.cancelled
                or returned.cancelled != cancelled
            ):
                raise SimulationError(
                    f"HTTP/Grant terminal mismatch for {terminal.rid}"
                )
            del self._pending[terminal.rid]
            await self._guard(self._apply_return(worker_id, returned))
            if cancelled:
                self._cancelling.remove(terminal.rid)
                if not self._cancelling:
                    self._cancellations_drained.set()

    async def drive(self) -> None:
        """Run until all phases and native simulator cleanup have completed."""
        try:
            if self._artifact_dir is not None:
                await asyncio.to_thread(
                    self._artifact_dir.mkdir, parents=True, exist_ok=True
                )
                self._trace = await aiofiles.open(
                    self._artifact_dir / "simulation_events.jsonl", "wb"
                )
            await self._guard(self.runtime.started.wait())
            round_id = 0
            while True:
                await self._guard(self.runtime.drain_current())
                finished = self.runtime.finished.is_set() and not self._pending
                if finished:
                    self.runtime.assert_drained()
                operations, self._operations = self._operations, []
                ready = Ready(
                    self._benchmark_id,
                    round_id,
                    operations,
                    next_wakeup_ns=None if finished else self.runtime.next_deadline_ns,
                    finished=finished,
                )
                if not finished and not self._pending and ready.next_wakeup_ns is None:
                    raise SimulationError("AgentX stalled without a request or timer")
                await self._write_event(ready)
                grant = await self._guard(self._client.request(ready))
                await self._write_event(grant)
                await self._apply_grant(grant)
                if grant.status == "FINISHED":
                    if not finished or self._pending:
                        raise SimulationError(
                            "simulator finished before AgentX drained"
                        )
                    return
                round_id += 1
        finally:
            self._client.close()
            if self._trace is not None:
                await self._trace.close()
                self._trace = None

    def close(self) -> None:
        """Release the control connection during service teardown."""
        self._client.close()
