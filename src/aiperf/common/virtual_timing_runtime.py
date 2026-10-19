# SPDX-License-Identifier: Apache-2.0
"""An externally advanced clock; AgentX still executes its own workflows."""

import asyncio
import heapq
import math
from collections.abc import Callable, Coroutine
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from aiperf.common.timing_activity import TimingActivity
from aiperf.common.timing_runtime import RealTimingRuntime

T = TypeVar("T")
_NANOSECONDS = 1_000_000_000


@dataclass(slots=True)
class _Callback:
    due_ns: int
    """Deadline in virtual nanoseconds."""
    callback: Callable[..., Any]
    """Synchronous callback released by the driver."""
    args: tuple[Any, ...]
    """Positional callback arguments."""
    cancelled: bool = False
    """Cancelled handles are ignored when they reach the timer heap's head."""

    def cancel(self) -> None:
        self.cancelled = True


class _Timer(_Callback):
    def when(self) -> float:
        return self.due_ns / _NANOSECONDS


class VirtualTimingRuntime(RealTimingRuntime):
    """Move time, release results, then drain deadlines and runnable work.

    ``advance_to`` never fires callbacks. The owner first applies completed
    requests at that timestamp, then calls ``drain_current``. Phase deadlines
    precede ordinary timers at the same timestamp.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        *,
        timestamp_origin_ns: int = 0,
    ) -> None:
        super().__init__(loop)
        self.activity = TimingActivity()
        self.started = self.activity.started
        self.finished = self.activity.finished
        self._now_ns = 0
        self._timestamp_origin_ns = timestamp_origin_ns
        self._sequence = 0
        self._timers: list[tuple[int, int, int, _Callback]] = []

    @property
    def is_virtual(self) -> bool:
        return True

    def now_ns(self) -> int:
        return self._now_ns

    def timestamp_ns(self) -> int:
        return self._timestamp_origin_ns + self._now_ns

    def timer_time(self) -> float:
        return self._now_ns / _NANOSECONDS

    def activity_scope(self, *, label: str = "work") -> AbstractContextManager[None]:
        return self.activity.scope(label=label)

    def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        on_done: Callable[[asyncio.Task[T]], None] | None = None,
    ) -> asyncio.Task[T]:
        return self.activity.create_task(self._event_loop(), coro, on_done=on_done)

    def _enqueue(
        self,
        when: float,
        callback: Callable[..., Any],
        args: tuple[Any, ...],
        *,
        priority: int = 1,
        immediate: bool = False,
    ) -> _Callback:
        if not math.isfinite(when):
            raise ValueError("Virtual timer deadline must be finite")
        due = max(self._now_ns, round(when * _NANOSECONDS))
        handle = (_Callback if immediate else _Timer)(due, callback, args)
        heapq.heappush(self._timers, (due, priority, self._sequence, handle))
        self._sequence += 1
        return handle

    def call_at(self, when: float, callback: Callable[..., Any], *args: Any) -> _Timer:
        return cast(_Timer, self._enqueue(when, callback, args))

    def call_later(
        self, delay: float, callback: Callable[..., Any], *args: Any
    ) -> _Timer:
        if not math.isfinite(delay):
            raise ValueError("Virtual timer delay must be finite")
        return self.call_at(self.timer_time() + max(0.0, delay), callback, *args)

    def call_soon(self, callback: Callable[..., Any], *args: Any) -> _Callback:
        return self._enqueue(self.timer_time(), callback, args, immediate=True)

    @property
    def next_deadline_ns(self) -> int | None:
        while self._timers and self._timers[0][3].cancelled:
            heapq.heappop(self._timers)
        return self._timers[0][0] if self._timers else None

    def advance_to(self, now_ns: int) -> None:
        if type(now_ns) is not int or now_ns < self._now_ns:
            raise ValueError(
                "Virtual time must be a monotonic integer number of nanoseconds"
            )
        deadline = self.next_deadline_ns
        if deadline is not None and now_ns > deadline:
            raise ValueError(f"Cannot advance past pending deadline {deadline}")
        self._now_ns = now_ns

    def assert_drained(self) -> None:
        """Do not let phase completion hide blocked work or future timers."""
        self.activity.assert_drained()
        if self.next_deadline_ns is not None:
            raise RuntimeError(f"Unfinished timing deadline: {self.next_deadline_ns}")

    async def drain_current(self) -> None:
        count = 0
        while True:
            await self.activity.wait_idle()
            deadline = self.next_deadline_ns
            if deadline is None or deadline > self._now_ns:
                return
            _, _, _, handle = heapq.heappop(self._timers)
            with self.activity_scope():
                handle.callback(*handle.args)
            count += 1
            if count > 10000:
                raise RuntimeError(
                    "Timing callbacks do not settle at the current timestamp"
                )

    async def wait_event(
        self, event: asyncio.Event, timeout: float | None = None
    ) -> None:
        await self._wait_event(event, timeout, wall_clock=False)

    async def wait_wall_event(self, event: asyncio.Event, timeout: float) -> None:
        await self._wait_event(event, timeout, wall_clock=True)

    async def _wait_event(
        self, event: asyncio.Event, timeout: float | None, *, wall_clock: bool
    ) -> None:
        if timeout is not None:
            if math.isnan(timeout):
                raise ValueError("Event timeout cannot be NaN")
            if timeout == math.inf:
                timeout = None
        if event.is_set():
            return
        if timeout is not None and timeout <= 0:
            raise TimeoutError
        loop = self._event_loop()
        event_task = loop.create_task(event.wait())
        expired = loop.create_future()
        handle = None
        if timeout is not None:
            if wall_clock:
                handle = loop.call_later(timeout, expired.set_result, None)
            else:
                handle = self._enqueue(
                    self.timer_time() + timeout, expired.set_result, (None,), priority=0
                )
        try:
            with self.activity.blocked(lambda: event.is_set() or expired.done()):
                await asyncio.wait(
                    (event_task, expired), return_when=asyncio.FIRST_COMPLETED
                )
            if not event.is_set():
                raise TimeoutError
        finally:
            if handle is not None:
                handle.cancel()
            event_task.cancel()
            expired.cancel()
            await asyncio.gather(event_task, return_exceptions=True)

    async def gather(
        self, *coros: Coroutine[Any, Any, T], return_exceptions: bool = False
    ) -> list[T | BaseException]:
        tasks = [self.create_task(coro) for coro in coros]
        result = asyncio.gather(*tasks, return_exceptions=return_exceptions)
        with self.activity.blocked(result.done):
            return await result

    async def sleep(self, delay: float) -> None:
        event = asyncio.Event()
        handle = self.call_later(delay, event.set)
        try:
            await self.wait_event(event)
        finally:
            handle.cancel()
