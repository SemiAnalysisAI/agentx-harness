# SPDX-License-Identifier: Apache-2.0
"""Injectable clocks and callback timers for workload timing."""

import asyncio
import time
from collections.abc import Callable, Coroutine
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class CallbackHandle(Protocol):
    """Cancellation shared by immediate callbacks and timers."""

    def cancel(self) -> None: ...


class TimerHandle(CallbackHandle, Protocol):
    """A deadline in the runtime's timer clock domain."""

    def when(self) -> float: ...


class TimingRuntime(Protocol):
    """Clocks and semantic waits; coroutine execution stays on the asyncio loop.

    ``now_ns`` is monotonic performance time. ``timestamp_ns`` is the timestamp
    published with phase events. Timer deadlines use ``timer_time`` seconds;
    they may have a different origin from performance time. Scheduling must
    enqueue callbacks, never invoke them inline.
    """

    def now_ns(self) -> int: ...

    def timestamp_ns(self) -> int: ...

    def timer_time(self) -> float: ...

    def call_later(
        self, delay: float, callback: Callable[..., Any], *args: Any
    ) -> TimerHandle: ...

    def call_at(
        self, when: float, callback: Callable[..., Any], *args: Any
    ) -> TimerHandle: ...

    def call_soon(self, callback: Callable[..., Any], *args: Any) -> CallbackHandle: ...

    def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        on_done: Callable[[asyncio.Task[T]], None] | None = None,
    ) -> asyncio.Task[T]: ...

    async def wait_event(
        self, event: asyncio.Event, timeout: float | None = None
    ) -> None: ...

    async def wait_wall_event(self, event: asyncio.Event, timeout: float) -> None: ...

    async def gather(
        self, *coros: Coroutine[Any, Any, T], return_exceptions: bool = False
    ) -> list[T | BaseException]: ...

    async def sleep(self, delay: float) -> None: ...

    def activity_scope(self) -> AbstractContextManager[None]: ...

    def run_scope(self) -> AbstractContextManager[None]:
        """Track a complete benchmark run across all phase boundaries."""
        ...

    @property
    def is_virtual(self) -> bool: ...


class RealTimingRuntime:
    """Default runtime using the existing Python clocks and asyncio timers."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop

    def _event_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop if self._loop is not None else asyncio.get_running_loop()

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def timestamp_ns(self) -> int:
        return time.time_ns()

    def timer_time(self) -> float:
        return self._event_loop().time()

    def call_later(
        self, delay: float, callback: Callable[..., Any], *args: Any
    ) -> TimerHandle:
        return self._event_loop().call_later(delay, callback, *args)

    def call_at(
        self, when: float, callback: Callable[..., Any], *args: Any
    ) -> TimerHandle:
        return self._event_loop().call_at(when, callback, *args)

    def call_soon(self, callback: Callable[..., Any], *args: Any) -> CallbackHandle:
        return self._event_loop().call_soon(callback, *args)

    @property
    def is_virtual(self) -> bool:
        return False

    def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        on_done: Callable[[asyncio.Task[T]], None] | None = None,
    ) -> asyncio.Task[T]:
        task = self._event_loop().create_task(coro)
        if on_done is not None:
            task.add_done_callback(on_done)
        return task

    async def wait_event(
        self, event: asyncio.Event, timeout: float | None = None
    ) -> None:
        if timeout is None:
            await event.wait()
        else:
            await asyncio.wait_for(event.wait(), timeout)

    async def sleep(self, delay: float) -> None:
        await asyncio.sleep(delay)

    async def wait_wall_event(self, event: asyncio.Event, timeout: float) -> None:
        await asyncio.wait_for(event.wait(), timeout)

    async def gather(
        self, *coros: Coroutine[Any, Any, T], return_exceptions: bool = False
    ) -> list[T | BaseException]:
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)

    def activity_scope(self) -> AbstractContextManager[None]:
        return nullcontext()

    def run_scope(self) -> AbstractContextManager[None]:
        return nullcontext()
