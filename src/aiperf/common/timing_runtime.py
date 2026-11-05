# SPDX-License-Identifier: Apache-2.0
"""Injectable clocks and callback timers for workload timing."""

import asyncio
import time
from collections.abc import Callable
from typing import Any, Protocol


class CallbackHandle(Protocol):
    """Cancellation shared by immediate callbacks and timers."""

    def cancel(self) -> None: ...


class TimerHandle(CallbackHandle, Protocol):
    """A deadline in the runtime's timer clock domain."""

    def when(self) -> float: ...


class TimingRuntime(Protocol):
    """Clock reads and timers; coroutine execution stays on the asyncio loop.

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
    ) -> asyncio.TimerHandle:
        return self._event_loop().call_later(delay, callback, *args)

    def call_at(
        self, when: float, callback: Callable[..., Any], *args: Any
    ) -> asyncio.TimerHandle:
        return self._event_loop().call_at(when, callback, *args)

    def call_soon(self, callback: Callable[..., Any], *args: Any) -> asyncio.Handle:
        return self._event_loop().call_soon(callback, *args)
