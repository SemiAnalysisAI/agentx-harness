# SPDX-License-Identifier: Apache-2.0
"""Explicit runnable and blocked work for a cooperative timing driver."""

import asyncio
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(eq=False, slots=True)
class _Activity:
    task: asyncio.Task | None = None
    """Owner, assigned when a synchronously reserved task starts."""
    ready: Callable[[], bool] | None = None
    """None for runnable work; a blocked activity becomes runnable when true."""


class TimingActivity:
    """Track only declared timing work, never inspect asyncio task internals.

    A blocked scope supplies an observable readiness predicate. Once its event
    is set it counts as runnable, including before the coroutine resumes.
    """

    def __init__(self) -> None:
        self._changed = asyncio.Event()
        self._entries: set[_Activity] = set()
        self._current: ContextVar[_Activity | None] = ContextVar(
            "timing_activity", default=None
        )
        self._failure: BaseException | None = None

    def _owned_entry(self) -> _Activity | None:
        entry = self._current.get()
        return (
            entry
            if entry is not None and entry.task is asyncio.current_task()
            else None
        )

    @property
    def idle(self) -> bool:
        return all(
            entry.ready is not None
            and not entry.ready()
            and entry.task is not None
            and not entry.task.done()
            and not entry.task.cancelling()
            for entry in self._entries
        )

    def check_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError("Tracked timing work failed") from self._failure

    def assert_drained(self) -> None:
        self.check_failure()
        if self._entries:
            raise RuntimeError(f"Unfinished timing activities: {len(self._entries)}")

    @contextmanager
    def scope(self) -> Iterator[None]:
        existing = self._owned_entry()
        entry = existing or _Activity(task=asyncio.current_task())
        token = self._current.set(entry)
        self._entries.add(entry)
        self._changed.set()
        try:
            yield
        except Exception as exc:
            self._failure = exc
            raise
        finally:
            self._current.reset(token)
            if existing is None:
                self._entries.discard(entry)
            self._changed.set()

    @contextmanager
    def blocked(self, ready: Callable[[], bool]) -> Iterator[None]:
        entry = self._owned_entry()
        if entry is None:
            raise RuntimeError(
                "A semantic timing wait requires a tracked activity scope"
            )
        previous = entry.ready
        entry.ready = ready
        self._changed.set()
        try:
            yield
        finally:
            entry.ready = previous
            self._changed.set()

    def create_task(
        self,
        loop: asyncio.AbstractEventLoop,
        coro: Coroutine[Any, Any, T],
        *,
        on_done: Callable[[asyncio.Task[T]], None] | None = None,
    ) -> asyncio.Task[T]:
        # Reserve before task creation, including when an eager factory is used.
        entry = _Activity()
        self._entries.add(entry)
        self._changed.set()
        began = False

        async def run() -> T:
            nonlocal began
            began = True
            entry.task = asyncio.current_task()
            token = self._current.set(entry)
            try:
                return await coro
            finally:
                self._current.reset(token)

        def done(task: asyncio.Task[T]) -> None:
            try:
                if not began:
                    coro.close()
                if not task.cancelled() and task.exception() is not None:
                    self._failure = task.exception()
                if on_done is not None:
                    on_done(task)
            except Exception as exc:
                self._failure = exc
            finally:
                self._entries.discard(entry)
                self._changed.set()

        wrapper = run()
        try:
            task = loop.create_task(wrapper)
        except BaseException:
            wrapper.close()
            coro.close()
            self._entries.discard(entry)
            self._changed.set()
            raise
        task.add_done_callback(done)
        return task

    async def wait_idle(self) -> None:
        while True:
            self.check_failure()
            self._changed.clear()
            if self.idle:
                return
            await self._changed.wait()
