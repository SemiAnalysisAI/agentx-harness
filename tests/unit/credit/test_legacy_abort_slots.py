# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failfast must return issuer-owned legacy capacity without inventing HTTP returns."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.session_tree import SessionTreeRegistry

pytest_plugins = ["tests.unit.credit.test_session_abort_continuation"]


async def prepare_legacy(
    flow: SimpleNamespace,
    root_final: bool,
    *,
    with_orphan: bool = True,
    tree_registry: bool = False,
    expected_sessions: int | None = None,
) -> SimpleNamespace:
    """Acquire real capacity and count real root/child credits through the issuer."""
    f = flow
    f.router._track_credit_returned("worker", 1, False, False)
    cm = ConcurrencyManager()
    cm.configure_for_phase(CreditPhase.PROFILING, concurrency=1, prefill_concurrency=8)
    progress = PhaseProgressTracker(
        CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            concurrency=1,
            total_expected_requests=20,
            expected_num_sessions=expected_sessions,
        )
    )
    registry = SessionTreeRegistry(cm) if tree_registry else None
    f.issuer._session_tree_registry = registry
    f.branch._session_tree_registry = registry
    f.callback._session_tree_registry = registry
    f.issuer._concurrency_manager = cm
    f.issuer._progress = progress
    f.callback._concurrency_manager = cm
    kwargs = dict(
        phase=CreditPhase.PROFILING,
        progress=progress,
        lifecycle=f.issuer._lifecycle,
        stop_checker=f.stop,
        strategy=f.strategy,
    )
    if "credit_issuer" in inspect.signature(f.callback.register_phase).parameters:
        kwargs["credit_issuer"] = f.issuer
    f.callback.register_phase(**kwargs)

    def sent(*, credit: Credit) -> None:
        f.router._track_credit_sent("worker", credit.id, credit.x_correlation_id)

    f.router.send_credit = AsyncMock(side_effect=sent)
    await f.issuer.issue_credit(
        TurnToSend(
            conversation_id="root",
            x_correlation_id="parent",
            turn_index=0,
            num_turns=1 if root_final else 2,
        )
    )
    root = f.router.send_credit.await_args.kwargs["credit"]
    children = {}
    names = ["failed", "orphan"] if with_orphan else ["failed"]
    if not with_orphan:
        f.branch._child_to_join.pop("orphan")
        f.branch._child_modes.pop("orphan")
        f.branch._descendant_counts["parent"] = 1
    for name in names:
        await f.issuer.dispatch_child_turn(
            TurnToSend(
                conversation_id=name,
                x_correlation_id=name,
                turn_index=0,
                num_turns=1 if name == "failed" else 2,
                agent_depth=1,
                parent_correlation_id="parent",
                root_correlation_id="parent",
            )
        )
        children[name] = f.router.send_credit.await_args.kwargs["credit"]
    if registry is not None:
        registry.register_descendants("parent", len(children))
        f.branch._child_root = {name: "parent" for name in children}
    f.real_cm, f.real_progress, f.root, f.children = cm, progress, root, children
    assert not cm.session_slot_available(CreditPhase.PROFILING)
    assert progress.in_flight_sessions == 1
    return f


async def returned(
    flow: SimpleNamespace,
    credit: Credit,
    *,
    cancelled: bool = False,
    error: str | None = None,
) -> None:
    """Retire router ownership before invoking the real production callback."""
    await flow.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=credit, cancelled=cancelled, error=error, first_token_sent=False
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("active_root", [pytest.param(True,id="active-root"),pytest.param(False,id="gated-root")])  # fmt: skip
@pytest.mark.parametrize("cutoff", [pytest.param(False,id="phase-open"),pytest.param(True,id="sending-cutoff")])  # fmt: skip
@pytest.mark.parametrize("tree_registry", [pytest.param(False,id="legacy-slots"),pytest.param(True,id="tree-registry")])  # fmt: skip
async def test_legacy_abort_releases_one_slot_and_next_root_can_start(
    abort_flow: SimpleNamespace,
    active_root: bool,
    cutoff: bool,
    tree_registry: bool,
) -> None:
    """Both active and parked roots release after the last descendant, not teardown."""
    f = await prepare_legacy(abort_flow, root_final=False, tree_registry=tree_registry)
    if not active_root:
        await returned(f, f.root)
    if cutoff:
        f.real_progress.freeze_sent_counts()
    await returned(f, f.children["failed"], error="intentional child failure")
    assert not f.real_cm.session_slot_available(CreditPhase.PROFILING)
    if active_root:
        await returned(f, f.root, cancelled=True)
        assert not f.real_cm.session_slot_available(CreditPhase.PROFILING)
    await returned(f, f.children["orphan"], cancelled=True)
    assert f.real_cm.session_slot_available(CreditPhase.PROFILING)
    assert f.real_progress.in_flight_sessions == 0
    assert f.real_progress.counter.cancelled_sessions == 1
    assert f.real_progress.counter.requests_sent == 3
    assert (
        f.real_progress.counter.requests_completed
        + f.real_progress.counter.requests_cancelled
        == 3
    )
    assert f.real_cm.get_session_stats(CreditPhase.PROFILING).release_count == 1
    if not cutoff:
        await f.issuer.issue_credit(
            TurnToSend(
                conversation_id="other",
                x_correlation_id="other",
                turn_index=0,
                num_turns=1,
            )
        )
        other = f.router.send_credit.await_args.kwargs["credit"]
        assert other.x_correlation_id == "other"
        await returned(f, other)
        assert f.real_cm.get_session_stats(CreditPhase.PROFILING).release_count == 2
        assert f.real_progress.in_flight_sessions == 0


@pytest.mark.asyncio
async def test_already_completed_legacy_root_not_released_again_on_child_failure(
    abort_flow: SimpleNamespace,
) -> None:
    """A background child failure must not return another root's capacity."""
    f = await prepare_legacy(abort_flow, root_final=True)
    await returned(f, f.root)
    assert f.real_cm.get_session_stats(CreditPhase.PROFILING).release_count == 1
    await returned(f, f.children["failed"], error="intentional child failure")
    await returned(f, f.children["orphan"], cancelled=True)
    assert f.real_cm.get_session_stats(CreditPhase.PROFILING).release_count == 1
    assert f.real_progress.counter.completed_sessions == 1
    assert f.real_progress.counter.cancelled_sessions == 0
    assert f.real_progress.in_flight_sessions == 0


@pytest.mark.asyncio
async def test_sole_failing_child_last_wire_return_classifies_root_before_cutoff_cleanup(
    abort_flow: SimpleNamespace,
) -> None:
    """Frozen sending counts must not retire the root before failfast classification."""
    f = await prepare_legacy(abort_flow, root_final=False, with_orphan=False)
    await returned(f, f.root)
    f.real_progress.freeze_sent_counts()
    await returned(f, f.children["failed"], error="intentional child failure")
    assert f.real_progress.counter.requests_sent == 2
    assert f.real_progress.counter.requests_completed == 2
    assert f.real_progress.counter.requests_cancelled == 0
    assert f.real_progress.counter.cancelled_sessions == 1
    assert f.real_progress.in_flight_sessions == 0
    assert f.real_cm.get_session_stats(CreditPhase.PROFILING).release_count == 1
    assert f.real_cm.session_slot_available(CreditPhase.PROFILING)


@pytest.mark.asyncio
async def test_seamless_handoff_preserves_active_legacy_capacity_until_returns_finish(
    abort_flow: SimpleNamespace,
) -> None:
    """Sending completion cannot release ownership ahead of live old-phase requests."""
    f = await prepare_legacy(abort_flow, root_final=False)
    runner = object.__new__(PhaseRunner)
    runner._config = SimpleNamespace(seamless=True)
    runner._lifecycle = SimpleNamespace(is_complete=False)
    runner._credit_issuer = f.issuer
    runner._session_tree_registry = None
    runner._progress_task = None
    runner._on_phase_complete = None
    runner._release_tree_slots()
    assert not f.real_cm.session_slot_available(CreditPhase.PROFILING)
    f.stop.can_send_child_turn.return_value = False
    f.stop.can_send_any_turn.return_value = False
    f.real_progress.freeze_sent_counts()
    await returned(f, f.root)
    await returned(f, f.children["failed"])
    await returned(f, f.children["orphan"], cancelled=True)
    runner._lifecycle.is_complete = True
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    runner._on_return_wait_complete(task)
    runner._on_return_wait_complete(task)
    assert f.real_cm.session_slot_available(CreditPhase.PROFILING)
    assert f.real_cm.get_session_stats(CreditPhase.PROFILING).release_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tree_registry", [pytest.param(False,id="legacy-slots"),pytest.param(True,id="tree-registry")])  # fmt: skip
@pytest.mark.parametrize("seamless", [pytest.param(False,id="ordinary-teardown"),pytest.param(True,id="seamless-done-hook")])  # fmt: skip
async def test_force_drain_then_phase_cleanup_releases_each_slot_once(
    abort_flow: SimpleNamespace,
    tree_registry: bool,
    seamless: bool,
) -> None:
    """Generic forced drain must reconcile root ownership before later cleanup."""
    f = await prepare_legacy(abort_flow, root_final=False, tree_registry=tree_registry)
    runner = object.__new__(PhaseRunner)
    runner._config = SimpleNamespace(
        seamless=seamless, phase=CreditPhase.PROFILING, phase_index=None, concurrency=1
    )
    runner._lifecycle = SimpleNamespace(is_complete=False)
    runner._credit_issuer = f.issuer
    runner._session_tree_registry = f.callback._session_tree_registry
    runner._concurrency_manager = f.real_cm
    runner._progress_task = None
    runner._on_phase_complete = None
    runner.warning = MagicMock()
    runner.info = MagicMock()
    runner._release_stuck_slots()
    runner._lifecycle.is_complete = True
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    runner._on_return_wait_complete(task)
    runner._release_tree_slots()
    assert f.real_cm.get_session_stats(CreditPhase.PROFILING).release_count == 1
    assert await f.real_cm.acquire_session_slot(CreditPhase.PROFILING, lambda: True)
    assert not f.real_cm.session_slot_available(CreditPhase.PROFILING)
    f.real_cm.release_session_slot(CreditPhase.PROFILING)


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_sessions", [pytest.param(1,id="sole-root"),pytest.param(2,id="two-roots-already-issued")])  # fmt: skip
async def test_abort_retires_unsent_plan_and_signals_session_quota_without_another_send(
    abort_flow: SimpleNamespace, expected_sessions: int
) -> None:
    """The last child abort itself must unblock phase sending, without duration expiry."""
    f = await prepare_legacy(
        abort_flow,
        root_final=False,
        with_orphan=False,
        expected_sessions=expected_sessions,
    )
    await returned(f, f.root)
    if expected_sessions == 2:
        # Admit a second root before the failure; no later issue can accidentally
        # supply the sending-complete event missing from abort settlement.
        f.real_cm.configure_for_phase(
            CreditPhase.PROFILING, concurrency=2, prefill_concurrency=8
        )
        await f.issuer.issue_credit(
            TurnToSend(
                conversation_id="other",
                x_correlation_id="other",
                turn_index=0,
                num_turns=1,
            )
        )
        other = f.router.send_credit.await_args.kwargs["credit"]
        await returned(f, other)
    assert not f.real_progress.all_credits_sent_event.is_set()
    assert not f.real_progress.all_credits_returned_event.is_set()
    sent_before = f.real_progress.counter.requests_sent
    await returned(f, f.children["failed"], error="intentional child failure")
    assert f.real_progress.all_credits_sent_event.is_set()
    assert f.real_progress.all_credits_returned_event.is_set()
    assert f.real_progress.counter.requests_sent == sent_before == expected_sessions + 1
    assert f.real_progress.counter.requests_completed == sent_before
    assert f.real_progress.counter.requests_cancelled == 0
    assert f.real_progress.counter.cancelled_sessions == 1
    assert f.real_progress.in_flight_sessions == 0
