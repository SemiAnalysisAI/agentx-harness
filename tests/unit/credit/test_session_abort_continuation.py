# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logical aborts must survive request returns, think times and admission races."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from msgspec.structs import replace

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.dispatch import ChildDispatchResult
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import CreditReturn
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.branch_orchestrator import BranchOrchestrator, ChildJoinEntry
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.strategies.request_rate import RequestRateStrategy


@pytest.fixture
def abort_flow(benchmark_run: BenchmarkRun) -> SimpleNamespace:
    """Real control functions with an observable mock wire and available slots."""
    phase = CreditPhase.PROFILING
    cm = MagicMock()
    cm.acquire_prefill_slot = AsyncMock(return_value=True)
    progress = MagicMock()
    progress.increment_sent.return_value = (2, False)
    progress.increment_returned.return_value = False
    progress.check_all_returned_or_cancelled.return_value = False
    progress.all_credits_returned_event = asyncio.Event()
    lifecycle = SimpleNamespace(
        is_complete=False,
        is_sending_complete=False,
        started_at_ns=time.time_ns(),
        started_at_perf_ns=time.perf_counter_ns(),
    )
    stop = MagicMock()
    stop.can_send_child_turn.return_value = True
    stop.can_send_any_turn.return_value = True
    policy = MagicMock()
    policy.next_cancellation_delay_ns.return_value = None
    router = StickyCreditRouter(run=benchmark_run, service_id="abort-flow")
    router._router_client.send_to = AsyncMock()
    router._register_worker("worker")
    router._track_credit_sent("worker", 1, "orphan")
    router.send_credit = AsyncMock()
    issuer = CreditIssuer(
        phase=phase,
        stop_checker=stop,
        progress=progress,
        concurrency_manager=cm,
        credit_router=router,
        cancellation_policy=policy,
        lifecycle=lifecycle,
    )
    meta = SimpleNamespace(delay_ms=None, has_forks=False, branch_ids=[])
    source = SimpleNamespace(
        dataset_metadata=None,
        get_metadata=lambda _: SimpleNamespace(turns=[meta, meta]),
        get_next_turn_metadata=lambda _: meta,
    )
    branch = BranchOrchestrator(source, issuer)
    branch._fail_fast = True
    branch._child_to_join = {
        cid: [ChildJoinEntry("parent", None, None)] for cid in ("failed", "orphan")
    }
    branch._child_modes = {
        cid: ConversationBranchMode.SPAWN for cid in ("failed", "orphan")
    }
    branch._descendant_counts = {"parent": 2}
    strategy = object.__new__(RequestRateStrategy)
    strategy._conversation_source = source
    strategy._credit_issuer = issuer
    strategy._branch_orchestrator = branch
    strategy._continuation_turns = asyncio.Queue()
    callback = CreditCallbackHandler(cm, branch_orchestrator=branch)
    callback.register_phase(
        phase=phase,
        progress=progress,
        lifecycle=lifecycle,
        stop_checker=stop,
        strategy=strategy,
    )
    router._on_return_callback = callback.on_credit_return
    credit = Credit(
        id=1,
        phase=phase,
        conversation_id="child-template",
        x_correlation_id="orphan",
        turn_index=0,
        num_turns=2,
        issued_at_ns=time.time_ns(),
        agent_depth=1,
        parent_correlation_id="parent",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    return SimpleNamespace(
        branch=branch,
        issuer=issuer,
        router=router,
        callback=callback,
        strategy=strategy,
        credit=credit,
        meta=meta,
        concurrency=cm,
        progress=progress,
        stop=stop,
    )


@pytest.mark.asyncio
async def test_cancelled_nonfinal_orphan_cannot_resume(
    abort_flow: SimpleNamespace,
) -> None:
    """Exercise the actual return -> intercept -> strategy -> issuance path."""
    f = abort_flow
    await f.branch.on_child_errored("failed")
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=f.credit, cancelled=True, first_token_sent=False, error=None
        ),
    )
    f.router.send_credit.assert_not_awaited()
    assert "orphan" not in f.branch._child_to_join
    assert not f.branch.has_pending_branch_work()
    f.progress.increment_returned.assert_called_once()


@pytest.mark.asyncio
async def test_parked_orphan_cannot_resume_after_abort(
    abort_flow: SimpleNamespace,
) -> None:
    """A timer that was already scheduled cannot revive the stopped session."""
    f = abort_flow
    f.router._track_credit_returned("worker", 1, False, False)
    f.branch.park_child_turn("orphan")
    await f.branch.on_child_errored("failed")
    await f.strategy._issue_parked_child_continuation(
        TurnToSend.from_previous_credit(f.credit, f.meta), f.credit
    )
    f.router.send_credit.assert_not_awaited()
    assert not f.branch._parked_child_turns


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [pytest.param("child", id="child"), pytest.param("blocking", id="blocking"), pytest.param("polling", id="polling")],
)  # fmt: skip
async def test_aborted_session_refused_on_every_issuance_path(
    abort_flow: SimpleNamespace,
    path: str,
) -> None:
    """Queued continuations and root joins share terminal admission."""
    f = abort_flow
    await f.issuer.abort_session("orphan")
    turn = TurnToSend.from_previous_credit(f.credit, f.meta)
    if path == "child":
        assert await f.issuer.dispatch_child_turn(turn) is ChildDispatchResult.REJECTED
    elif path == "blocking":
        assert await f.issuer.issue_credit(turn) is False
    else:
        assert await f.issuer.try_issue_credit(turn) is False
    f.router.send_credit.assert_not_awaited()
    f.concurrency.acquire_prefill_slot.assert_not_awaited()
    f.progress.increment_sent.assert_not_called()


@pytest.mark.asyncio
async def test_abort_during_prefill_wait_releases_only_acquired_slot(
    abort_flow: SimpleNamespace,
) -> None:
    """Recheck terminal admission after an awaited slot becomes available."""
    f = abort_flow

    async def acquire(*_args: object) -> bool:
        await f.issuer.abort_session("orphan")
        return True

    f.concurrency.acquire_prefill_slot.side_effect = acquire
    result = await f.issuer.dispatch_child_turn(
        TurnToSend.from_previous_credit(f.credit, f.meta)
    )
    assert result is ChildDispatchResult.REJECTED
    f.concurrency.release_prefill_slot.assert_called_once()
    f.concurrency.release_session_slot.assert_not_called()
    f.router.send_credit.assert_not_awaited()
    f.progress.increment_sent.assert_not_called()


@pytest.mark.asyncio
async def test_all_siblings_closed_before_first_cancellation_await(
    abort_flow: SimpleNamespace,
) -> None:
    """A sibling cannot acquire capacity while another cancellation is sent."""
    f = abort_flow
    f.router._track_credit_sent("worker", 99, "parent")
    observed = []

    async def send(*_args: object) -> None:
        check = getattr(f.issuer, "is_session_aborted", lambda _: False)
        observed.append(all(check(x) for x in ["parent", "failed", "orphan"]))
        await f.issuer.dispatch_child_turn(
            TurnToSend.from_previous_credit(f.credit, f.meta)
        )

    f.router._router_client.send_to.side_effect = send
    await f.branch.on_child_errored("failed")
    assert observed and all(observed)
    f.router.send_credit.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_cancellation_without_logical_abort_preserves_continuation(
    abort_flow: SimpleNamespace,
) -> None:
    """Intentional request cancellation is not a terminal session abort."""
    f = abort_flow
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=f.credit, cancelled=True, first_token_sent=False, error=None
        ),
    )
    sent = f.router.send_credit.await_args.kwargs["credit"]
    assert sent.turn_index == 1 and sent.x_correlation_id == "orphan"


@pytest.mark.asyncio
async def test_unrelated_session_continues_after_failfast(
    abort_flow: SimpleNamespace,
) -> None:
    """Failfast is scoped to the failed parent's branch, not the whole phase."""
    f = abort_flow
    await f.branch.on_child_errored("failed")
    f.router._track_credit_sent("worker", 3, "unrelated")
    other = replace(f.credit, id=3, x_correlation_id="unrelated")
    await f.router._handle_router_message(
        "worker",
        CreditReturn(credit=other, cancelled=False, first_token_sent=False, error=None),
    )
    assert (
        f.router.send_credit.await_args.kwargs["credit"].x_correlation_id == "unrelated"
    )


@pytest.mark.asyncio
async def test_aborted_return_cannot_spawn_fresh_descendants(
    abort_flow: SimpleNamespace,
) -> None:
    """A cancelled spawning turn cannot start grandchildren with fresh session IDs."""
    f = abort_flow
    f.branch.get_branch_ids = MagicMock(return_value=["nested-branch"])
    f.branch._spawn_children_and_register_gates = AsyncMock()
    await f.branch.on_child_errored("failed")
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=f.credit, cancelled=True, first_token_sent=False, error=None
        ),
    )
    f.branch._spawn_children_and_register_gates.assert_not_awaited()
    f.router.send_credit.assert_not_awaited()


def attach_registry(
    flow: SimpleNamespace, *, root_pending: bool = True
) -> SessionTreeRegistry:
    """Hold one real tree slot across the failing child and active orphan."""
    registry = SessionTreeRegistry(flow.concurrency)
    registry.open_tree("parent", CreditPhase.PROFILING, root_pending=root_pending)
    registry.register_descendants("parent", 2)
    flow.branch._session_tree_registry = registry
    flow.callback._session_tree_registry = registry
    flow.branch._child_root = {"failed": "parent", "orphan": "parent"}
    return registry


@pytest.mark.asyncio
@pytest.mark.parametrize("root_final", [pytest.param(True,id="final"),pytest.param(False,id="nonfinal")])  # fmt: skip
async def test_aborted_root_return_finishes_tree_only_after_active_orphan(
    abort_flow: SimpleNamespace,
    root_final: bool,
) -> None:
    """Root cancellation is terminal but cannot release a still-active descendant."""
    f = abort_flow
    registry = attach_registry(f)
    f.router._track_credit_sent("worker", 99, "parent")
    root = replace(
        f.credit,
        id=99,
        x_correlation_id="parent",
        agent_depth=0,
        parent_correlation_id=None,
        root_correlation_id="parent",
        turn_index=1 if root_final else 0,
    )
    await f.branch.on_child_errored("failed")
    f.concurrency.release_session_slot.assert_not_called()
    await f.router._handle_router_message(
        "worker",
        CreditReturn(credit=root, cancelled=True, first_token_sent=False, error=None),
    )
    assert registry.has_tree("parent")
    f.concurrency.release_session_slot.assert_not_called()
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=f.credit, cancelled=True, first_token_sent=False, error=None
        ),
    )
    assert not registry.has_tree("parent")
    assert not f.branch.has_pending_branch_work()
    f.concurrency.release_session_slot.assert_called_once_with(CreditPhase.PROFILING)
    f.router.send_credit.assert_not_awaited()
    assert registry.late_events == 0


@pytest.mark.asyncio
async def test_gated_root_without_live_credit_settles_after_last_child_return(
    abort_flow: SimpleNamespace,
) -> None:
    """An aborted gated root cannot leave root_pending stuck until teardown."""
    f = abort_flow
    registry = attach_registry(f)
    await f.branch.on_child_errored("failed")
    assert registry.has_tree("parent") and f.branch.has_pending_branch_work()
    f.concurrency.release_session_slot.assert_not_called()
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=f.credit, cancelled=True, first_token_sent=False, error=None
        ),
    )
    assert not registry.has_tree("parent")
    f.concurrency.release_session_slot.assert_called_once_with(CreditPhase.PROFILING)
    assert registry.late_events == 0


@pytest.mark.asyncio
async def test_nested_active_grandchild_keeps_terminal_roots_slot(
    abort_flow: SimpleNamespace,
) -> None:
    """Abort reaches grandchildren even when the ancestor root has already finished."""
    f = abort_flow
    registry = attach_registry(f, root_pending=False)
    registry.register_descendants("parent")
    f.branch._child_to_join["grandchild"] = [ChildJoinEntry("orphan", None, None)]
    f.branch._child_modes["grandchild"] = ConversationBranchMode.SPAWN
    f.branch._child_root["grandchild"] = "parent"
    f.router._track_credit_sent("worker", 22, "grandchild")
    await f.branch.on_child_errored("failed")
    assert f.issuer.is_session_aborted("grandchild")
    f.concurrency.release_session_slot.assert_not_called()
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=f.credit, cancelled=True, first_token_sent=False, error=None
        ),
    )
    assert registry.has_tree("parent")
    f.concurrency.release_session_slot.assert_not_called()
    grandchild = replace(
        f.credit,
        id=22,
        x_correlation_id="grandchild",
        agent_depth=2,
        parent_correlation_id="orphan",
        root_correlation_id="parent",
    )
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=grandchild, cancelled=True, first_token_sent=False, error=None
        ),
    )
    assert not registry.has_tree("parent")
    f.concurrency.release_session_slot.assert_called_once_with(CreditPhase.PROFILING)
    assert registry.late_events == 0


@pytest.mark.asyncio
async def test_aborted_session_retained_until_its_last_credit_returns(
    abort_flow: SimpleNamespace,
) -> None:
    """Multiple active credits with one session ID must not release on the first return."""
    f = abort_flow
    registry = attach_registry(f)
    f.router._track_credit_sent("worker", 2, "orphan")
    await f.branch.on_child_errored("failed")
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=f.credit, cancelled=True, first_token_sent=False, error=None
        ),
    )
    assert registry.has_tree("parent") and f.branch.has_pending_branch_work()
    f.concurrency.release_session_slot.assert_not_called()
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=replace(f.credit, id=2),
            cancelled=True,
            first_token_sent=False,
            error=None,
        ),
    )
    assert not registry.has_tree("parent")
    f.concurrency.release_session_slot.assert_called_once_with(CreditPhase.PROFILING)
    assert registry.late_events == 0


@pytest.mark.asyncio
async def test_aborted_root_retained_until_its_last_credit_returns(
    abort_flow: SimpleNamespace,
) -> None:
    """A terminal root return cannot free a lane still owned by another root credit."""
    f = abort_flow
    registry = attach_registry(f)
    for credit_id in [99, 100]:
        f.router._track_credit_sent("worker", credit_id, "parent")
    root = replace(
        f.credit,
        id=99,
        x_correlation_id="parent",
        agent_depth=0,
        parent_correlation_id=None,
        root_correlation_id="parent",
        turn_index=1,
    )
    await f.branch.on_child_errored("failed")
    await f.router._handle_router_message(
        "worker",
        CreditReturn(credit=root, cancelled=True, first_token_sent=False, error=None),
    )
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=f.credit, cancelled=True, first_token_sent=False, error=None
        ),
    )
    assert registry.has_tree("parent")
    f.concurrency.release_session_slot.assert_not_called()
    await f.router._handle_router_message(
        "worker",
        CreditReturn(
            credit=replace(root, id=100),
            cancelled=True,
            first_token_sent=False,
            error=None,
        ),
    )
    assert not registry.has_tree("parent")
    f.concurrency.release_session_slot.assert_called_once_with(CreditPhase.PROFILING)
    assert registry.late_events == 0


@pytest.mark.asyncio
async def test_overlap_spawn_rechecks_abort_after_waiting_for_parent_lock(
    abort_flow: SimpleNamespace,
) -> None:
    """A previously admitted overlap hook cannot spawn fresh IDs after the abort."""
    f = abort_flow
    entered, released = asyncio.Event(), asyncio.Event()

    class HeldLock:
        async def __aenter__(self) -> None:
            entered.set()
            await released.wait()

        async def __aexit__(self, *_args: object) -> None:
            return None

    f.branch._parent_locks["parent"] = HeldLock()
    f.branch._cs.get_metadata = MagicMock(
        return_value=SimpleNamespace(
            replay_scope_id="trace",
            branches=[
                SimpleNamespace(branch_id="b", mode=ConversationBranchMode.SPAWN)
            ],
            turns=[SimpleNamespace(timestamp_ms=0, api_time_ms=100, branch_ids=["b"])],
        )
    )
    f.branch._branch_start_timestamp_ms = MagicMock(return_value=20)
    f.branch._spawn_children_and_register_gates = AsyncMock()
    credit = replace(
        f.credit, x_correlation_id="parent", agent_depth=0, parent_correlation_id=None
    )
    task = asyncio.create_task(f.branch.on_credit_issued(credit))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await f.branch.on_child_errored("failed")
    finally:
        released.set()
        await task
    f.branch._spawn_children_and_register_gates.assert_not_awaited()
