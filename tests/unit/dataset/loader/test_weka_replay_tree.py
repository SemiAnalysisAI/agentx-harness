# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``_weka_replay_tree.plan_session_tree``.

Scenarios run through the real ``normalize_session -> plan_session_tree``
pipeline via :func:`normalize_session_dict`, so a regression in the IR
layer's link classification (``_weka_replay_ir.py``) is also caught here -
only the one deliberately-unreachable defensive bug-guard branch
(:class:`TestUnreachableInstanceBugGuard`) hand-builds IR dataclasses
directly, since a real causal cycle can't survive the IR layer's own
classification (time-inversion/self-spawn/group-mismatch guards).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aiperf.dataset.loader._weka_replay_ir import (
    InstanceIR,
    LinkIR,
    LinkKind,
    ReplayRole,
    RequestIR,
    SamplingIR,
    SessionDiagnostics,
    SessionIR,
)
from aiperf.dataset.loader._weka_replay_tree import (
    PlannedConversation,
    SessionTreePlan,
    TreeFilterOptions,
    plan_session_tree,
)
from tests.unit.dataset.loader._weka_replay_builders import (
    build_instance,
    build_request,
    build_session,
    normalize_session_dict,
)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _ts(offset_ms: float) -> str:
    """RFC3339 (millisecond precision) timestamp ``offset_ms`` after a fixed base."""
    dt = _BASE + timedelta(milliseconds=offset_ms)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _ms(offset_ms: float) -> float:
    """The absolute epoch-ms value the IR layer will parse ``_ts(offset_ms)`` into."""
    dt = _BASE + timedelta(milliseconds=offset_ms)
    return dt.timestamp() * 1000.0


def _plan(
    session_dict: dict, **option_overrides
) -> tuple[SessionTreePlan | None, SessionDiagnostics]:
    ir, diag = normalize_session_dict(session_dict)
    options = TreeFilterOptions(**option_overrides)
    plan = plan_session_tree(ir, options=options, diag=diag)
    return plan, diag


def _find(plan: SessionTreePlan, cid: str) -> PlannedConversation:
    for conv in plan.conversations:
        if conv.cid == cid:
            return conv
    raise AssertionError(f"no conversation with cid {cid!r} in plan")


def _branch(conv: PlannedConversation, branch_id: str):
    for b in conv.branches:
        if b.branch_id == branch_id:
            return b
    raise AssertionError(
        f"no branch {branch_id!r} on {conv.cid!r}; have "
        f"{[b.branch_id for b in conv.branches]}"
    )


def _assert_tree_invariants(plan: SessionTreePlan) -> None:
    """Generic structural properties that must hold for any planned tree."""
    cids = [c.cid for c in plan.conversations]
    assert len(cids) == len(set(cids)), "duplicate cid in planned conversations"

    index_of = {cid: i for i, cid in enumerate(cids)}
    by_cid = {c.cid: c for c in plan.conversations}

    for conv in plan.conversations:
        if conv.parent_cid is not None:
            assert conv.parent_cid in index_of, (
                f"{conv.cid}'s parent_cid {conv.parent_cid!r} not in plan"
            )
            # Topological order: parent strictly before child.
            assert index_of[conv.parent_cid] < index_of[conv.cid]
            # The child must appear in exactly one of its parent's branches.
            parent = by_cid[conv.parent_cid]
            owning_branches = [b for b in parent.branches if conv.cid in b.child_cids]
            assert len(owning_branches) == 1, (
                f"{conv.cid} should be declared by exactly one branch of "
                f"{conv.parent_cid}, found {len(owning_branches)}"
            )
        else:
            assert conv.is_root, f"{conv.cid} has no parent_cid but is_root=False"

        for branch in conv.branches:
            if branch.is_background:
                assert branch.join_turn is None
            else:
                assert branch.join_turn == branch.declaring_turn + 1

        # Acyclic: walking parent_cid pointers from any conversation must
        # terminate at a root without revisiting a node.
        seen: set[str] = set()
        cur = conv.cid
        while cur is not None:
            assert cur not in seen, f"cycle detected walking ancestors of {conv.cid}"
            seen.add(cur)
            cur = by_cid[cur].parent_cid


class TestLinkClassificationDepthAndSyncAsync:
    """One rich fixture covering every §4.1 link-classification row, 3-level
    nesting, the sync/async split (including last-turn-is-always-async), and
    co-session anchoring both before the root's first request and between
    two root requests.
    """

    @pytest.fixture
    def session_dict(self):
        main = build_instance(
            instance_id="a:main",
            role="main",
            requests=[
                build_request(request_id=101, ts=_ts(0), total_ms=500),
                build_request(request_id=102, ts=_ts(2000), total_ms=500),
                build_request(request_id=103, ts=_ts(4000), total_ms=500),
            ],
        )
        # CAUSAL, sync (ends well before turn 102 begins).
        sync_child = build_instance(
            instance_id="a:sync",
            role="sub-agent",
            fan_out_group="101",
            parent_spawn_request_id=101,
            fan_out_size=2,
            fan_out_position=0,
            requests=[build_request(request_id=201, ts=_ts(600), total_ms=300)],
        )
        # CAUSAL, async (still running well past turn 102's start).
        async_child = build_instance(
            instance_id="a:async",
            role="sub-agent",
            fan_out_group="101",
            parent_spawn_request_id=101,
            fan_out_size=2,
            fan_out_position=1,
            requests=[build_request(request_id=301, ts=_ts(1000), total_ms=5000)],
        )
        # CAUSAL depth-2: child of a:sync's only (= last) turn -> always async.
        depth2 = build_instance(
            instance_id="a:sync:child2",
            role="sub-agent",
            fan_out_group="201",
            parent_spawn_request_id=201,
            fan_out_size=1,
            fan_out_position=0,
            requests=[build_request(request_id=401, ts=_ts(650), total_ms=100)],
        )
        # CAUSAL depth-3: child of a:sync:child2's only (= last) turn -> async.
        depth3 = build_instance(
            instance_id="a:sync:child2:child3",
            role="sub-agent",
            fan_out_group="401",
            parent_spawn_request_id=401,
            fan_out_size=1,
            fan_out_position=0,
            requests=[build_request(request_id=501, ts=_ts(700), total_ms=20)],
        )
        # IMPLICIT: chronological filler only, not causal.
        implicit = build_instance(
            instance_id="a:implicit",
            role="helper-or-isolated",
            parent_spawn_request_id=101,
            requests=[build_request(request_id=601, ts=_ts(1200), total_ms=100)],
        )
        # CYCLE_BROKEN: fan_out_group set, no parent_spawn_request_id.
        cycle = build_instance(
            instance_id="a:cycle",
            role="helper-or-isolated",
            fan_out_group="999",
            requests=[build_request(request_id=701, ts=_ts(1500), total_ms=100)],
        )
        # ANOMALOUS: fan_out_group doesn't match parent_spawn_request_id.
        anomalous = build_instance(
            instance_id="a:anom_gm",
            role="helper-or-isolated",
            fan_out_group="555",
            parent_spawn_request_id=101,
            requests=[build_request(request_id=801, ts=_ts(1600), total_ms=100)],
        )
        # NONE, standalone, starts before the root's first request.
        before_root = build_instance(
            instance_id="a:before_root",
            role="helper-or-isolated",
            requests=[build_request(request_id=901, ts=_ts(-500), total_ms=100)],
        )
        # NONE, standalone, starts between the root's turn 1 and turn 2.
        between = build_instance(
            instance_id="a:between",
            role="helper-or-isolated",
            requests=[build_request(request_id=902, ts=_ts(3000), total_ms=100)],
        )
        return build_session(
            session_id="sess_a",
            instances=[
                main,
                sync_child,
                async_child,
                depth2,
                depth3,
                implicit,
                cycle,
                anomalous,
                before_root,
                between,
            ],
        )

    def test_full_tree_shape(self, session_dict):
        plan, diag = _plan(session_dict)
        assert plan is not None
        assert diag.cycle_broken_count == 1
        assert diag.anomalous_count == 1

        root = _find(plan, "a:main")
        assert root.is_root
        assert root.agent_depth == 0
        assert root.parent_cid is None

        sync_branch = _branch(root, "a:main:0:spawn")
        assert sync_branch.is_background is False
        assert sync_branch.join_turn == 1
        assert sync_branch.child_cids == ("a:sync",)

        async_branch = _branch(root, "a:main:0:spawn_bg")
        assert async_branch.is_background is True
        assert async_branch.join_turn is None
        assert async_branch.child_cids == ("a:async",)

        # All four non-causal streams anchored before/at turn 0 (implicit at
        # 1200ms, cycle at 1500ms, anomalous at 1600ms, before_root at
        # -500ms - all <= turn 1's start at 2000ms) share ONE branch,
        # ordered by (first_ts_ms, instance_id).
        cosession_0 = _branch(root, "a:main:0:cosession")
        assert cosession_0.is_background is True
        assert cosession_0.join_turn is None
        assert cosession_0.child_cids == (
            "a:before_root",
            "a:implicit",
            "a:cycle",
            "a:anom_gm",
        )

        # "between" starts at 3000ms, after turn 1 (2000ms) but before turn 2
        # (4000ms) -> anchored at turn index 1, its own branch.
        cosession_1 = _branch(root, "a:main:1:cosession")
        assert cosession_1.child_cids == ("a:between",)

        for cid in (
            "a:sync",
            "a:async",
            "a:implicit",
            "a:cycle",
            "a:anom_gm",
            "a:before_root",
            "a:between",
        ):
            conv = _find(plan, cid)
            assert conv.agent_depth == 1
            assert conv.parent_cid == "a:main"
            assert conv.is_root is False

        sync_conv = _find(plan, "a:sync")
        # a:sync has only one request -> spawning a:sync:child2 at its own
        # (last) turn is always async, regardless of timing.
        depth2_branch = _branch(sync_conv, "a:sync:0:spawn_bg")
        assert depth2_branch.is_background is True
        assert depth2_branch.child_cids == ("a:sync:child2",)

        depth2_conv = _find(plan, "a:sync:child2")
        assert depth2_conv.agent_depth == 2
        assert depth2_conv.parent_cid == "a:sync"
        depth3_branch = _branch(depth2_conv, "a:sync:child2:0:spawn_bg")
        assert depth3_branch.child_cids == ("a:sync:child2:child3",)

        depth3_conv = _find(plan, "a:sync:child2:child3")
        assert depth3_conv.agent_depth == 3
        assert depth3_conv.parent_cid == "a:sync:child2"
        assert depth3_conv.branches == ()

        assert len(plan.conversations) == 10
        _assert_tree_invariants(plan)


class TestSyncAsyncToleranceBoundary:
    def test_exact_1ms_boundary_is_sync_and_just_over_is_async(self):
        main = build_instance(
            instance_id="b:main",
            role="main",
            requests=[
                build_request(request_id=1, ts=_ts(0), total_ms=10),
                build_request(request_id=2, ts=_ts(1000), total_ms=10),
            ],
        )
        # last_end_ms == 500 + 501.0 == 1001.0 == t_next(1000) + 1ms exactly.
        child_exact = build_instance(
            instance_id="b:exact",
            role="sub-agent",
            fan_out_group="1",
            parent_spawn_request_id=1,
            fan_out_size=2,
            fan_out_position=0,
            requests=[build_request(request_id=11, ts=_ts(500), total_ms=501.0)],
        )
        # last_end_ms == 500 + 501.1 == 1001.1 - just over the tolerance.
        child_over = build_instance(
            instance_id="b:over",
            role="sub-agent",
            fan_out_group="1",
            parent_spawn_request_id=1,
            fan_out_size=2,
            fan_out_position=1,
            requests=[build_request(request_id=12, ts=_ts(500), total_ms=501.1)],
        )
        session_dict = build_session(
            session_id="sess_b", instances=[main, child_exact, child_over]
        )
        plan, _diag = _plan(session_dict)
        assert plan is not None
        root = _find(plan, "b:main")
        sync_branch = _branch(root, "b:main:0:spawn")
        assert sync_branch.child_cids == ("b:exact",)
        async_branch = _branch(root, "b:main:0:spawn_bg")
        assert async_branch.child_cids == ("b:over",)
        _assert_tree_invariants(plan)


class TestSessionMode:
    @pytest.fixture
    def two_stream_session(self):
        main = build_instance(
            instance_id="c:main",
            role="main",
            requests=[build_request(request_id=1, ts=_ts(0))],
        )
        other = build_instance(
            instance_id="c:other",
            role="helper-or-isolated",
            requests=[build_request(request_id=2, ts=_ts(100))],
        )
        return build_session(session_id="sess_c", instances=[main, other])

    def test_tree_mode_attaches_other_as_cosession_child(self, two_stream_session):
        plan, _diag = _plan(two_stream_session, session_mode="tree")
        assert plan is not None
        roots = [c for c in plan.conversations if c.is_root]
        assert len(roots) == 1
        assert roots[0].cid == "c:main"
        other = _find(plan, "c:other")
        assert other.is_root is False
        assert other.agent_depth == 1
        assert other.parent_cid == "c:main"
        _assert_tree_invariants(plan)

    def test_independent_mode_both_are_roots(self, two_stream_session):
        plan, _diag = _plan(two_stream_session, session_mode="independent")
        assert plan is not None
        roots = [c for c in plan.conversations if c.is_root]
        assert {c.cid for c in roots} == {"c:main", "c:other"}
        for root in roots:
            assert root.agent_depth == 0
            assert root.parent_cid is None
            assert root.branches == ()
        _assert_tree_invariants(plan)


class TestOrphanSessions:
    @pytest.fixture
    def two_stream_instances(self):
        main = build_instance(
            instance_id="orphan:no-session-id:main",
            role="main",
            requests=[build_request(request_id=1, ts=_ts(0))],
        )
        other = build_instance(
            instance_id="orphan:no-session-id:other",
            role="helper-or-isolated",
            requests=[build_request(request_id=2, ts=_ts(100))],
        )
        return main, other

    def test_shared_bucket_dropped_by_default(self, two_stream_instances):
        main, other = two_stream_instances
        session_dict = build_session(
            session_id="orphan:no-session-id", instances=[main, other]
        )
        plan, diag = _plan(session_dict, include_orphans=False)
        assert plan is None
        assert diag.orphan_session_dropped == 1

    def test_shared_bucket_forced_independent_even_in_tree_mode(
        self, two_stream_instances
    ):
        main, other = two_stream_instances
        session_dict = build_session(
            session_id="orphan:no-session-id", instances=[main, other]
        )
        plan, _diag = _plan(session_dict, include_orphans=True, session_mode="tree")
        assert plan is not None
        roots = [c for c in plan.conversations if c.is_root]
        # Forced independent: both instances are roots, no cosession branch,
        # even though the caller asked for "tree" mode.
        assert {c.cid for c in roots} == {
            "orphan:no-session-id:main",
            "orphan:no-session-id:other",
        }
        main_conv = _find(plan, "orphan:no-session-id:main")
        assert main_conv.branches == ()

    def test_per_record_orphan_follows_requested_session_mode(
        self, two_stream_instances
    ):
        main, other = two_stream_instances
        session_dict = build_session(
            session_id="orphan:no-session-id:abc123", instances=[main, other]
        )
        plan, _diag = _plan(session_dict, include_orphans=True, session_mode="tree")
        assert plan is not None
        roots = [c for c in plan.conversations if c.is_root]
        # Per-record orphan shape: real tree mode applies, not forced
        # independent - only the designated root is is_root.
        assert len(roots) == 1


class TestEphemeralAndRoleFiltering:
    @pytest.fixture
    def session_with_ephemeral_and_helper(self):
        main = build_instance(
            instance_id="d:main",
            role="main",
            requests=[build_request(request_id=1, ts=_ts(0))],
        )
        ephemeral = build_instance(
            instance_id="d:ephemeral",
            role="ephemeral (no system)",
            requests=[build_request(request_id=2, ts=_ts(100))],
        )
        helper = build_instance(
            instance_id="d:helper",
            role="helper-or-isolated",
            requests=[build_request(request_id=3, ts=_ts(200))],
        )
        return build_session(session_id="sess_d", instances=[main, ephemeral, helper])

    def test_ephemeral_dropped_by_default(self, session_with_ephemeral_and_helper):
        plan, diag = _plan(session_with_ephemeral_and_helper)
        assert plan is not None
        assert diag.ephemeral_dropped_count == 1
        assert all(c.cid != "d:ephemeral" for c in plan.conversations)

    def test_ephemeral_included_when_opted_in(self, session_with_ephemeral_and_helper):
        plan, diag = _plan(session_with_ephemeral_and_helper, include_ephemeral=True)
        assert plan is not None
        assert diag.ephemeral_dropped_count == 0
        assert any(c.cid == "d:ephemeral" for c in plan.conversations)

    def test_role_allow_list_drops_non_listed_roles(
        self, session_with_ephemeral_and_helper
    ):
        plan, diag = _plan(
            session_with_ephemeral_and_helper,
            include_roles=frozenset({ReplayRole.MAIN}),
        )
        assert plan is not None
        assert diag.role_filtered_dropped_count == 1
        cids = {c.cid for c in plan.conversations}
        assert cids == {"d:main"}


class TestCascadeDrop:
    def test_causal_descendant_of_dropped_instance_is_dropped_but_implicit_child_is_not(
        self,
    ):
        main = build_instance(
            instance_id="e:main",
            role="main",
            requests=[build_request(request_id=1, ts=_ts(0))],
        )
        ephemeral = build_instance(
            instance_id="e:eph",
            role="ephemeral (no system)",
            requests=[build_request(request_id=10, ts=_ts(100))],
        )
        causal_descendant = build_instance(
            instance_id="e:eph:causal_child",
            role="sub-agent",
            fan_out_group="10",
            parent_spawn_request_id=10,
            requests=[build_request(request_id=11, ts=_ts(150))],
        )
        implicit_descendant = build_instance(
            instance_id="e:eph:implicit_child",
            role="helper-or-isolated",
            parent_spawn_request_id=10,
            requests=[build_request(request_id=12, ts=_ts(160))],
        )
        session_dict = build_session(
            session_id="sess_e",
            instances=[main, ephemeral, causal_descendant, implicit_descendant],
        )
        plan, diag = _plan(session_dict)  # ephemeral dropped by default
        assert plan is not None
        assert diag.ephemeral_dropped_count == 1
        assert diag.cascade_dropped_count == 1
        cids = {c.cid for c in plan.conversations}
        assert "e:eph" not in cids
        assert "e:eph:causal_child" not in cids, (
            "a CAUSAL descendant of a dropped instance must cascade-drop"
        )
        assert "e:eph:implicit_child" in cids, (
            "an IMPLICIT (chronological-only) 'child' of a dropped instance "
            "is not a real causal descendant and must survive as its own stream"
        )
        implicit_conv = _find(plan, "e:eph:implicit_child")
        assert implicit_conv.parent_cid == "e:main"
        assert implicit_conv.is_root is False


class TestSynthesisMaxIsl:
    def test_partial_drop_and_full_empty_with_causal_parent_demotion(self):
        main = build_instance(
            instance_id="f:main",
            role="main",
            requests=[build_request(request_id=1, ts=_ts(0))],
        )
        # One over-threshold request dropped, one survives: instance kept,
        # rebuilt around the surviving request.
        partial = build_instance(
            instance_id="f:partial",
            role="sub-agent",
            fan_out_group="1",
            parent_spawn_request_id=1,
            fan_out_size=2,
            fan_out_position=0,
            requests=[
                build_request(request_id=21, ts=_ts(100), input_tokens=200),
                build_request(request_id=22, ts=_ts(200), input_tokens=50),
            ],
        )
        # Every request over threshold: instance emptied and dropped.
        emptied = build_instance(
            instance_id="f:emptied",
            role="sub-agent",
            fan_out_group="1",
            parent_spawn_request_id=1,
            fan_out_size=2,
            fan_out_position=1,
            requests=[build_request(request_id=31, ts=_ts(300), input_tokens=500)],
        )
        # Causally linked to f:emptied's only (now-dropped) request - should
        # be DEMOTED to a non-causal stream, not cascade-dropped.
        emptied_child = build_instance(
            instance_id="f:emptied_child",
            role="sub-agent",
            fan_out_group="31",
            parent_spawn_request_id=31,
            requests=[build_request(request_id=32, ts=_ts(350), input_tokens=30)],
        )
        session_dict = build_session(
            session_id="sess_f",
            instances=[main, partial, emptied, emptied_child],
        )
        plan, diag = _plan(session_dict, synthesis_max_isl=100)
        assert plan is not None
        assert diag.dropped_for_max_isl_count == 1
        assert diag.causal_parent_dropped_by_max_isl_count == 1

        partial_conv = _find(plan, "f:partial")
        assert [r.request_id for r in partial_conv.instance.requests] == [22]
        assert partial_conv.instance.first_ts_ms == pytest.approx(_ms(200))

        cids = {c.cid for c in plan.conversations}
        assert "f:emptied" not in cids

        demoted_conv = _find(plan, "f:emptied_child")
        assert demoted_conv.parent_cid == "f:main", (
            "demoted (not cascade-dropped) child becomes its own non-causal "
            "stream, attached as a co-session child of the primary root"
        )
        assert demoted_conv.is_root is False
        _assert_tree_invariants(plan)


class TestMaxContextLength:
    @pytest.fixture
    def session_with_high_peak_subagent(self):
        main = build_instance(
            instance_id="g:main",
            role="main",
            requests=[
                build_request(
                    request_id=1, ts=_ts(0), input_tokens=100, output_tokens=50
                )
            ],
        )
        child = build_instance(
            instance_id="g:child",
            role="sub-agent",
            fan_out_group="1",
            parent_spawn_request_id=1,
            requests=[
                build_request(
                    request_id=2,
                    ts=_ts(100),
                    input_tokens=5000,
                    output_tokens=5000,
                )
            ],
        )
        return build_session(session_id="sess_g", instances=[main, child])

    def test_whole_session_dropped_when_peak_exceeds_limit(
        self, session_with_high_peak_subagent
    ):
        plan, diag = _plan(session_with_high_peak_subagent, max_context_length=1000)
        assert plan is None
        assert diag.dropped_for_max_context_length == 1

    def test_session_kept_when_peak_is_under_limit(
        self, session_with_high_peak_subagent
    ):
        plan, diag = _plan(session_with_high_peak_subagent, max_context_length=20000)
        assert plan is not None
        assert diag.dropped_for_max_context_length == 0


class TestMaxContextLengthSynthesisMaxOslAsymmetry:
    """``synthesis_max_osl`` caps the ROOT's peak-context output_tokens only
    (mirroring weka_trace._trace_peak_context_length's own asymmetry) -
    a subagent's huge output_tokens must still trigger the drop uncapped."""

    def test_root_huge_output_capped_below_threshold_is_kept(self):
        main = build_instance(
            instance_id="h:main",
            role="main",
            requests=[
                build_request(
                    request_id=1, ts=_ts(0), input_tokens=100, output_tokens=50_000
                )
            ],
        )
        session_dict = build_session(session_id="sess_h", instances=[main])
        plan, diag = _plan(session_dict, max_context_length=1000, synthesis_max_osl=500)
        assert plan is not None
        assert diag.dropped_for_max_context_length == 0

    def test_subagent_huge_output_not_capped_still_drops(self):
        main = build_instance(
            instance_id="i:main",
            role="main",
            requests=[
                build_request(
                    request_id=1, ts=_ts(0), input_tokens=100, output_tokens=50
                )
            ],
        )
        child = build_instance(
            instance_id="i:child",
            role="sub-agent",
            fan_out_group="1",
            parent_spawn_request_id=1,
            requests=[
                build_request(
                    request_id=2,
                    ts=_ts(100),
                    input_tokens=100,
                    output_tokens=50_000,
                )
            ],
        )
        session_dict = build_session(session_id="sess_i", instances=[main, child])
        plan, diag = _plan(session_dict, max_context_length=1000, synthesis_max_osl=500)
        assert plan is None
        assert diag.dropped_for_max_context_length == 1


class TestUnreachableInstanceBugGuard:
    """A real causal cycle can't survive the IR layer's own classification
    (time-inversion/self-spawn/group-mismatch all demote to a non-causal
    kind before a cycle could form), so this hand-builds IR dataclasses
    directly to exercise the defensive guard: every instance classified
    CAUSAL with no non-causal instance anywhere to serve as a root.
    """

    def test_no_non_causal_root_raises(self):
        sampling = SamplingIR(temperature=None, top_p=None, top_k=None, thinking=None)

        def _req(request_id: int, ts_ms: float) -> RequestIR:
            return RequestIR(
                request_id=request_id,
                ts_ms=ts_ms,
                model="m",
                stream=False,
                max_tokens_recorded=0,
                input_tokens=10,
                prefill_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                output_tokens=5,
                sampling=sampling,
                stop_reason=None,
                api_time_ms=None,
                upstream_latency_ms=None,
                think_ms=None,
                think_ref=None,
                system=(),
                tools=None,
                messages=(),
            )

        req_a = _req(10, 0.0)
        req_b = _req(20, 100.0)
        inst_a = InstanceIR(
            instance_id="x:a",
            role=ReplayRole.SUB_AGENT,
            raw_role="sub-agent",
            link=LinkIR(LinkKind.CAUSAL, 20, "20", 1, 0, "ok"),
            requests=(req_a,),
            first_ts_ms=0.0,
            last_end_ms=0.0,
        )
        inst_b = InstanceIR(
            instance_id="x:b",
            role=ReplayRole.SUB_AGENT,
            raw_role="sub-agent",
            link=LinkIR(LinkKind.CAUSAL, 10, "10", 1, 0, "ok"),
            requests=(req_b,),
            first_ts_ms=100.0,
            last_end_ms=100.0,
        )
        session = SessionIR(
            session_id="sess_x",
            schema_version="replay-v3",
            start_ts_ms=0.0,
            orphan_shape="none",
            instances=(inst_a, inst_b),
            request_owner={10: (0, 0), 20: (1, 0)},
        )
        diag = SessionDiagnostics()
        with pytest.raises(NotImplementedError, match="unreachable from any root"):
            plan_session_tree(session, options=TreeFilterOptions(), diag=diag)
