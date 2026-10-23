# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session tree planning: filtering, root selection, and causal/background
branch construction over the IR produced by ``_weka_replay_ir.normalize_session``.

``plan_session_tree`` is a pure function - no AIPerf framework types, no I/O,
no dependency on anything but stdlib and ``_weka_replay_ir``/
``_weka_replay_timing``. See plan §4.2-§4.4 for the authoritative spec this
implements; two genuine edge cases not fully specified there
(``--synthesis-max-isl``'s interaction with a dropped causal parent, and
``--max-context-length``'s peak-context definition) are resolved here with
documented, defensible choices - see the comments at their call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from aiperf.dataset.loader._weka_replay_ir import (
    InstanceIR,
    LinkKind,
    ReplayRole,
    SessionDiagnostics,
    SessionIR,
)
from aiperf.dataset.loader._weka_replay_timing import compute_branch_start_ms

# Plan §4.4: a child whose last response ends at or before the parent's next
# turn begins (within this tolerance) gates that next turn; floating-point
# timestamp arithmetic can land a hair past the boundary even when the real
# event ordering was synchronous.
_SYNC_TOLERANCE_MS = 1.0

_EXAMPLE_CAP = 5


def _cap_example(examples: list[str], item: str) -> None:
    if len(examples) < _EXAMPLE_CAP:
        examples.append(item)


@dataclass(frozen=True, slots=True)
class TreeFilterOptions:
    """Filtering/mode knobs needed by :func:`plan_session_tree`.

    A later integration step wires CLI/env config into an instance of this;
    this module never reads AIPerf config/env itself, so it (and its tests)
    stay fully self-contained.
    """

    session_mode: Literal["tree", "independent"] = "tree"
    include_orphans: bool = False
    include_ephemeral: bool = False
    include_roles: frozenset[ReplayRole] | None = None  # None = all roles allowed
    synthesis_max_isl: int | None = None  # --synthesis-max-isl; None = no limit
    max_context_length: int | None = None  # --max-context-length; None = no limit
    synthesis_max_osl: int | None = None  # --synthesis-max-osl; None = no limit


@dataclass(frozen=True, slots=True)
class PlannedBranch:
    """One SPAWN branch: a group of children attached at a single parent turn."""

    branch_id: str
    declaring_turn: int  # turn index on the parent conversation
    child_cids: tuple[str, ...]
    is_background: bool
    join_turn: int | None  # turn index on the parent conversation gated by this branch
    start_timestamp_ms: float | None


@dataclass(frozen=True, slots=True)
class PlannedConversation:
    """One conversation (root or spawn/co-session child) in the planned tree."""

    cid: str  # == instance_id verbatim
    instance: InstanceIR
    is_root: bool
    agent_depth: int
    parent_cid: str | None
    branches: tuple[PlannedBranch, ...]


@dataclass(frozen=True, slots=True)
class SessionTreePlan:
    """The full planned tree for one retained session."""

    session: SessionIR
    conversations: tuple[
        PlannedConversation, ...
    ]  # topological: parent before children


def _anchor_turn(root: InstanceIR, stream_first_ts_ms: float) -> int:
    """Largest turn index ``k`` on ``root`` whose request starts at or before
    ``stream_first_ts_ms``; 0 if the stream started before root's very first
    request. ``root.requests`` is already sorted ascending by ``ts_ms`` (the
    IR layer's own invariant), so a linear scan that stops at the first
    request starting after the stream is both correct and cheap.
    """
    k = 0
    for idx, req in enumerate(root.requests):
        if req.ts_ms <= stream_first_ts_ms:
            k = idx
        else:
            break
    return k


def _process_instance(
    inst: InstanceIR,
    *,
    is_root: bool,
    agent_depth: int,
    parent_cid: str | None,
    final_map: dict[str, InstanceIR],
    effective_children_by_request_id: dict[int, list[str]],
    visited: set[str],
    out: list[PlannedConversation],
    extra_branches: tuple[PlannedBranch, ...] = (),
) -> None:
    """Recursively place ``inst`` and every instance causally reachable from
    it (at any depth) into ``out``, in pre-order (parent before children).
    """
    visited.add(inst.instance_id)
    branches: list[PlannedBranch] = list(extra_branches)
    recurse_children: list[InstanceIR] = []

    for s, req in enumerate(inst.requests):
        child_ids = effective_children_by_request_id.get(req.request_id)
        if not child_ids:
            continue
        children = sorted(
            (final_map[cid] for cid in child_ids),
            key=lambda c: (c.link.fan_out_position, c.first_ts_ms, c.instance_id),
        )
        recurse_children.extend(children)

        t_next = inst.requests[s + 1].ts_ms if s + 1 < len(inst.requests) else None
        sync_children: list[InstanceIR] = []
        async_children: list[InstanceIR] = []
        for child in children:
            if t_next is not None and child.last_end_ms <= t_next + _SYNC_TOLERANCE_MS:
                sync_children.append(child)
            else:
                async_children.append(child)

        if sync_children:
            branches.append(
                PlannedBranch(
                    branch_id=f"{inst.instance_id}:{s}:spawn",
                    declaring_turn=s,
                    child_cids=tuple(c.instance_id for c in sync_children),
                    is_background=False,
                    join_turn=s + 1,
                    start_timestamp_ms=compute_branch_start_ms(
                        req.ts_ms,
                        req.api_time_ms,
                        [c.first_ts_ms for c in sync_children],
                    ),
                )
            )
        if async_children:
            branches.append(
                PlannedBranch(
                    branch_id=f"{inst.instance_id}:{s}:spawn_bg",
                    declaring_turn=s,
                    child_cids=tuple(c.instance_id for c in async_children),
                    is_background=True,
                    join_turn=None,
                    start_timestamp_ms=compute_branch_start_ms(
                        req.ts_ms,
                        req.api_time_ms,
                        [c.first_ts_ms for c in async_children],
                    ),
                )
            )

    branches.sort(key=lambda b: (b.declaring_turn, b.branch_id))
    out.append(
        PlannedConversation(
            cid=inst.instance_id,
            instance=inst,
            is_root=is_root,
            agent_depth=agent_depth,
            parent_cid=parent_cid,
            branches=tuple(branches),
        )
    )
    for child in recurse_children:
        _process_instance(
            child,
            is_root=False,
            agent_depth=agent_depth + 1,
            parent_cid=inst.instance_id,
            final_map=final_map,
            effective_children_by_request_id=effective_children_by_request_id,
            visited=visited,
            out=out,
        )


def plan_session_tree(
    session: SessionIR, *, options: TreeFilterOptions, diag: SessionDiagnostics
) -> SessionTreePlan | None:
    """Plan one session's conversation tree, or return ``None`` if the whole
    session is dropped by filtering. See the module docstring and plan
    §4.2-§4.4 for the full spec.
    """
    # --- (a) session-level drop ---
    if session.orphan_shape != "none" and not options.include_orphans:
        diag.orphan_session_dropped += 1
        return None

    # The shared orphan bucket's implicit links are meaningless chronology
    # across unrelated users (plan §4.3's closing paragraph) - it always uses
    # independent mode internally, regardless of the caller's session_mode,
    # whenever it's included at all. A per-record orphan session has no such
    # problem (each line really is one session) and follows session_mode
    # normally.
    forced_independent = (
        session.orphan_shape == "shared_bucket" and options.include_orphans
    )
    effective_session_mode: Literal["tree", "independent"] = (
        "independent" if forced_independent else options.session_mode
    )

    # --- (b) instance-level filtering: role/ephemeral drop ---
    dropped_ids: set[str] = set()
    for inst in session.instances:
        if inst.role == ReplayRole.EPHEMERAL and not options.include_ephemeral:
            dropped_ids.add(inst.instance_id)
            diag.ephemeral_dropped_count += 1
        elif (
            options.include_roles is not None and inst.role not in options.include_roles
        ):
            dropped_ids.add(inst.instance_id)
            diag.role_filtered_dropped_count += 1

    # Cascade: an instance causally descended (via a *verified* CAUSAL link,
    # never IMPLICIT/NONE chronology) from a dropped instance loses its real
    # parent context and must be dropped too. Built from the ORIGINAL link
    # classification, before any --synthesis-max-isl demotion below, since
    # cascade-drop is a different (and stronger) rule than that demotion -
    # see the comment at the max_isl loop for why the two are deliberately
    # different.
    original_causal_children_by_request_id: dict[int, list[str]] = {}
    for inst in session.instances:
        if inst.link.kind == LinkKind.CAUSAL:
            assert inst.link.parent_request_id is not None
            original_causal_children_by_request_id.setdefault(
                inst.link.parent_request_id, []
            ).append(inst.instance_id)

    request_ids_by_instance: dict[str, set[int]] = {
        inst.instance_id: {r.request_id for r in inst.requests}
        for inst in session.instances
    }

    cascade_queue = list(dropped_ids)
    while cascade_queue:
        cur = cascade_queue.pop()
        for rid in request_ids_by_instance[cur]:
            for child_id in original_causal_children_by_request_id.get(rid, ()):
                if child_id not in dropped_ids:
                    dropped_ids.add(child_id)
                    diag.cascade_dropped_count += 1
                    _cap_example(diag.cascade_dropped_examples, child_id)
                    cascade_queue.append(child_id)

    # --- (b) --synthesis-max-isl: drop individual over-length requests ---
    #
    # Judgment call (not specified verbatim by the design doc): when a
    # dropped request was some other instance's verified CAUSAL parent, that
    # child is DEMOTED to a non-causal top-level stream for this planning
    # pass, not cascade-dropped. This is deliberately different from the
    # role/ephemeral cascade-drop above: losing one oversized anchor request
    # says nothing about whether the child's own content is worth keeping -
    # the child's data is still real and complete, it just loses the one
    # fact that tied it to this specific parent turn. Dropping the whole
    # subtree would destroy real, retained data for a synthesis-sizing
    # reason, which is a much weaker justification than a policy exclusion
    # (ephemeral/role) propagating down a tree whose root was deliberately
    # excluded. This mirrors the IR layer's own precedent for a
    # zero-requests drop (see _weka_replay_ir.py's module docstring): that
    # drop doesn't cascade either, it just leaves descendants to fall back to
    # ANOMALOUS ("unowned") classification - demotion-not-drop is the same
    # idea applied one layer up.
    demoted_to_none: set[str] = set()
    final_map: dict[str, InstanceIR] = {}
    for inst in session.instances:
        if inst.instance_id in dropped_ids:
            continue
        if options.synthesis_max_isl is None:
            final_map[inst.instance_id] = inst
            continue

        kept = tuple(
            r for r in inst.requests if r.input_tokens <= options.synthesis_max_isl
        )
        removed = [
            r for r in inst.requests if r.input_tokens > options.synthesis_max_isl
        ]

        if removed:
            for r in removed:
                for child_id in original_causal_children_by_request_id.get(
                    r.request_id, ()
                ):
                    if child_id in dropped_ids or child_id in demoted_to_none:
                        continue
                    demoted_to_none.add(child_id)
                    diag.causal_parent_dropped_by_max_isl_count += 1
                    _cap_example(
                        diag.causal_parent_dropped_by_max_isl_examples, child_id
                    )

        if not kept:
            # Emptied entirely by the filter - drop it like any other
            # zero-request instance (counted separately, per spec, from the
            # IR layer's own "zero requests on disk" diagnostic).
            dropped_ids.add(inst.instance_id)
            diag.dropped_for_max_isl_count += 1
            _cap_example(diag.dropped_for_max_isl_examples, inst.instance_id)
            continue

        if removed:
            last_kept = kept[-1]
            inst = replace(
                inst,
                requests=kept,
                first_ts_ms=kept[0].ts_ms,
                last_end_ms=last_kept.ts_ms + (last_kept.api_time_ms or 0.0),
            )
        final_map[inst.instance_id] = inst

    if not final_map:
        diag.empty_session_after_filtering += 1
        return None

    # --- (c)/(d) root selection, co-session attachment, and causal edges ---
    #
    # A CAUSAL instance that got demoted above is treated as non-causal for
    # THIS pass only - final_map/InstanceIR themselves are untouched.
    effective_children_by_request_id: dict[int, list[str]] = {}
    non_causal: list[InstanceIR] = []
    for inst in final_map.values():
        is_causal = (
            inst.link.kind == LinkKind.CAUSAL
            and inst.instance_id not in demoted_to_none
        )
        if is_causal:
            assert inst.link.parent_request_id is not None
            effective_children_by_request_id.setdefault(
                inst.link.parent_request_id, []
            ).append(inst.instance_id)
        else:
            non_causal.append(inst)

    out: list[PlannedConversation] = []
    visited: set[str] = set()

    if effective_session_mode == "independent":
        roots = sorted(non_causal, key=lambda i: (i.first_ts_ms, i.instance_id))
        for root in roots:
            _process_instance(
                root,
                is_root=True,
                agent_depth=0,
                parent_cid=None,
                final_map=final_map,
                effective_children_by_request_id=effective_children_by_request_id,
                visited=visited,
                out=out,
            )
    elif non_causal:
        mains = [i for i in non_causal if i.role == ReplayRole.MAIN]
        if mains:
            primary = min(mains, key=lambda i: (i.first_ts_ms, i.instance_id))
        else:
            primary = min(non_causal, key=lambda i: (i.first_ts_ms, i.instance_id))
            diag.root_fallback_used_count += 1

        other_streams = [i for i in non_causal if i.instance_id != primary.instance_id]
        groups: dict[int, list[InstanceIR]] = {}
        for stream in other_streams:
            k = _anchor_turn(primary, stream.first_ts_ms)
            groups.setdefault(k, []).append(stream)

        cosession_branches: list[PlannedBranch] = []
        stream_order: list[InstanceIR] = []
        for k in sorted(groups):
            streams = sorted(groups[k], key=lambda s: (s.first_ts_ms, s.instance_id))
            stream_order.extend(streams)
            anchor_req = primary.requests[k]
            cosession_branches.append(
                PlannedBranch(
                    branch_id=f"{primary.instance_id}:{k}:cosession",
                    declaring_turn=k,
                    child_cids=tuple(s.instance_id for s in streams),
                    is_background=True,
                    join_turn=None,
                    start_timestamp_ms=compute_branch_start_ms(
                        anchor_req.ts_ms,
                        anchor_req.api_time_ms,
                        [s.first_ts_ms for s in streams],
                    ),
                )
            )

        _process_instance(
            primary,
            is_root=True,
            agent_depth=0,
            parent_cid=None,
            final_map=final_map,
            effective_children_by_request_id=effective_children_by_request_id,
            visited=visited,
            out=out,
            extra_branches=tuple(cosession_branches),
        )
        for stream in stream_order:
            _process_instance(
                stream,
                is_root=False,
                agent_depth=1,
                parent_cid=primary.instance_id,
                final_map=final_map,
                effective_children_by_request_id=effective_children_by_request_id,
                visited=visited,
                out=out,
            )

    # Bug-guard: every retained instance must be reachable from exactly one
    # root. This should be unreachable given the IR layer already demotes
    # incoherent links to ANOMALOUS/CYCLE_BROKEN and time-inversion breaks
    # any real cycle, but a structural impossibility here is a genuine bug,
    # not an ordinary data-variation case, so it raises rather than being
    # silently dropped.
    unvisited = set(final_map) - visited
    if unvisited:
        raise NotImplementedError(
            f"session '{session.session_id}' instance '{sorted(unvisited)[0]}': "
            f"unreachable from any root after tree planning "
            f"({len(unvisited)} instance(s) affected) - likely a causal link "
            "cycle not caught by IR-layer link classification"
        )

    # --- (e) --max-context-length: whole-session drop ---
    #
    # Judgment call (not specified verbatim by the design doc): this mirrors
    # weka_trace._trace_peak_context_length's actual semantics as read
    # directly from src/aiperf/dataset/loader/weka_trace.py - peak SINGLE-
    # REQUEST (input_tokens + output_tokens), maxed across every request in
    # every conversation of the causal tree (root, sync/async spawn children,
    # and co-session streams alike), not a cumulative running total across a
    # conversation. This matches vLLM's own admission check (prompt tokens +
    # requested output tokens against the model's context window on each
    # individual request), which is what the flag is actually guarding
    # against. Mirrors weka_trace._trace_peak_context_length's own asymmetry:
    # --synthesis-max-osl capping is applied only to the ROOT conversation's
    # own requests (conv.agent_depth == 0) - subagent/co-session children's
    # output_tokens are deliberately left uncapped, matching weka_trace's own
    # "subagent child turns emit the RECORDED output_length" rule (see
    # weka_trace.py: child_req.input_length + _peak_output_tokens(
    # child_req.output_length, max_osl=None) for subagents, vs max_osl=max_osl
    # for the parent/flat chain). Without this, --max-context-length would
    # over-drop sessions whose root peak only exceeds the limit because of an
    # output_tokens value the loader will actually clamp at Turn-building time.
    if options.max_context_length is not None:
        peak = 0
        for conv in out:
            for req in conv.instance.requests:
                output_tokens = req.output_tokens
                if conv.agent_depth == 0 and options.synthesis_max_osl is not None:
                    output_tokens = min(output_tokens, options.synthesis_max_osl)
                total = req.input_tokens + output_tokens
                if total > peak:
                    peak = total
        if peak > options.max_context_length:
            diag.dropped_for_max_context_length += 1
            return None

    return SessionTreePlan(session=session, conversations=tuple(out))
