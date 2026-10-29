# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-identity resolution for ``weka_replay`` (plan section 5.1).

Replay-v3/v4/v5 never records real text, only a ``hash`` (which depends on
``cache_control`` and so changes almost every turn even for unchanged real
content — the "churn problem"), a byte count, and an approximate token
count per content block. This module resolves each block to a stable
content id (``cid``) across a session's requests, using four cheaper-to-
more-expensive identity signals in order: the previous request's own hash
sequence (fast path), a session-wide exact-hash match, a ``seed_hash`` match
(user text-only messages only), and a same-slot ``cache_control``-flip
("churn") match. Anything left over is newly minted.

Pure logic: only stdlib plus the sibling IR dataclasses
(:mod:`aiperf.dataset.loader._weka_replay_ir`). No tokenizer, no AIPerf
framework types — see ``_weka_replay_synth.py`` for the layer that turns a
resolved ``cid`` into actual text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from aiperf.dataset.loader._weka_replay_ir import (
    BlockKind,
    BlockSlotIR,
    InstanceIR,
    RequestIR,
)

# Upper bound on how many stacked cache_control-fragment-sized jumps a single
# churn flip is allowed to represent (k in `abs(delta) == k * fragment`).
# Real captures are expected to flip by one, occasionally two, fragments per
# turn; 8 is a generous bound chosen to tolerate multiple reminder/wrapper
# insertions without being large enough to let an unrelated byte delta pass
# by coincidence (see plan section 5.1 "tight check").
_MAX_CHURN_FRAGMENT_MULTIPLE = 8


class Resolution(IntEnum):
    """How a content block's id was determined, cheapest signal first."""

    SLOT_SAME_HASH = 0
    """Same hash as the immediately preceding occurrence (fast path)."""
    EXACT_HASH = 1
    """Session-wide exact hash match (not the immediate predecessor)."""
    SEED_HASH = 2
    CHURN = 3
    MINTED = 4


@dataclass(slots=True)
class ChurnStats:
    """Plain counters and validation histograms, surfaced to the caller.

    ``flip_byte_delta_hist`` and ``tight_loose_disagreement_count`` are
    computed in shadow for every observed ``cache_control`` presence flip,
    whether or not it ended up matching — they never gate behavior, they
    only validate the tight (fragment-size) gate against real data per plan
    section 5.1.
    """

    slot_same_hash_count: int = 0
    exact_hash_count: int = 0
    seed_hash_count: int = 0
    churn_count: int = 0
    minted_count: int = 0
    flip_byte_delta_hist: dict[int, int] = field(default_factory=dict)
    tight_loose_disagreement_count: int = 0


@dataclass(slots=True)
class _SlotState:
    """Per-instance, per-(kind, slot_index) state used for churn matching."""

    hash: str
    cid: int
    role: str
    cache_control: str | None
    byte_count: int


def wire_blocks(request: RequestIR) -> list[BlockSlotIR]:
    """Return ``request``'s content blocks in wire order: tools (if present),
    then system blocks, then messages — per plan section 5.1.

    This is deliberately NOT the same order
    :mod:`aiperf.dataset.loader._weka_replay_ir` uses internally for its own
    token-estimate proportional-fill bookkeeping (system, then tools, then
    messages in ``_collect_raw_slots``) — that order only matters there for
    filling missing per-block token estimates; wire order is a property of
    how a request is actually sent to a server and is defined independently
    here and in ``_weka_replay_synth.py``, which both import this helper so
    the two layers can never disagree on it.
    """
    blocks: list[BlockSlotIR] = []
    if request.tools is not None:
        blocks.append(request.tools)
    blocks.extend(request.system)
    blocks.extend(request.messages)
    return blocks


def _seed_hash_candidates(
    seed_hash: str,
    byte_count: int,
    cache_control: str | None,
    fragments: tuple[int, ...],
) -> list[tuple[str, int]]:
    """Candidate ``(seed_hash, normalized_bytes)`` keys for a block.

    Used identically for both storing and looking up, so that whichever side
    of a repeated-user-turn pair happens to carry ``cache_control`` (and
    therefore extra bytes), the other side's unadjusted byte count still has
    a chance to land on the same normalized key (plan section 5.1).
    """
    if cache_control is not None:
        return [(seed_hash, byte_count - fragment) for fragment in fragments]
    return [(seed_hash, byte_count)]


def _is_tight_match(delta: int, fragments: tuple[int, ...]) -> bool:
    magnitude = abs(delta)
    for fragment in fragments:
        if fragment <= 0 or magnitude % fragment != 0:
            continue
        multiple = magnitude // fragment
        if 1 <= multiple <= _MAX_CHURN_FRAGMENT_MULTIPLE:
            return True
    return False


class SessionContentIdentity:
    """Per-session identity resolver (one instance per session).

    Exact-hash and ``seed_hash`` identity maps are shared across every
    instance of the session (so a system prompt or tool spec shared between
    a parent and its subagent resolves to the same content id); fast-path
    and same-slot churn state are local to each instance. Call
    :meth:`resolve_request` (or the convenience wrapper
    :meth:`resolve_instance`) once per request, in chronological order
    within each instance; instances themselves may be resolved in any order
    relative to each other.
    """

    def __init__(self, *, cc_fragment_bytes: tuple[int, ...] = (37, 48, 73)) -> None:
        self._cc_fragment_bytes = cc_fragment_bytes
        self._next_cid = 0
        self._exact_hash: dict[str, int] = {}
        self._seed_hash: dict[tuple[str, int], int] = {}
        self._anchor: dict[int, str] = {}
        self._length: dict[int, int] = {}
        self._slot_state: dict[str, dict[tuple[BlockKind, int], _SlotState]] = {}
        self._prev_hashes: dict[str, tuple[str, ...]] = {}
        self._prev_cids: dict[str, tuple[int, ...]] = {}
        self._stats = ChurnStats()

    def resolve_instance(
        self, instance: InstanceIR
    ) -> list[list[tuple[int, Resolution]]]:
        """Resolve every request of ``instance``, in order.

        Convenience wrapper around :meth:`resolve_request` for callers (and
        tests) that don't need control over per-request state between calls.
        """
        return [
            self.resolve_request(instance.instance_id, request)
            for request in instance.requests
        ]

    def resolve_request(
        self, instance_id: str, request: RequestIR
    ) -> list[tuple[int, Resolution]]:
        """Resolve every content block of one request, in wire order.

        Newly minted blocks are frozen at ``block.tokens_est`` by default;
        callers that know a better length for a request's exact
        ``input_tokens`` split (``_weka_replay_synth.py``) should call
        :meth:`set_length` immediately afterward, before any text is
        materialized for that id.
        """
        blocks = wire_blocks(request)
        hashes = tuple(block.hash for block in blocks)
        prev_hashes = self._prev_hashes.get(instance_id, ())
        prev_cids = self._prev_cids.get(instance_id, ())

        prefix_len = 0
        for a, b in zip(hashes, prev_hashes, strict=False):
            if a != b:
                break
            prefix_len += 1

        slot_state = self._slot_state.setdefault(instance_id, {})
        results: list[tuple[int, Resolution]] = []

        for i, block in enumerate(blocks):
            if i < prefix_len:
                cid = prev_cids[i]
                resolution = Resolution.SLOT_SAME_HASH
                self._stats.slot_same_hash_count += 1
            else:
                cid, resolution = self._resolve_tail_block(instance_id, block)

            results.append((cid, resolution))

            # Always refresh exact-hash/seed-hash/slot state to the new
            # block's values, regardless of which path resolved it, so a
            # later request can match against the latest occurrence.
            slot_state[(block.kind, block.slot_index)] = _SlotState(
                hash=block.hash,
                cid=cid,
                role=block.role,
                cache_control=block.cache_control,
                byte_count=block.byte_count,
            )
            self._exact_hash[block.hash] = cid
            if block.seed_hash is not None:
                for key in _seed_hash_candidates(
                    block.seed_hash,
                    block.byte_count,
                    block.cache_control,
                    self._cc_fragment_bytes,
                ):
                    self._seed_hash[key] = cid

        self._prev_hashes[instance_id] = hashes
        self._prev_cids[instance_id] = tuple(cid for cid, _ in results)
        return results

    def _resolve_tail_block(
        self, instance_id: str, block: BlockSlotIR
    ) -> tuple[int, Resolution]:
        cid = self._exact_hash.get(block.hash)
        if cid is not None:
            self._stats.exact_hash_count += 1
            return cid, Resolution.EXACT_HASH

        if block.seed_hash is not None:
            for key in _seed_hash_candidates(
                block.seed_hash,
                block.byte_count,
                block.cache_control,
                self._cc_fragment_bytes,
            ):
                cid = self._seed_hash.get(key)
                if cid is not None:
                    self._stats.seed_hash_count += 1
                    return cid, Resolution.SEED_HASH

        prev_state = self._slot_state.get(instance_id, {}).get(
            (block.kind, block.slot_index)
        )
        if prev_state is not None and prev_state.role == block.role:
            cc_flipped = (prev_state.cache_control is None) != (
                block.cache_control is None
            )
            if cc_flipped:
                delta = block.byte_count - prev_state.byte_count
                self._stats.flip_byte_delta_hist[delta] = (
                    self._stats.flip_byte_delta_hist.get(delta, 0) + 1
                )
                tight = _is_tight_match(delta, self._cc_fragment_bytes)
                loose = abs(delta) / max(1, prev_state.byte_count) <= 0.10
                if tight != loose:
                    self._stats.tight_loose_disagreement_count += 1
                # Deliberately NOT gated on `prefix_intact` (see the
                # `_resolve_tail_block` docstring / plan correction): real
                # Claude Code captures carry an isolated, genuinely-volatile
                # early block (observed: a ~106-byte system block with no
                # cache_control, re-minted on essentially every request --
                # almost certainly a per-turn cwd/env/timestamp header). That
                # single, expected, unrelated MINT used to poison
                # `prefix_intact` for the rest of the request, which silently
                # zeroed out real churn matching on every session measured
                # (932,369 blocks, 0 churns, despite 1,500+ exact-fragment
                # byte deltas sitting right there in flip_byte_delta_hist).
                # role==role + a cache_control presence flip + an exact
                # cache_control-fragment-sized byte delta is already a
                # narrow, position-local signal; it doesn't need corroboration
                # from unrelated earlier positions in the same request.
                if tight:
                    self._stats.churn_count += 1
                    return prev_state.cid, Resolution.CHURN

        return self._mint(block), Resolution.MINTED

    def _mint(self, block: BlockSlotIR) -> int:
        cid = self._next_cid
        self._next_cid += 1
        self._anchor[cid] = block.hash
        self._length[cid] = block.tokens_est
        self._stats.minted_count += 1
        return cid

    def anchor_and_length(self, cid: int) -> tuple[str, int]:
        """Return ``(anchor_hash, frozen_length_tokens)`` for ``cid``.

        The anchor is the hash of whichever occurrence first minted this id;
        the length is frozen at that same moment (via :meth:`set_length` if
        the caller overrode it) and never resized on reuse.
        """
        return self._anchor[cid], self._length[cid]

    def set_length(self, cid: int, length: int) -> None:
        """Override a minted id's length (see :meth:`resolve_request`).

        Intended to be called at most once per id, immediately after it was
        minted and before any text is materialized for it — e.g.
        ``_weka_replay_synth.py`` replaces the raw ``tokens_est`` default
        with its own proportional-scale split of the request's exact
        ``input_tokens`` once it knows every minted block in that request.
        """
        self._length[cid] = length

    def stats(self) -> ChurnStats:
        return self._stats
