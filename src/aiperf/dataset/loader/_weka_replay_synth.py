# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content synthesis for ``weka_replay`` (plan sections 5.2-5.5).

``ReplaySessionSynthesizer`` (satisfying ``SessionContentBuilder`` from
``_weka_replay_content.py``) turns the content ids resolved by
``_weka_replay_identity.py`` into actual synthetic text, assigns lengths to
newly minted ids from each request's exact ``input_tokens``, diffs each
request's wire against the previous one on its instance to decide between a
delta and a full-reset emission, and computes a theoretical prefix-cache-hit
metric across the whole session once every instance has been built.

Deliberately decoupled from AIPerf's tokenizer/``PromptGenerator`` machinery:
this module only depends on the small ``TokenCorpusSource`` protocol defined
here. A later integration step adapts the real ``PromptGenerator`` to it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from aiperf.dataset.loader._weka_replay_content import TurnContent
from aiperf.dataset.loader._weka_replay_identity import (
    ChurnStats,
    Resolution,
    SessionContentIdentity,
    wire_blocks,
)
from aiperf.dataset.loader._weka_replay_ir import (
    BlockKind,
    BlockSlotIR,
    InstanceIR,
    RequestIR,
    SessionIR,
)

# Metric block size used for the theoretical-cache-hit computation in
# `finish()`, matching the plan's own convention (plan section 5.4).
_METRIC_BLOCK_TOKENS = 64


class TokenCorpusSource(Protocol):
    """Minimal read-only view of a pre-tokenized prompt corpus.

    The real integration step adapts AIPerf's ``PromptGenerator`` to this
    protocol; tests here use a trivial deterministic fake.
    """

    def token_at(self, index: int) -> int:
        """Return the token id at absolute corpus position ``index``."""
        ...

    @property
    def size(self) -> int:
        """Total number of tokens in the corpus."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a sequence of token ids back to text."""
        ...


@dataclass(frozen=True, slots=True)
class ReplaySynthConfig:
    """Tunables for content synthesis. See plan sections 5.1-5.5."""

    # Verified against the real corpus (see _weka_replay_identity.py's
    # SessionContentIdentity default): 37 = `,"cache_control":{"type":
    # "ephemeral"}`, 48 = the same with a `"ttl"` field, 73 = a third,
    # empirically-confirmed variant (consistently exact, dozens of
    # occurrences, always role=user text-only messages going
    # "ephemeral" -> None) whose precise JSON shape wasn't reverse-engineered
    # but whose byte delta is unambiguous and deterministic.
    churn_cc_fragment_bytes: tuple[int, ...] = (37, 48, 73)
    residual_scale_bounds: tuple[float, float] = (0.25, 4.0)
    chunk_tokens: int = 256
    tools_placement: Literal["system_prefix", "raw_tools"] = "system_prefix"
    system_joiner: str = "\n\n"


@dataclass(slots=True)
class SynthDiagnostics:
    """Counters surfaced after synthesis, to correct plan section 6's memory
    estimate against reality and to validate the length-scaling heuristic."""

    requests_processed: int = 0
    minted_blocks: int = 0
    reused_blocks: int = 0
    length_scale_clamped_count: int = 0
    residual_token_count: int = 0
    chars_emitted: int = 0
    unique_content_tokens_emitted: int = 0


class ChunkedCorpusStream:
    """Materializes text for a ``(anchor, length)`` content id via
    independently-seeded fixed-size chunks (plan section 5.3).

    Chunk ``i``'s starting corpus offset is
    ``sha256(f"{base_seed}:{session_id}:{anchor}:{i}")`` truncated to its
    first 8 bytes, taken modulo the corpus size, so a corpus-wide accidental
    collision is confined to at most one chunk rather than an entire
    multi-thousand-token system prompt. Because a chunk's offset depends
    only on ``(anchor, chunk_index)`` and never on the requested length,
    ``text_for(anchor, n)`` is always a token-level prefix of
    ``text_for(anchor, n + k)``: both draw the same chunk sequence, the
    longer call just keeps going for more chunks.
    """

    def __init__(
        self,
        corpus: TokenCorpusSource,
        *,
        session_id: str,
        chunk_tokens: int,
        base_seed: str = "weka_replay",
    ) -> None:
        self._corpus = corpus
        self._session_id = session_id
        self._chunk_tokens = max(1, chunk_tokens)
        self._base_seed = base_seed
        self._cache: dict[tuple[str, int], str] = {}

    def text_for(self, anchor: str, length_tokens: int) -> str:
        key = (anchor, length_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        token_ids: list[int] = []
        remaining = length_tokens
        chunk_idx = 0
        size = self._corpus.size
        while remaining > 0:
            offset = self._chunk_offset(anchor, chunk_idx)
            take = min(self._chunk_tokens, remaining)
            token_ids.extend(
                self._corpus.token_at((offset + j) % size) for j in range(take)
            )
            remaining -= take
            chunk_idx += 1

        text = self._corpus.decode(token_ids) if token_ids else ""
        self._cache[key] = text
        return text

    def _chunk_offset(self, anchor: str, chunk_idx: int) -> int:
        seed = f"{self._base_seed}:{self._session_id}:{anchor}:{chunk_idx}".encode()
        digest = hashlib.sha256(seed).digest()
        return int.from_bytes(digest[:8], "big") % self._corpus.size


@dataclass(slots=True)
class _RenderItem:
    """One block's rendered text, carried alongside its kind/role/cid so a
    (possibly partial, delta-only) slice of a request's wire can still be
    turned into the right message dicts."""

    kind: BlockKind
    role: str
    text: str


def _render_wire(items: Sequence[_RenderItem], *, joiner: str) -> list[dict[str, Any]]:
    """Render a (possibly partial) wire slice to OpenAI-chat-shaped dicts.

    Any leading run of non-MESSAGE items (tools/system) is folded into one
    combined ``{"role": "system", ...}`` message, joined by ``joiner`` —
    this applies uniformly whether rendering a full wire (where tools/system
    are always the leading items) or a delta tail that happens to still
    start with one (rare, but handled rather than assumed away).
    """
    i = 0
    system_parts: list[str] = []
    while i < len(items) and items[i].kind != BlockKind.MESSAGE:
        system_parts.append(items[i].text)
        i += 1

    rendered: list[dict[str, Any]] = []
    if system_parts:
        rendered.append({"role": "system", "content": joiner.join(system_parts)})
    for item in items[i:]:
        rendered.append({"role": item.role, "content": item.text})
    return rendered


class ReplaySessionSynthesizer:
    """Synthesizes content for every retained instance of one session.

    Satisfies ``SessionContentBuilder`` structurally (see
    ``_weka_replay_content.py``).
    """

    def __init__(
        self,
        session: SessionIR,
        *,
        corpus: TokenCorpusSource,
        cfg: ReplaySynthConfig,
    ) -> None:
        self._session = session
        self._cfg = cfg
        self._identity = SessionContentIdentity(
            cc_fragment_bytes=cfg.churn_cc_fragment_bytes
        )
        self._corpus_stream = ChunkedCorpusStream(
            corpus, session_id=session.session_id, chunk_tokens=cfg.chunk_tokens
        )
        self._diag = SynthDiagnostics()
        self._seen_content_ids: set[tuple[str, int]] = set()
        # (ts_ms, instance_id, request_id, wire_cids) for every turn built so
        # far, across every instance of this session; consumed by finish().
        self._wire_records: list[tuple[float, str, int, tuple[int, ...]]] = []

    def build_instance_turns(self, instance: InstanceIR) -> list[TurnContent]:
        turns: list[TurnContent] = []
        prev_wire_cids: tuple[int, ...] | None = None

        for request in instance.requests:
            blocks = wire_blocks(request)
            resolutions = self._identity.resolve_request(instance.instance_id, request)
            self._assign_minted_lengths(request, blocks, resolutions)

            # Cheap (no text materialization): lets `is_extension` below be
            # known before deciding which blocks actually need a rendered
            # `_RenderItem` -- the matched leading prefix never reaches
            # `raw_messages` either way (it's sliced off below exactly as
            # before), so building its render items is pure waste on every
            # non-reset turn of a long-running instance.
            wire_cids_t = tuple(cid for cid, _ in resolutions)
            # Strictly longer, not just equal: an exact repeat (retry) has an
            # identical wire of the same length and must be a reset, not a
            # zero-length "extension" with an empty delta.
            is_extension = (
                prev_wire_cids is not None
                and len(wire_cids_t) > len(prev_wire_cids)
                and wire_cids_t[: len(prev_wire_cids)] == prev_wire_cids
            )
            # Turn 0 and a full reset both need every block's render item;
            # only a confirmed extension can skip the already-matched
            # leading prefix (it would just be sliced away below anyway).
            skip_prefix = len(prev_wire_cids) if is_extension else 0

            items: list[_RenderItem] = []
            for idx, (block, (cid, resolution)) in enumerate(
                zip(blocks, resolutions, strict=True)
            ):
                if resolution == Resolution.MINTED:
                    self._diag.minted_blocks += 1
                else:
                    self._diag.reused_blocks += 1

                # Unconditional regardless of `skip_prefix`: first-seen
                # bookkeeping must happen exactly once per content id no
                # matter which turn/position first introduces it, so this
                # diagnostic can never depend on whether this turn happens
                # to render that block's text.
                anchor, length = self._identity.anchor_and_length(cid)
                seen_key = (anchor, length)
                if seen_key not in self._seen_content_ids:
                    self._seen_content_ids.add(seen_key)
                    self._diag.chars_emitted += len(
                        self._corpus_stream.text_for(anchor, length)
                    )
                    self._diag.unique_content_tokens_emitted += length

                if idx < skip_prefix:
                    continue

                text = self._corpus_stream.text_for(anchor, length)
                role = block.role if block.kind == BlockKind.MESSAGE else "system"
                items.append(_RenderItem(kind=block.kind, role=role, text=text))

            # `items` already holds exactly the suffix past `skip_prefix`
            # (nothing before it was ever appended), so no further slicing is
            # needed here -- for turn 0 / a reset, skip_prefix is 0 and
            # `items` covers the full wire, identically to before.
            reset = not (prev_wire_cids is None or is_extension)
            raw_messages = _render_wire(items, joiner=self._cfg.system_joiner)

            turns.append(
                TurnContent(
                    raw_messages=raw_messages, raw_tools=None, reset_context=reset
                )
            )
            self._wire_records.append(
                (request.ts_ms, instance.instance_id, request.request_id, wire_cids_t)
            )
            self._diag.requests_processed += 1
            prev_wire_cids = wire_cids_t

        return turns

    def _assign_minted_lengths(
        self,
        request: RequestIR,
        blocks: list[BlockSlotIR],
        resolutions: list[tuple[int, Resolution]],
    ) -> None:
        """Split ``request.input_tokens``, minus reused blocks' frozen
        lengths, across newly minted blocks in proportion to their
        (unreliable) ``tokens_est`` — plan section 5.2."""
        minted_idx = [
            i
            for i, (_, resolution) in enumerate(resolutions)
            if resolution == Resolution.MINTED
        ]
        if not minted_idx:
            return

        frozen_reused_sum = sum(
            self._identity.anchor_and_length(cid)[1]
            for cid, resolution in resolutions
            if resolution != Resolution.MINTED
        )
        remaining = max(0, request.input_tokens - frozen_reused_sum)
        raw_total = sum(blocks[i].tokens_est for i in minted_idx)

        if raw_total == 0:
            for i in minted_idx:
                cid, _ = resolutions[i]
                self._identity.set_length(cid, 1)
            self._diag.residual_token_count += remaining
            return

        lo, hi = self._cfg.residual_scale_bounds
        raw_scale = remaining / raw_total
        scale = max(lo, min(hi, raw_scale))
        clamped = scale != raw_scale
        if clamped:
            self._diag.length_scale_clamped_count += 1

        assigned = [max(1, round(blocks[i].tokens_est * scale)) for i in minted_idx]
        if not clamped:
            # Largest-remainder adjustment on the last minted block so the
            # assigned lengths sum to exactly `remaining`.
            diff = remaining - sum(assigned)
            assigned[-1] = max(1, assigned[-1] + diff)
        else:
            self._diag.residual_token_count += abs(remaining - sum(assigned))

        for i, length in zip(minted_idx, assigned, strict=True):
            cid, _ = resolutions[i]
            self._identity.set_length(cid, length)

    def finish(self) -> dict[tuple[str, int], tuple[int, int]]:
        """Theoretical prefix-cache-hit metric, combined across every
        instance of this session in global timestamp order (plan section
        5.4).

        For each request (processed in ``ts_ms`` order across all
        instances), the "hit" portion is the longest common prefix between
        its own resolved wire and any wire already fully emitted so far.
        Computed via an incremental prefix trie (plain dict-of-dicts) over
        resolved content-id sequences, scoped to this one session and
        discarded along with this synthesizer once the session is done:
        walking the trie from the root along ``wire``'s own content ids, for
        as long as a matching child exists, reaches exactly the longest
        prefix of ``wire`` that is also a prefix of some previously-inserted
        wire — which is the same quantity the original pairwise linear scan
        over every earlier wire computed, just without re-scanning each one
        (O(len(wire)) per request instead of O(requests × len(wire))). The
        wire is then inserted into the trie (only the unseen suffix past the
        matched depth needs new nodes) before moving to the next request, so
        a wire never matches against itself.
        """
        records = sorted(self._wire_records, key=lambda record: record[0])
        trie_root: dict[int, dict] = {}
        result: dict[tuple[str, int], tuple[int, int]] = {}

        for _ts_ms, instance_id, request_id, wire in records:
            node = trie_root
            best_hit = 0
            for cid in wire:
                child = node.get(cid)
                if child is None:
                    break
                node = child
                best_hit += 1

            for cid in wire[best_hit:]:
                node = node.setdefault(cid, {})

            hit_tokens = sum(
                self._identity.anchor_and_length(cid)[1] for cid in wire[:best_hit]
            )
            total_tokens = sum(self._identity.anchor_and_length(cid)[1] for cid in wire)
            hit_blocks = hit_tokens // _METRIC_BLOCK_TOKENS
            total_blocks = -(-total_tokens // _METRIC_BLOCK_TOKENS)
            result[(instance_id, request_id)] = (hit_blocks, total_blocks)

        return result

    @property
    def diagnostics(self) -> SynthDiagnostics:
        return self._diag

    @property
    def identity_stats(self) -> ChurnStats:
        """Exposes the underlying :class:`SessionContentIdentity`'s resolution
        counters and ``flip_byte_delta_hist`` -- needed by the loader to
        validate the churn heuristic's byte-fragment assumption against real
        data (plan section 5.1's "pending one real-corpus validation run")."""
        return self._identity.stats()


class ReplayContentBuilderImpl:
    """Factory for a per-session :class:`ReplaySessionSynthesizer`.

    Satisfies ``ReplayContentBuilder`` structurally (see
    ``_weka_replay_content.py``).
    """

    def __init__(
        self, corpus: TokenCorpusSource, *, cfg: ReplaySynthConfig | None = None
    ) -> None:
        self._corpus = corpus
        self._cfg = cfg or ReplaySynthConfig()

    def for_session(self, session: SessionIR) -> ReplaySessionSynthesizer:
        return ReplaySessionSynthesizer(session, corpus=self._corpus, cfg=self._cfg)
