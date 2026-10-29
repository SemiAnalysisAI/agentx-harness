# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.dataset.loader._weka_replay_content import TurnContent
from aiperf.dataset.loader._weka_replay_identity import Resolution, wire_blocks
from aiperf.dataset.loader._weka_replay_ir import (
    BlockKind,
    BlockSlotIR,
    InstanceIR,
    LinkIR,
    LinkKind,
    ReplayRole,
    RequestIR,
    SamplingIR,
    SessionIR,
)
from aiperf.dataset.loader._weka_replay_synth import (
    _METRIC_BLOCK_TOKENS,
    ChunkedCorpusStream,
    ReplaySessionSynthesizer,
    ReplaySynthConfig,
    _render_wire,
    _RenderItem,
)

_SAMPLING = SamplingIR(temperature=None, top_p=None, top_k=None, thinking=None)
_LINK_NONE = LinkIR(
    kind=LinkKind.NONE,
    parent_request_id=None,
    fan_out_group=None,
    fan_out_size=None,
    fan_out_position=0,
    reason="ok",
)


class FakeCorpus:
    """Deterministic word-per-token fake corpus for tests.

    `token_at(i)` is just `i % size`, and `decode` renders each token id as
    its own space-separated word, so word count == token count and two
    different token-id sequences are virtually guaranteed to decode to
    different text.
    """

    def __init__(self, size: int = 100_000) -> None:
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    def token_at(self, index: int) -> int:
        return index % self._size

    def decode(self, token_ids) -> str:
        return " ".join(f"tok{t}" for t in token_ids)


def make_block(
    *,
    kind: BlockKind = BlockKind.MESSAGE,
    slot_index: int = 0,
    role: str = "user",
    hash: str = "h0",  # noqa: A002
    byte_count: int = 100,
    tokens_est: int = 10,
    cache_control: str | None = None,
    seed_hash: str | None = None,
) -> BlockSlotIR:
    return BlockSlotIR(
        kind=kind,
        slot_index=slot_index,
        role=role,
        hash=hash,
        byte_count=byte_count,
        tokens_est=tokens_est,
        cache_control=cache_control,
        block_types=(),
        tool_use_ids=(),
        tool_result_ids=(),
        seed_hash=seed_hash,
        tool_count=0,
    )


def make_request(
    *,
    request_id: int = 1,
    ts_ms: float = 0.0,
    input_tokens: int = 100,
    system: tuple[BlockSlotIR, ...] = (),
    tools: BlockSlotIR | None = None,
    messages: tuple[BlockSlotIR, ...] = (),
) -> RequestIR:
    return RequestIR(
        request_id=request_id,
        ts_ms=ts_ms,
        model="test-model",
        stream=False,
        max_tokens_recorded=0,
        input_tokens=input_tokens,
        prefill_tokens=input_tokens,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        output_tokens=10,
        sampling=_SAMPLING,
        stop_reason=None,
        api_time_ms=None,
        upstream_latency_ms=None,
        think_ms=None,
        think_ref=None,
        system=system,
        tools=tools,
        messages=messages,
    )


def make_instance(
    *, instance_id: str = "sess:sig:fp", requests: tuple[RequestIR, ...]
) -> InstanceIR:
    return InstanceIR(
        instance_id=instance_id,
        role=ReplayRole.MAIN,
        raw_role="main",
        link=_LINK_NONE,
        requests=requests,
        first_ts_ms=requests[0].ts_ms,
        last_end_ms=requests[-1].ts_ms,
    )


def make_session(
    *, session_id: str = "sess1", instances: tuple[InstanceIR, ...]
) -> SessionIR:
    return SessionIR(
        session_id=session_id,
        schema_version="replay-v3",
        start_ts_ms=0.0,
        orphan_shape="none",
        instances=instances,
        request_owner={},
    )


class TestChunkedCorpusStreamPrefixProperty:
    def test_text_for_is_prefix_as_length_grows(self):
        stream = ChunkedCorpusStream(FakeCorpus(), session_id="s1", chunk_tokens=4)
        lengths = [1, 3, 5, 13, 37, 64, 100]
        texts = [stream.text_for("anchor-a", n) for n in lengths]
        for shorter, longer in zip(texts, texts[1:], strict=False):
            assert longer.startswith(shorter)

    def test_different_anchors_produce_different_text(self):
        stream = ChunkedCorpusStream(FakeCorpus(), session_id="s1", chunk_tokens=4)
        assert stream.text_for("anchor-a", 10) != stream.text_for("anchor-b", 10)

    def test_same_anchor_different_sessions_produce_different_text(self):
        corpus = FakeCorpus()
        stream1 = ChunkedCorpusStream(corpus, session_id="s1", chunk_tokens=4)
        stream2 = ChunkedCorpusStream(corpus, session_id="s2", chunk_tokens=4)
        assert stream1.text_for("same-anchor", 20) != stream2.text_for(
            "same-anchor", 20
        )


class TestDeterminism:
    def test_building_same_session_twice_is_identical(self):
        def build():
            block = make_block(hash="m0", byte_count=100, tokens_est=20)
            req = make_request(
                request_id=0, ts_ms=0.0, input_tokens=20, messages=(block,)
            )
            instance = make_instance(requests=(req,))
            session = make_session(instances=(instance,))
            synth = ReplaySessionSynthesizer(
                session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
            )
            return synth.build_instance_turns(instance)

        assert build() == build()


class TestLengthAssignment:
    def test_unclamped_scale_sums_exactly_to_remaining(self):
        block_a = make_block(slot_index=0, hash="a", tokens_est=30)
        block_b = make_block(slot_index=1, hash="b", tokens_est=10)
        # remaining=20, raw_total=40 -> scale=0.5, within default bounds.
        req = make_request(
            request_id=0, ts_ms=0.0, input_tokens=20, messages=(block_a, block_b)
        )
        instance = make_instance(requests=(req,))
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )

        turns = synth.build_instance_turns(instance)

        lengths = [len(m["content"].split()) for m in turns[0].raw_messages]
        assert sum(lengths) == 20
        assert synth.diagnostics.length_scale_clamped_count == 0
        assert synth.diagnostics.residual_token_count == 0

    def test_clamped_scale_counts_drift(self):
        block_a = make_block(slot_index=0, hash="a", tokens_est=5)
        block_b = make_block(slot_index=1, hash="b", tokens_est=5)
        # remaining=1000, raw_total=10 -> raw_scale=100, clamped to 4.0.
        req = make_request(
            request_id=0, ts_ms=0.0, input_tokens=1000, messages=(block_a, block_b)
        )
        instance = make_instance(requests=(req,))
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )

        turns = synth.build_instance_turns(instance)

        lengths = [len(m["content"].split()) for m in turns[0].raw_messages]
        assert sum(lengths) == 40  # 5*4 + 5*4, not forced to equal 1000
        assert synth.diagnostics.length_scale_clamped_count == 1
        assert synth.diagnostics.residual_token_count == 1000 - 40


class TestDeltaAndReset:
    def test_append_only_extension_is_delta_with_reset_false(self):
        msg_a = make_block(slot_index=0, hash="a-same", tokens_est=5)
        req0 = make_request(request_id=0, ts_ms=0.0, input_tokens=5, messages=(msg_a,))

        msg_a_again = make_block(slot_index=0, hash="a-same", tokens_est=5)
        msg_b = make_block(slot_index=1, hash="b-new", tokens_est=7, role="assistant")
        req1 = make_request(
            request_id=1, ts_ms=1.0, input_tokens=12, messages=(msg_a_again, msg_b)
        )

        instance = make_instance(requests=(req0, req1))
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )

        turns = synth.build_instance_turns(instance)

        assert turns[0].reset_context is False
        assert turns[1].reset_context is False
        assert len(turns[1].raw_messages) == 1
        assert turns[1].raw_messages[0]["role"] == "assistant"

    def test_genuine_divergence_is_full_reset(self):
        msg0 = make_block(slot_index=0, hash="m0", tokens_est=5)
        req0 = make_request(request_id=0, ts_ms=0.0, input_tokens=5, messages=(msg0,))

        msg1 = make_block(slot_index=0, hash="m1", tokens_est=5, byte_count=999)
        req1 = make_request(request_id=1, ts_ms=1.0, input_tokens=5, messages=(msg1,))

        instance = make_instance(requests=(req0, req1))
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )

        turns = synth.build_instance_turns(instance)

        assert turns[1].reset_context is True
        assert len(turns[1].raw_messages) == 1

    def test_exact_repeat_retry_is_full_reset(self):
        msg = make_block(slot_index=0, hash="same-every-time", tokens_est=5)
        req0 = make_request(request_id=0, ts_ms=0.0, input_tokens=5, messages=(msg,))
        req1 = make_request(request_id=1, ts_ms=1.0, input_tokens=5, messages=(msg,))

        instance = make_instance(requests=(req0, req1))
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )

        turns = synth.build_instance_turns(instance)

        assert turns[1].reset_context is True
        assert len(turns[1].raw_messages) == 1

    def test_turn_zero_is_never_reset_and_carries_full_wire(self):
        msg = make_block(slot_index=0, hash="first", tokens_est=5)
        req0 = make_request(request_id=0, ts_ms=0.0, input_tokens=5, messages=(msg,))
        instance = make_instance(requests=(req0,))
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )

        turns = synth.build_instance_turns(instance)

        assert turns[0].reset_context is False
        assert len(turns[0].raw_messages) == 1


class TestTheoreticalCacheHitMetric:
    def test_pure_churn_fixture_is_near_full_hit(self):
        # Lengths must be an exact multiple of the metric's 64-token block
        # size (so a full hit never loses a block to floor/ceil rounding),
        # and there must be enough turns that the always-zero first turn
        # (nothing seen yet) doesn't swamp the ratio.
        specs = [
            (None if i % 2 == 0 else "eph", 100 if i % 2 == 0 else 137, f"h{i}")
            for i in range(10)
        ]
        requests = tuple(
            make_request(
                request_id=i,
                ts_ms=float(i),
                input_tokens=512,
                messages=(
                    make_block(cache_control=cc, byte_count=b, hash=h, tokens_est=512),
                ),
            )
            for i, (cc, b, h) in enumerate(specs)
        )
        instance = make_instance(requests=requests)
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )
        synth.build_instance_turns(instance)

        metrics = synth.finish()
        hit = sum(h for h, _ in metrics.values())
        total = sum(t for _, t in metrics.values())
        assert total > 0
        assert hit / total > 0.8

    def test_fresh_content_each_turn_is_near_zero_hit(self):
        requests = tuple(
            make_request(
                request_id=i,
                ts_ms=float(i),
                input_tokens=50,
                messages=(
                    make_block(byte_count=100 + i, hash=f"fresh{i}", tokens_est=50),
                ),
            )
            for i in range(5)
        )
        instance = make_instance(requests=requests)
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )
        synth.build_instance_turns(instance)

        metrics = synth.finish()
        hit = sum(h for h, _ in metrics.values())
        total = sum(t for _, t in metrics.values())
        assert total > 0
        assert hit / total < 0.2


class TestFinishTrieEquivalence:
    """Proves the incremental-trie ``finish()`` produces byte-identical
    ``(hit_blocks, total_blocks)`` results to the original O(n^2) pairwise
    linear scan it replaced, across several hand-built fixtures.

    ``_naive_finish`` is a deliberately dumb reimplementation of the
    algorithm ``finish()`` used before the trie rewrite — the same
    semantics (longest common prefix against any earlier wire, scanned
    linearly), not exported, used only as this test's reference oracle.
    """

    @staticmethod
    def _naive_finish(
        records: list[tuple[float, str, int, tuple[int, ...]]],
        identity,
    ) -> dict[tuple[str, int], tuple[int, int]]:
        ordered = sorted(records, key=lambda record: record[0])
        seen_wires: list[tuple[int, ...]] = []
        result: dict[tuple[str, int], tuple[int, int]] = {}

        for _ts_ms, instance_id, request_id, wire in ordered:
            best_hit = 0
            for prior in seen_wires:
                common = 0
                for a, b in zip(prior, wire, strict=False):
                    if a != b:
                        break
                    common += 1
                if common > best_hit:
                    best_hit = common
                if best_hit == len(wire):
                    break

            hit_tokens = sum(
                identity.anchor_and_length(cid)[1] for cid in wire[:best_hit]
            )
            total_tokens = sum(identity.anchor_and_length(cid)[1] for cid in wire)
            hit_blocks = hit_tokens // _METRIC_BLOCK_TOKENS
            total_blocks = -(-total_tokens // _METRIC_BLOCK_TOKENS)
            result[(instance_id, request_id)] = (hit_blocks, total_blocks)
            seen_wires.append(wire)

        return result

    def _assert_matches_naive(
        self, synth: ReplaySessionSynthesizer
    ) -> dict[tuple[str, int], tuple[int, int]]:
        expected = self._naive_finish(synth._wire_records, synth._identity)
        actual = synth.finish()
        assert actual == expected
        return actual

    def test_single_instance_growing_wires_no_resets(self):
        """(a) one instance, strictly growing wires, no resets."""

        def messages(n: int) -> tuple[BlockSlotIR, ...]:
            return tuple(
                make_block(slot_index=i, hash=f"m{i}", tokens_est=64) for i in range(n)
            )

        requests = tuple(
            make_request(
                request_id=i,
                ts_ms=float(i),
                input_tokens=64 * (i + 1),
                messages=messages(i + 1),
            )
            for i in range(5)
        )
        instance = make_instance(requests=requests)
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )
        synth.build_instance_turns(instance)

        self._assert_matches_naive(synth)

    def test_two_sibling_instances_share_prefix_then_diverge(self):
        """(b) two sibling instances sharing a system/tools prefix, then
        diverging on their own message content."""
        sys_block = make_block(
            kind=BlockKind.SYSTEM, slot_index=0, hash="sys-shared", tokens_est=64
        )
        tools_block = make_block(
            kind=BlockKind.TOOLS, slot_index=0, hash="tools-shared", tokens_est=64
        )
        msg_a = make_block(slot_index=0, hash="msg-a", tokens_est=64)
        msg_b = make_block(slot_index=0, hash="msg-b", tokens_est=64)

        req_a = make_request(
            request_id=0,
            ts_ms=0.0,
            input_tokens=192,
            system=(sys_block,),
            tools=tools_block,
            messages=(msg_a,),
        )
        req_b = make_request(
            request_id=0,
            ts_ms=1.0,
            input_tokens=192,
            system=(sys_block,),
            tools=tools_block,
            messages=(msg_b,),
        )

        instance_a = make_instance(instance_id="inst-a", requests=(req_a,))
        instance_b = make_instance(instance_id="inst-b", requests=(req_b,))
        session = make_session(instances=(instance_a, instance_b))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )
        synth.build_instance_turns(instance_a)
        synth.build_instance_turns(instance_b)

        result = self._assert_matches_naive(synth)
        # instance_b's shared system+tools prefix (2 blocks) must register as
        # a hit against instance_a's already-emitted wire.
        hit_b, _ = result[("inst-b", 0)]
        assert hit_b > 0

    def test_mid_session_reset_finds_best_match_past_immediate_predecessor(self):
        """(c) a wire with a mid-session reset/context-truncation: the best
        match for the third request lives two requests back, not in its
        immediate (unrelated) predecessor."""
        req0 = make_request(
            request_id=0,
            ts_ms=0.0,
            input_tokens=128,
            messages=(
                make_block(slot_index=0, hash="m0", tokens_est=64),
                make_block(slot_index=1, hash="m1", tokens_est=64),
            ),
        )
        # Full reset: shares nothing with req0.
        req1 = make_request(
            request_id=1,
            ts_ms=1.0,
            input_tokens=64,
            messages=(make_block(slot_index=0, hash="reset-x", tokens_est=64),),
        )
        # Reuses req0's first block even though its immediate predecessor
        # (req1) shares nothing with it at all.
        req2 = make_request(
            request_id=2,
            ts_ms=2.0,
            input_tokens=128,
            messages=(
                make_block(slot_index=0, hash="m0", tokens_est=64),
                make_block(slot_index=1, hash="m-new", tokens_est=64),
            ),
        )
        instance = make_instance(requests=(req0, req1, req2))
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )
        synth.build_instance_turns(instance)

        result = self._assert_matches_naive(synth)
        hit2, _ = result[(instance.instance_id, 2)]
        assert hit2 > 0

    def test_fully_disjoint_wire_has_zero_overlap_with_any_prior(self):
        """(d) an instance whose wire is not a superset of (or sharing any
        prefix with) any prior wire at all."""
        req0 = make_request(
            request_id=0,
            ts_ms=0.0,
            input_tokens=64,
            messages=(make_block(slot_index=0, hash="alpha", tokens_est=64),),
        )
        req1 = make_request(
            request_id=1,
            ts_ms=1.0,
            input_tokens=64,
            messages=(make_block(slot_index=0, hash="beta", tokens_est=64),),
        )
        instance = make_instance(requests=(req0, req1))
        session = make_session(instances=(instance,))
        synth = ReplaySessionSynthesizer(
            session, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )
        synth.build_instance_turns(instance)

        result = self._assert_matches_naive(synth)
        hit1, _ = result[(instance.instance_id, 1)]
        assert hit1 == 0


def _naive_build_instance_turns(
    synth: ReplaySessionSynthesizer, instance: InstanceIR
) -> list[TurnContent]:
    """Reference copy of ``build_instance_turns`` as it existed before the
    skip-matched-prefix optimization: builds a ``_RenderItem`` for every
    block of every wire unconditionally (even ones a later slice discards),
    then slices the matched prefix away only when rendering. Operates
    in-place on ``synth``'s own identity/corpus-stream/diagnostics state, so
    its cid assignments and diagnostics are directly comparable to the
    optimized method's.
    """
    turns: list[TurnContent] = []
    prev_wire_cids: tuple[int, ...] | None = None

    for request in instance.requests:
        blocks = wire_blocks(request)
        resolutions = synth._identity.resolve_request(instance.instance_id, request)
        synth._assign_minted_lengths(request, blocks, resolutions)

        items: list[_RenderItem] = []
        wire_cids: list[int] = []
        for block, (cid, resolution) in zip(blocks, resolutions, strict=True):
            if resolution == Resolution.MINTED:
                synth._diag.minted_blocks += 1
            else:
                synth._diag.reused_blocks += 1

            anchor, length = synth._identity.anchor_and_length(cid)
            text = synth._corpus_stream.text_for(anchor, length)

            seen_key = (anchor, length)
            if seen_key not in synth._seen_content_ids:
                synth._seen_content_ids.add(seen_key)
                synth._diag.chars_emitted += len(text)
                synth._diag.unique_content_tokens_emitted += length

            role = block.role if block.kind == BlockKind.MESSAGE else "system"
            items.append(_RenderItem(kind=block.kind, role=role, text=text))
            wire_cids.append(cid)

        wire_cids_t = tuple(wire_cids)
        is_extension = (
            prev_wire_cids is not None
            and len(wire_cids_t) > len(prev_wire_cids)
            and wire_cids_t[: len(prev_wire_cids)] == prev_wire_cids
        )

        if prev_wire_cids is None:
            reset = False
            raw_messages = _render_wire(items, joiner=synth._cfg.system_joiner)
        elif is_extension:
            reset = False
            raw_messages = _render_wire(
                items[len(prev_wire_cids) :], joiner=synth._cfg.system_joiner
            )
        else:
            reset = True
            raw_messages = _render_wire(items, joiner=synth._cfg.system_joiner)

        turns.append(
            TurnContent(raw_messages=raw_messages, raw_tools=None, reset_context=reset)
        )
        synth._wire_records.append(
            (request.ts_ms, instance.instance_id, request.request_id, wire_cids_t)
        )
        synth._diag.requests_processed += 1
        prev_wire_cids = wire_cids_t

    return turns


class TestSkipMatchedPrefixOptimizationEquivalence:
    """Snapshot-compares the optimized ``build_instance_turns`` (which skips
    building a ``_RenderItem`` for blocks in an already-matched, about-to-be-
    sliced-away prefix) against the pre-optimization reference above, on a
    fixture exercising every branch: turn 0, a growing extension, an exact-
    repeat retry (full reset), a genuine divergence (full reset), and an
    extension resuming right after that reset.
    """

    def test_full_turn_content_and_diagnostics_match_naive_reference(self):
        growing = [
            make_block(slot_index=i, hash=f"m{i}", tokens_est=64) for i in range(3)
        ]
        req0 = make_request(
            request_id=0, ts_ms=0.0, input_tokens=64, messages=(growing[0],)
        )
        req1 = make_request(
            request_id=1,
            ts_ms=1.0,
            input_tokens=128,
            messages=(growing[0], growing[1]),
        )
        req2 = make_request(
            request_id=2,
            ts_ms=2.0,
            input_tokens=192,
            messages=(growing[0], growing[1], growing[2]),
        )
        # Exact repeat of req2's wire: must be a full reset, not a zero-delta
        # extension.
        req3 = make_request(
            request_id=3,
            ts_ms=3.0,
            input_tokens=192,
            messages=(growing[0], growing[1], growing[2]),
        )
        # Genuine divergence: shares nothing with req3.
        diverged = make_block(slot_index=0, hash="diverged", tokens_est=64)
        req4 = make_request(
            request_id=4, ts_ms=4.0, input_tokens=64, messages=(diverged,)
        )
        # Resume growing right after the reset.
        extra = make_block(slot_index=1, hash="extra", tokens_est=64)
        req5 = make_request(
            request_id=5,
            ts_ms=5.0,
            input_tokens=128,
            messages=(diverged, extra),
        )

        requests = (req0, req1, req2, req3, req4, req5)
        instance_optimized = make_instance(requests=requests)
        instance_naive = make_instance(requests=requests)

        session_optimized = make_session(instances=(instance_optimized,))
        session_naive = make_session(instances=(instance_naive,))

        synth_optimized = ReplaySessionSynthesizer(
            session_optimized, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )
        synth_naive = ReplaySessionSynthesizer(
            session_naive, corpus=FakeCorpus(), cfg=ReplaySynthConfig()
        )

        optimized_turns = synth_optimized.build_instance_turns(instance_optimized)
        naive_turns = _naive_build_instance_turns(synth_naive, instance_naive)

        assert optimized_turns == naive_turns
        assert optimized_turns[0].reset_context is False  # turn 0
        assert optimized_turns[1].reset_context is False  # growing extension
        assert optimized_turns[2].reset_context is False  # growing extension
        assert optimized_turns[3].reset_context is True  # exact-repeat retry
        assert optimized_turns[4].reset_context is True  # genuine divergence
        assert optimized_turns[5].reset_context is False  # resumed extension

        assert synth_optimized.diagnostics == synth_naive.diagnostics
        assert synth_optimized.finish() == synth_naive.finish()
