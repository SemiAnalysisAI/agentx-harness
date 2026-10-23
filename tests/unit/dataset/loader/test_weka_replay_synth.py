# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    ChunkedCorpusStream,
    ReplaySessionSynthesizer,
    ReplaySynthConfig,
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
