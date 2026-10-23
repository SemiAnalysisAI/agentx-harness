# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.dataset.loader._weka_replay_identity import (
    ChurnStats,
    Resolution,
    SessionContentIdentity,
)
from aiperf.dataset.loader._weka_replay_ir import (
    BlockKind,
    BlockSlotIR,
    InstanceIR,
    LinkIR,
    LinkKind,
    ReplayRole,
    RequestIR,
    SamplingIR,
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


class TestRollingCacheControlChurn:
    def test_churn_chain_keeps_one_cid_across_many_turns(self):
        identity = SessionContentIdentity()
        specs = [
            (None, 100, "h0"),
            ("eph", 137, "h1"),
            (None, 100, "h2"),
            ("eph", 137, "h3"),
            (None, 100, "h4"),
            ("eph", 137, "h5"),
        ]
        requests = tuple(
            make_request(
                request_id=i,
                ts_ms=float(i),
                input_tokens=10,
                messages=(
                    make_block(cache_control=cc, byte_count=b, hash=h, tokens_est=10),
                ),
            )
            for i, (cc, b, h) in enumerate(specs)
        )
        instance = make_instance(requests=requests)

        results = identity.resolve_instance(instance)

        cids = [turn[0][0] for turn in results]
        resolutions = [turn[0][1] for turn in results]
        assert resolutions[0] == Resolution.MINTED
        assert all(r == Resolution.CHURN for r in resolutions[1:])
        assert len(set(cids)) == 1

    def test_genuine_content_change_is_not_merged(self):
        identity = SessionContentIdentity()
        requests = (
            make_request(
                request_id=0,
                ts_ms=0.0,
                input_tokens=10,
                messages=(make_block(cache_control=None, byte_count=100, hash="g0"),),
            ),
            make_request(
                request_id=1,
                ts_ms=1.0,
                input_tokens=10,
                # cc flips, but the byte delta (50) is not a fragment multiple.
                messages=(make_block(cache_control="eph", byte_count=150, hash="g1"),),
            ),
        )
        instance = make_instance(requests=requests)

        results = identity.resolve_instance(instance)

        assert results[0][0][1] == Resolution.MINTED
        assert results[1][0][1] == Resolution.MINTED
        assert results[0][0][0] != results[1][0][0]

    def test_role_mismatch_blocks_churn_even_with_qualifying_delta(self):
        identity = SessionContentIdentity()
        requests = (
            make_request(
                request_id=0,
                ts_ms=0.0,
                input_tokens=10,
                messages=(
                    make_block(
                        role="user", cache_control=None, byte_count=100, hash="r0"
                    ),
                ),
            ),
            make_request(
                request_id=1,
                ts_ms=1.0,
                input_tokens=10,
                messages=(
                    make_block(
                        role="assistant", cache_control="eph", byte_count=137, hash="r1"
                    ),
                ),
            ),
        )
        instance = make_instance(requests=requests)

        results = identity.resolve_instance(instance)

        assert results[1][0][1] == Resolution.MINTED


class TestIndependentPerSlotChurn:
    """Regression test for a real production bug found by running this
    resolver against the real corpus: the identity resolver used to gate
    CHURN matching on a session-wide ``prefix_intact`` flag that went False
    the moment ANY position in the request diverged for any reason -- which
    silently zeroed out essentially all real cache_control-churn matching,
    because real captures carry an isolated, genuinely-volatile early block
    (observed: a small per-turn system block with no cache_control, most
    likely a cwd/env/timestamp header) that legitimately re-mints on nearly
    every request. That one expected, unrelated MINT used to poison churn
    matching for the rest of the request. Fixed: each slot's churn eligibility
    is now independent -- role match + a cache_control presence flip + an
    exact cache_control-fragment-sized byte delta is already a narrow,
    position-local signal that doesn't need corroboration from unrelated
    earlier positions in the same request.
    """

    def test_unrelated_earlier_divergence_does_not_block_later_churn_match(self):
        identity = SessionContentIdentity()
        block_a0 = make_block(slot_index=0, role="user", hash="a0", byte_count=100)
        block_b0 = make_block(slot_index=1, role="user", hash="b0", byte_count=200)
        # Turn 1: A changes genuinely (no cc flip at all -> can never churn-match
        # on its own -- e.g. a volatile per-turn header), B's change is
        # churn-shaped (role match, cc flip, delta=37) and must resolve via
        # CHURN regardless of A's unrelated divergence.
        block_a1 = make_block(slot_index=0, role="user", hash="a1", byte_count=500)
        block_b1 = make_block(
            slot_index=1, role="user", cache_control="eph", hash="b1", byte_count=237
        )

        req0 = make_request(
            request_id=0, ts_ms=0.0, input_tokens=20, messages=(block_a0, block_b0)
        )
        req1 = make_request(
            request_id=1, ts_ms=1.0, input_tokens=20, messages=(block_a1, block_b1)
        )
        instance = make_instance(requests=(req0, req1))

        results = identity.resolve_instance(instance)
        turn0, turn1 = results

        assert turn0[0][1] == Resolution.MINTED
        assert turn0[1][1] == Resolution.MINTED
        # A genuinely diverged (no cc flip) -> correctly MINTED.
        assert turn1[0][1] == Resolution.MINTED
        # B's own flip is churn-shaped and independent of A -> must CHURN,
        # reusing B's own id from turn 0 (not a fresh mint).
        assert turn1[1][1] == Resolution.CHURN
        assert turn1[1][0] == turn0[1][0]


class TestRealCorpusFragmentSizes:
    """Regression test for the third cache_control-fragment byte size found
    by scanning the real corpus: a consistent, deterministic -73 byte delta
    (role=user, text-only messages, 'ephemeral' -> None) occurring dozens of
    times across multiple real sessions -- distinct from the 37/48-byte
    fragments already covered, and far too clean/repeated to be a
    coincidental content change. Its exact JSON shape wasn't reverse
    engineered, but the byte delta itself is unambiguous.
    """

    def test_73_byte_fragment_churns(self):
        identity = SessionContentIdentity()
        block0 = make_block(
            slot_index=0,
            role="user",
            hash="u0",
            byte_count=190,
            cache_control="ephemeral",
        )
        block1 = make_block(
            slot_index=0,
            role="user",
            hash="u1",
            byte_count=117,
            cache_control=None,
        )
        req0 = make_request(
            request_id=0, ts_ms=0.0, input_tokens=10, messages=(block0,)
        )
        req1 = make_request(
            request_id=1, ts_ms=1.0, input_tokens=10, messages=(block1,)
        )
        instance = make_instance(requests=(req0, req1))

        results = identity.resolve_instance(instance)
        assert results[1][0][1] == Resolution.CHURN
        assert results[1][0][0] == results[0][0][0]
        assert identity.stats().churn_count == 1


class TestSeedHashResolution:
    def test_shifted_repeated_user_message_resolves_via_seed_hash(self):
        identity = SessionContentIdentity()
        original = make_block(
            slot_index=0,
            role="user",
            hash="u0",
            byte_count=120,
            cache_control=None,
            seed_hash="seed-A",
        )
        filler_1 = make_block(slot_index=0, role="user", hash="x0", byte_count=10)
        filler_2 = make_block(slot_index=1, role="assistant", hash="x1", byte_count=10)
        # Same underlying text (same seed_hash), but now wrapped with a
        # system-reminder and cache_control, so its raw hash and byte count
        # both differ from the original occurrence; normalizing out the
        # cache_control fragment (237 - 37 == 120) recovers the match.
        shifted = make_block(
            slot_index=2,
            role="user",
            hash="u1",
            byte_count=157,
            cache_control="eph",
            seed_hash="seed-A",
        )

        req0 = make_request(
            request_id=0, ts_ms=0.0, input_tokens=10, messages=(original,)
        )
        req1 = make_request(
            request_id=1,
            ts_ms=1.0,
            input_tokens=30,
            messages=(filler_1, filler_2, shifted),
        )
        instance = make_instance(requests=(req0, req1))

        results = identity.resolve_instance(instance)

        original_cid = results[0][0][0]
        shifted_cid, shifted_resolution = results[1][2]
        assert shifted_resolution == Resolution.SEED_HASH
        assert shifted_cid == original_cid


class TestCrossInstanceSharing:
    def test_exact_hash_shared_across_instances_of_same_session(self):
        identity = SessionContentIdentity()
        shared_block_a = make_block(
            kind=BlockKind.SYSTEM,
            slot_index=0,
            role="system",
            hash="sys-shared",
            byte_count=500,
        )
        shared_block_b = make_block(
            kind=BlockKind.SYSTEM,
            slot_index=0,
            role="system",
            hash="sys-shared",
            byte_count=500,
        )

        instance_a = make_instance(
            instance_id="sess:a:fp",
            requests=(
                make_request(
                    request_id=0, ts_ms=0.0, input_tokens=50, system=(shared_block_a,)
                ),
            ),
        )
        instance_b = make_instance(
            instance_id="sess:b:fp",
            requests=(
                make_request(
                    request_id=1, ts_ms=1.0, input_tokens=50, system=(shared_block_b,)
                ),
            ),
        )

        results_a = identity.resolve_instance(instance_a)
        results_b = identity.resolve_instance(instance_b)

        cid_a, res_a = results_a[0][0]
        cid_b, res_b = results_b[0][0]
        assert res_a == Resolution.MINTED
        assert res_b == Resolution.EXACT_HASH
        assert cid_a == cid_b


class TestAnchorAndLength:
    def test_anchor_and_length_frozen_at_first_mint(self):
        identity = SessionContentIdentity()
        specs = [
            (None, 100, "h0", 42),
            ("eph", 137, "h1", 999),
            (None, 100, "h2", 7),
        ]
        requests = tuple(
            make_request(
                request_id=i,
                ts_ms=float(i),
                input_tokens=10,
                messages=(
                    make_block(cache_control=cc, byte_count=b, hash=h, tokens_est=tok),
                ),
            )
            for i, (cc, b, h, tok) in enumerate(specs)
        )
        instance = make_instance(requests=requests)

        results = identity.resolve_instance(instance)
        cid = results[0][0][0]
        assert all(turn[0][0] == cid for turn in results)

        anchor, length = identity.anchor_and_length(cid)
        assert anchor == "h0"
        assert length == 42


class TestChurnStatsHistogram:
    def test_flip_delta_hist_and_tight_loose_disagreement(self):
        identity = SessionContentIdentity()

        # Case A: delta=37 against a large previous byte count -> tight and
        # loose agree (both pass) -> no disagreement.
        agree_prev = make_block(
            slot_index=0,
            role="user",
            hash="agree0",
            byte_count=1000,
            cache_control=None,
        )
        agree_next = make_block(
            slot_index=0,
            role="user",
            hash="agree1",
            byte_count=1037,
            cache_control="eph",
        )
        # Case B: delta=37 against a small previous byte count -> tight
        # passes but loose fails -> disagreement.
        disagree_prev = make_block(
            slot_index=1, role="user", hash="dis0", byte_count=50, cache_control=None
        )
        disagree_next = make_block(
            slot_index=1, role="user", hash="dis1", byte_count=87, cache_control="eph"
        )

        req0 = make_request(
            request_id=0,
            ts_ms=0.0,
            input_tokens=20,
            messages=(agree_prev, disagree_prev),
        )
        req1 = make_request(
            request_id=1,
            ts_ms=1.0,
            input_tokens=20,
            messages=(agree_next, disagree_next),
        )
        instance = make_instance(requests=(req0, req1))

        results = identity.resolve_instance(instance)
        assert results[1][0][1] == Resolution.CHURN
        assert results[1][1][1] == Resolution.CHURN

        stats: ChurnStats = identity.stats()
        assert stats.flip_byte_delta_hist.get(37, 0) == 2
        assert stats.tight_loose_disagreement_count == 1
