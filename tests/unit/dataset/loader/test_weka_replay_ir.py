# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.dataset.loader._weka_replay_ir import (
    BlockKind,
    LinkKind,
    ReplayRole,
    SessionDiagnostics,
    normalize_session,
)
from aiperf.dataset.loader._weka_replay_reader import (
    WekaReplayLoadError,
    read_header,
    read_session,
    scan_session_spans,
)
from aiperf.dataset.loader.weka_replay_models import ReplaySession
from tests.unit.dataset.loader._weka_replay_builders import (
    build_header,
    build_instance,
    build_request,
    build_session,
    write_lines,
)


def _normalize(
    session_dict: dict, *, schema_version: str = "replay-v3", lineno: int = 2
):
    raw = ReplaySession.model_validate(session_dict)
    diag = SessionDiagnostics()
    ir = normalize_session(raw, schema_version=schema_version, lineno=lineno, diag=diag)
    return ir, diag


class TestEndToEndThroughReaderAndIR:
    def test_v3_file_parses_end_to_end(self, tmp_path):
        header_dict = build_header(schema_version="replay-v3")
        session_dict = build_session(
            session_id="sess1",
            instances=[
                build_instance(
                    instance_id="sess1:sig:fp",
                    role="main",
                    requests=[
                        build_request(request_id=1),
                        build_request(request_id=2, ts="2024-01-01T00:00:02Z"),
                    ],
                )
            ],
        )
        path = write_lines(tmp_path, [header_dict, session_dict])
        header = read_header(path)
        spans, _index = scan_session_spans(path, header)
        assert len(spans) == 1
        raw_session = read_session(path, spans[0])
        diag = SessionDiagnostics()
        ir = normalize_session(
            raw_session,
            schema_version=header.schema_version,
            lineno=spans[0].lineno,
            diag=diag,
        )
        assert ir.session_id == "sess1"
        assert len(ir.instances) == 1
        assert ir.instances[0].role == ReplayRole.MAIN
        assert [r.request_id for r in ir.instances[0].requests] == [1, 2]
        assert ir.request_owner == {1: (0, 0), 2: (0, 1)}


class TestTimestampParsing:
    @pytest.mark.parametrize(
        "ts",
        [
            "2024-01-01T00:00:01Z",
            "2024-01-01T00:00:01.5Z",
            "2024-01-01T00:00:01.123Z",
            "2024-01-01T00:00:01.123456Z",
            "2024-01-01T00:00:01.123456789Z",
            "2024-01-01T00:00:01+00:00",
            "2024-01-01T00:00:01.123456789+02:00",
        ],
    )  # fmt: skip
    def test_various_rfc3339_shapes_parse(self, ts):
        session_dict = build_session(
            instances=[build_instance(requests=[build_request(ts=ts)])]
        )
        ir, diag = _normalize(session_dict)
        assert ir.instances[0].requests[0].ts_ms > 0
        assert diag.dropped_instance_count == 0

    def test_trailing_zero_trimmed_fraction_is_mathematically_correct(self):
        """ ".5" means 500ms, not 5ms - right-padding with zeros (not left) is
        the correct interpretation of Go's trailing-zero-trimmed fraction."""
        session_a = build_session(
            session_id="a",
            instances=[
                build_instance(requests=[build_request(ts="2024-01-01T00:00:01Z")])
            ],
        )
        session_b = build_session(
            session_id="b",
            instances=[
                build_instance(requests=[build_request(ts="2024-01-01T00:00:01.5Z")])
            ],
        )
        ir_a, _ = _normalize(session_a)
        ir_b, _ = _normalize(session_b)
        delta_ms = (
            ir_b.instances[0].requests[0].ts_ms - ir_a.instances[0].requests[0].ts_ms
        )
        assert delta_ms == pytest.approx(500.0)

    def test_invalid_timestamp_raises(self):
        session_dict = build_session(
            instances=[build_instance(requests=[build_request(ts="not-a-timestamp")])]
        )
        with pytest.raises(WekaReplayLoadError, match="invalid timestamp"):
            _normalize(session_dict)

    def test_invalid_start_ts_raises(self):
        session_dict = build_session(start_ts="also-not-a-timestamp")
        with pytest.raises(WekaReplayLoadError, match="invalid timestamp"):
            _normalize(session_dict)


class TestOrphanShapeDetection:
    def test_shared_bucket(self):
        session_dict = build_session(session_id="orphan:no-session-id")
        ir, _ = _normalize(session_dict)
        assert ir.orphan_shape == "shared_bucket"

    def test_per_record(self):
        session_dict = build_session(session_id="orphan:no-session-id:abc123")
        ir, _ = _normalize(session_dict)
        assert ir.orphan_shape == "per_record"

    def test_none(self):
        session_dict = build_session(session_id="a-real-session")
        ir, _ = _normalize(session_dict)
        assert ir.orphan_shape == "none"


class TestRoleMapping:
    @pytest.mark.parametrize(
        "raw_role,expected",
        [
            ("main", ReplayRole.MAIN),
            ("sub-agent", ReplayRole.SUB_AGENT),
            ("helper-or-isolated", ReplayRole.HELPER),
            ("orphan-sub-agent", ReplayRole.ORPHAN_SUB_AGENT),
            ("ephemeral (no system)", ReplayRole.EPHEMERAL),
        ],
    )  # fmt: skip
    def test_known_roles(self, raw_role, expected):
        session_dict = build_session(instances=[build_instance(role=raw_role)])
        ir, diag = _normalize(session_dict)
        assert ir.instances[0].role == expected
        assert diag.unknown_role_count == 0

    def test_unknown_role_warns_not_raises(self):
        session_dict = build_session(instances=[build_instance(role="something-new")])
        ir, diag = _normalize(session_dict)
        assert ir.instances[0].role == ReplayRole.UNKNOWN
        assert ir.instances[0].raw_role == "something-new"
        assert diag.unknown_role_count == 1
        assert "something-new" in diag.unknown_role_examples


class TestEmptyInstanceDropped:
    def test_instance_with_no_requests_is_dropped(self):
        session_dict = build_session(
            instances=[
                build_instance(instance_id="empty:sig:fp", requests=[]),
                build_instance(instance_id="real:sig:fp", requests=[build_request()]),
            ]
        )
        ir, diag = _normalize(session_dict)
        assert [i.instance_id for i in ir.instances] == ["real:sig:fp"]
        assert diag.dropped_instance_count == 1
        assert "empty:sig:fp:empty_requests" in diag.dropped_instance_examples

    def test_session_with_no_instances(self):
        session_dict = build_session()
        session_dict["instances"] = None
        ir, diag = _normalize(session_dict)
        assert ir.instances == ()
        assert ir.request_owner == {}


class TestRequestSorting:
    def test_already_sorted_not_flagged(self):
        session_dict = build_session(
            instances=[
                build_instance(
                    requests=[
                        build_request(request_id=1, ts="2024-01-01T00:00:01Z"),
                        build_request(request_id=2, ts="2024-01-01T00:00:02Z"),
                    ]
                )
            ]
        )
        ir, diag = _normalize(session_dict)
        assert [r.request_id for r in ir.instances[0].requests] == [1, 2]
        assert diag.requests_resorted == 0

    def test_out_of_order_is_sorted_and_flagged(self):
        session_dict = build_session(
            instances=[
                build_instance(
                    requests=[
                        build_request(request_id=2, ts="2024-01-01T00:00:02Z"),
                        build_request(request_id=1, ts="2024-01-01T00:00:01Z"),
                    ]
                )
            ]
        )
        ir, diag = _normalize(session_dict)
        assert [r.request_id for r in ir.instances[0].requests] == [1, 2]
        assert diag.requests_resorted == 1

    def test_tie_broken_by_request_id(self):
        session_dict = build_session(
            instances=[
                build_instance(
                    requests=[
                        build_request(request_id=5, ts="2024-01-01T00:00:01Z"),
                        build_request(request_id=3, ts="2024-01-01T00:00:01Z"),
                    ]
                )
            ]
        )
        ir, _ = _normalize(session_dict)
        assert [r.request_id for r in ir.instances[0].requests] == [3, 5]


class TestDuplicateRequestIds:
    def test_duplicate_across_instances_warns(self):
        session_dict = build_session(
            instances=[
                build_instance(
                    instance_id="i1:sig:fp", requests=[build_request(request_id=1)]
                ),
                build_instance(
                    instance_id="i2:sig:fp",
                    requests=[build_request(request_id=1, ts="2024-01-01T00:00:05Z")],
                ),
            ]
        )
        ir, diag = _normalize(session_dict)
        assert diag.duplicate_request_id_count == 1
        assert ir.request_owner[1] == (0, 0)  # first owner wins

    def test_duplicate_referenced_by_parent_spawn_raises(self):
        session_dict = build_session(
            instances=[
                build_instance(
                    instance_id="i1:sig:fp", requests=[build_request(request_id=1)]
                ),
                build_instance(
                    instance_id="i2:sig:fp",
                    requests=[build_request(request_id=1, ts="2024-01-01T00:00:05Z")],
                ),
                build_instance(
                    instance_id="i3:sig:fp",
                    parent_spawn_request_id=1,
                    fan_out_group="1",
                    requests=[build_request(request_id=2, ts="2024-01-01T00:00:06Z")],
                ),
            ]
        )
        with pytest.raises(WekaReplayLoadError, match="ambiguous"):
            _normalize(session_dict)


class TestLinkClassification:
    """One case per row of plan §4.1's link-classification table."""

    def _two_instance_session(
        self, *, parent_ts: str, child_overrides: dict, child_first_ts: str
    ):
        return build_session(
            instances=[
                build_instance(
                    instance_id="parent:sig:fp",
                    role="main",
                    requests=[build_request(request_id=1, ts=parent_ts)],
                ),
                build_instance(
                    instance_id="child:sig:fp",
                    role="sub-agent",
                    requests=[build_request(request_id=2, ts=child_first_ts)],
                    **child_overrides,
                ),
            ]
        )

    def test_row1_cycle_broken_group_set_parent_absent(self):
        session_dict = self._two_instance_session(
            parent_ts="2024-01-01T00:00:01Z",
            child_overrides={"fan_out_group": "1"},
            child_first_ts="2024-01-01T00:00:02Z",
        )
        ir, diag = _normalize(session_dict)
        child = ir.instances[1]
        assert child.link.kind == LinkKind.CYCLE_BROKEN
        assert child.link.reason == "cycle_broken"
        assert diag.cycle_broken_count == 1

    def test_row1_cycle_broken_group_set_parent_zero(self):
        session_dict = self._two_instance_session(
            parent_ts="2024-01-01T00:00:01Z",
            child_overrides={"fan_out_group": "1", "parent_spawn_request_id": 0},
            child_first_ts="2024-01-01T00:00:02Z",
        )
        ir, _ = _normalize(session_dict)
        assert ir.instances[1].link.kind == LinkKind.CYCLE_BROKEN

    def test_row2_anomalous_group_mismatch(self):
        session_dict = self._two_instance_session(
            parent_ts="2024-01-01T00:00:01Z",
            child_overrides={"fan_out_group": "99", "parent_spawn_request_id": 1},
            child_first_ts="2024-01-01T00:00:02Z",
        )
        ir, diag = _normalize(session_dict)
        child = ir.instances[1]
        assert child.link.kind == LinkKind.ANOMALOUS
        assert child.link.reason == "group_mismatch"
        assert diag.anomalous_count == 1

    def test_row3_anomalous_unowned(self):
        session_dict = self._two_instance_session(
            parent_ts="2024-01-01T00:00:01Z",
            child_overrides={"fan_out_group": "404", "parent_spawn_request_id": 404},
            child_first_ts="2024-01-01T00:00:02Z",
        )
        ir, diag = _normalize(session_dict)
        child = ir.instances[1]
        assert child.link.kind == LinkKind.ANOMALOUS
        assert child.link.reason == "unowned"

    def test_row3_anomalous_self_spawn(self):
        session_dict = build_session(
            instances=[
                build_instance(
                    instance_id="solo:sig:fp",
                    role="main",
                    fan_out_group="1",
                    parent_spawn_request_id=1,
                    requests=[
                        build_request(request_id=1, ts="2024-01-01T00:00:01Z"),
                        build_request(request_id=2, ts="2024-01-01T00:00:02Z"),
                    ],
                ),
            ]
        )
        ir, diag = _normalize(session_dict)
        assert ir.instances[0].link.kind == LinkKind.ANOMALOUS
        assert ir.instances[0].link.reason == "self_spawn"

    def test_row4_anomalous_time_inversion(self):
        # Child's first_ts_ms is BEFORE the parent's spawning request's ts_ms.
        session_dict = self._two_instance_session(
            parent_ts="2024-01-01T00:00:10Z",
            child_overrides={"fan_out_group": "1", "parent_spawn_request_id": 1},
            child_first_ts="2024-01-01T00:00:01Z",
        )
        ir, diag = _normalize(session_dict)
        child = ir.instances[1]
        assert child.link.kind == LinkKind.ANOMALOUS
        assert child.link.reason == "time_inversion"

    def test_row5_causal(self):
        session_dict = self._two_instance_session(
            parent_ts="2024-01-01T00:00:01Z",
            child_overrides={
                "fan_out_group": "1",
                "parent_spawn_request_id": 1,
                "fan_out_size": 2,
            },
            child_first_ts="2024-01-01T00:00:02Z",
        )
        ir, diag = _normalize(session_dict)
        child = ir.instances[1]
        assert child.link.kind == LinkKind.CAUSAL
        assert child.link.reason == "ok"
        assert child.link.parent_request_id == 1
        assert child.link.fan_out_group == "1"
        assert child.link.fan_out_size == 2
        assert diag.anomalous_count == 0
        assert diag.cycle_broken_count == 0

    def test_row6_implicit(self):
        session_dict = self._two_instance_session(
            parent_ts="2024-01-01T00:00:01Z",
            child_overrides={"parent_spawn_request_id": 1},
            child_first_ts="2024-01-01T00:00:02Z",
        )
        ir, _ = _normalize(session_dict)
        assert ir.instances[1].link.kind == LinkKind.IMPLICIT
        assert ir.instances[1].link.parent_request_id == 1

    def test_row7_none(self):
        session_dict = build_session(
            instances=[
                build_instance(role="main", requests=[build_request(request_id=1)])
            ]
        )
        ir, _ = _normalize(session_dict)
        assert ir.instances[0].link.kind == LinkKind.NONE
        assert ir.instances[0].link.parent_request_id is None

    def test_fan_out_position_defaults_to_zero_when_absent(self):
        session_dict = build_session(
            instances=[
                build_instance(role="main", requests=[build_request(request_id=1)])
            ]
        )
        ir, _ = _normalize(session_dict)
        assert ir.instances[0].link.fan_out_position == 0


class TestTokenEstimateFill:
    def test_all_recorded_no_fill(self):
        req = build_request(
            input_tokens=30,
            system_blocks=[
                {"type": "text", "hash": "sha256:s", "bytes": 300, "tokens": 20}
            ],
            messages=[{"hash": "sha256:m", "bytes": 50, "tokens": 10, "role": "user"}],
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, diag = _normalize(session_dict)
        blocks = ir.instances[0].requests[0]
        assert blocks.system[0].tokens_est == 20
        assert blocks.messages[0].tokens_est == 10
        assert diag.token_estimate_fill_count == 0
        assert diag.input_tokens_mismatch_count == 0

    def test_mismatch_counted_when_no_fill_needed(self):
        req = build_request(
            input_tokens=999,  # deliberately doesn't match recorded block tokens
            system_blocks=[
                {"type": "text", "hash": "sha256:s", "bytes": 300, "tokens": 20}
            ],
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, diag = _normalize(session_dict)
        assert diag.input_tokens_mismatch_count == 1
        assert diag.token_estimate_fill_count == 0

    def test_fill_sums_exactly_to_input_tokens(self):
        req = build_request(
            input_tokens=100,
            system_blocks=[
                {"type": "text", "hash": "sha256:s0", "bytes": 300, "tokens": 0},
                {"type": "text", "hash": "sha256:s1", "bytes": 100, "tokens": 0},
            ],
            messages=[{"hash": "sha256:m", "bytes": 600, "tokens": 0, "role": "user"}],
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, diag = _normalize(session_dict)
        blocks = ir.instances[0].requests[0]
        total = (
            blocks.system[0].tokens_est
            + blocks.system[1].tokens_est
            + blocks.messages[0].tokens_est
        )
        assert total == 100
        assert diag.token_estimate_fill_count == 3
        # proportional by bytes: 300:100:600 of 100 tokens -> 30:10:60
        assert blocks.system[0].tokens_est == 30
        assert blocks.system[1].tokens_est == 10
        assert blocks.messages[0].tokens_est == 60

    def test_partial_fill_keeps_recorded_and_fills_remainder_exactly(self):
        req = build_request(
            input_tokens=100,
            system_blocks=[
                {"type": "text", "hash": "sha256:s", "bytes": 300, "tokens": 40}
            ],  # recorded, frozen
            messages=[
                {"hash": "sha256:m0", "bytes": 100, "tokens": 0, "role": "user"},
                {"hash": "sha256:m1", "bytes": 300, "tokens": 0, "role": "assistant"},
            ],
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, diag = _normalize(session_dict)
        blocks = ir.instances[0].requests[0]
        assert blocks.system[0].tokens_est == 40  # untouched, recorded
        remaining = blocks.messages[0].tokens_est + blocks.messages[1].tokens_est
        assert remaining == 60
        total = blocks.system[0].tokens_est + remaining
        assert total == 100
        assert diag.token_estimate_fill_count == 2

    def test_remainder_goes_to_last_fill_block(self):
        # 3 blocks of equal size sharing a remaining budget that doesn't divide evenly.
        req = build_request(
            input_tokens=10,
            system_blocks=[
                {"type": "text", "hash": "sha256:s0", "bytes": 300, "tokens": 0},
                {"type": "text", "hash": "sha256:s1", "bytes": 300, "tokens": 0},
                {"type": "text", "hash": "sha256:s2", "bytes": 300, "tokens": 0},
            ],
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        blocks = ir.instances[0].requests[0].system
        assert sum(b.tokens_est for b in blocks) == 10
        assert blocks[0].tokens_est == 3
        assert blocks[1].tokens_est == 3
        assert blocks[2].tokens_est == 4  # remainder


class TestBlockTypesNullRoundTrip:
    def test_null_block_types_becomes_empty_tuple(self):
        req = build_request(
            system_blocks=[
                {
                    "type": "text",
                    "hash": "sha256:s",
                    "bytes": 300,
                    "tokens": 5,
                    "block_types": None,
                }
            ]
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        assert ir.instances[0].requests[0].system[0].block_types == ()

    def test_present_block_types_round_trip(self):
        req = build_request(
            system_blocks=[
                {
                    "type": "text",
                    "hash": "sha256:s",
                    "bytes": 300,
                    "tokens": 5,
                    "block_types": ["text", "tool_use"],
                }
            ]
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        assert ir.instances[0].requests[0].system[0].block_types == ("text", "tool_use")


class TestBlockSlotAssembly:
    def test_kinds_and_slot_indices(self):
        req = build_request(
            system_blocks=[
                {"type": "text", "hash": "sha256:s0", "bytes": 300, "tokens": 5},
                {"type": "text", "hash": "sha256:s1", "bytes": 10, "tokens": 5},
            ],
            tools={"hash": "sha256:t", "bytes": 50, "tokens": 10, "count": 3},
            messages=[
                {"hash": "sha256:m0", "bytes": 10, "tokens": 5, "role": "user"},
                {"hash": "sha256:m1", "bytes": 10, "tokens": 5, "role": "assistant"},
            ],
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        r = ir.instances[0].requests[0]
        assert [b.slot_index for b in r.system] == [0, 1]
        assert r.tools.kind == BlockKind.TOOLS
        assert r.tools.slot_index == 0
        assert r.tools.tool_count == 3
        assert [b.slot_index for b in r.messages] == [0, 1]
        assert [b.role for b in r.messages] == ["user", "assistant"]

    def test_no_tools_block_is_none(self):
        req = build_request()
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        assert ir.instances[0].requests[0].tools is None

    def test_seed_hash_only_on_messages(self):
        req = build_request(
            system_blocks=[
                {"type": "text", "hash": "sha256:s", "bytes": 300, "tokens": 5}
            ],
            messages=[
                {
                    "hash": "sha256:m",
                    "bytes": 10,
                    "tokens": 5,
                    "role": "user",
                    "seed_hash": "sha256:seed",
                }
            ],
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        r = ir.instances[0].requests[0]
        assert r.system[0].seed_hash is None
        assert r.messages[0].seed_hash == "sha256:seed"


class TestBillingHeaderDropped:
    """``system_blocks[0]`` is a replay-capture billing/preamble header when
    small (``bytes < 200``) -- mirrors wekai's own ``effectiveSystemBlocks``
    (``benchmark/replay_router_wire.go``). Unlike Anthropic's real
    ``cache_control`` breakpoints (which sit on the shared system blocks that
    follow it), this header carries a near-unique hash per request and must
    be dropped before any cache-relevant processing, not just excluded from
    the metric after the fact."""

    def test_small_first_system_block_is_dropped(self):
        req = build_request(
            system_blocks=[
                {"type": "text", "hash": "sha256:header", "bytes": 106, "tokens": 5},
                {"type": "text", "hash": "sha256:real", "bytes": 300, "tokens": 20},
            ]
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, diag = _normalize(session_dict)
        r = ir.instances[0].requests[0]
        assert [b.hash for b in r.system] == ["sha256:real"]
        assert diag.billing_header_dropped_count == 1

    def test_large_first_system_block_is_kept(self):
        req = build_request(
            system_blocks=[
                {"type": "text", "hash": "sha256:real", "bytes": 200, "tokens": 20}
            ]
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, diag = _normalize(session_dict)
        r = ir.instances[0].requests[0]
        assert [b.hash for b in r.system] == ["sha256:real"]
        assert diag.billing_header_dropped_count == 0

    def test_only_index_zero_is_checked(self):
        """A small block at any OTHER position is real content, not the
        header, and must never be dropped."""
        req = build_request(
            system_blocks=[
                {"type": "text", "hash": "sha256:real0", "bytes": 300, "tokens": 20},
                {"type": "text", "hash": "sha256:small1", "bytes": 10, "tokens": 5},
            ]
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, diag = _normalize(session_dict)
        r = ir.instances[0].requests[0]
        assert [b.hash for b in r.system] == ["sha256:real0", "sha256:small1"]
        assert diag.billing_header_dropped_count == 0

    def test_dropping_is_consistent_across_the_whole_file(self):
        """Every request in a real capture carries this header, so dropping
        it should never produce a one-off slot-index mismatch between turns."""
        reqs = [
            build_request(
                request_id=i,
                ts=f"2024-01-01T00:00:{i:02d}Z",
                system_blocks=[
                    {
                        "type": "text",
                        "hash": f"sha256:header{i}",
                        "bytes": 100,
                        "tokens": 5,
                    },
                    {"type": "text", "hash": "sha256:real", "bytes": 300, "tokens": 20},
                ],
            )
            for i in range(1, 4)
        ]
        session_dict = build_session(instances=[build_instance(requests=reqs)])
        ir, diag = _normalize(session_dict)
        for request in ir.instances[0].requests:
            assert [b.slot_index for b in request.system] == [0]
            assert request.system[0].hash == "sha256:real"
        assert diag.billing_header_dropped_count == 3

    def test_slot_indices_renumbered_from_zero_after_drop(self):
        req = build_request(
            system_blocks=[
                {"type": "text", "hash": "sha256:header", "bytes": 50, "tokens": 5},
                {"type": "text", "hash": "sha256:s0", "bytes": 300, "tokens": 20},
                {"type": "text", "hash": "sha256:s1", "bytes": 300, "tokens": 20},
            ]
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        r = ir.instances[0].requests[0]
        assert [b.slot_index for b in r.system] == [0, 1]
        assert [b.hash for b in r.system] == ["sha256:s0", "sha256:s1"]

    def test_only_system_block_being_the_header_yields_no_system_blocks(self):
        req = build_request(
            system_blocks=[
                {"type": "text", "hash": "sha256:header", "bytes": 50, "tokens": 5}
            ]
        )
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, diag = _normalize(session_dict)
        assert ir.instances[0].requests[0].system == ()
        assert diag.billing_header_dropped_count == 1


class TestApiTimeAndLastEnd:
    def test_v3_api_time_always_none(self):
        req = build_request(total_ms=0.0)
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        assert ir.instances[0].requests[0].api_time_ms is None

    def test_v4_api_time_from_total_ms(self):
        req = build_request(total_ms=250.0)
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict, schema_version="replay-v4")
        assert ir.instances[0].requests[0].api_time_ms == 250.0

    def test_last_end_ms_includes_api_time(self):
        req = build_request(request_id=1, ts="2024-01-01T00:00:01Z", total_ms=500.0)
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        inst = ir.instances[0]
        assert inst.last_end_ms == pytest.approx(inst.first_ts_ms + 500.0)

    def test_last_end_ms_without_api_time_equals_ts(self):
        req = build_request(request_id=1, ts="2024-01-01T00:00:01Z")
        session_dict = build_session(instances=[build_instance(requests=[req])])
        ir, _ = _normalize(session_dict)
        inst = ir.instances[0]
        assert inst.last_end_ms == inst.first_ts_ms


class TestDepthPreservedNoFlattening:
    def test_three_level_chain_all_present(self):
        session_dict = build_session(
            instances=[
                build_instance(
                    instance_id="root:sig:fp",
                    role="main",
                    requests=[build_request(request_id=1, ts="2024-01-01T00:00:01Z")],
                ),
                build_instance(
                    instance_id="mid:sig:fp",
                    role="sub-agent",
                    fan_out_group="1",
                    parent_spawn_request_id=1,
                    requests=[build_request(request_id=2, ts="2024-01-01T00:00:02Z")],
                ),
                build_instance(
                    instance_id="leaf:sig:fp",
                    role="sub-agent",
                    fan_out_group="2",
                    parent_spawn_request_id=2,
                    requests=[build_request(request_id=3, ts="2024-01-01T00:00:03Z")],
                ),
            ]
        )
        ir, _ = _normalize(session_dict)
        assert len(ir.instances) == 3
        root, mid, leaf = ir.instances
        assert root.link.kind == LinkKind.NONE
        assert mid.link.kind == LinkKind.CAUSAL
        assert mid.link.parent_request_id == 1
        assert leaf.link.kind == LinkKind.CAUSAL
        assert leaf.link.parent_request_id == 2
