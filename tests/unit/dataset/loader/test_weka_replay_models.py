# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest
from pydantic import ValidationError

from aiperf.dataset.loader.weka_replay_models import (
    ReplayHeader,
    ReplayIndex,
    ReplayInstance,
    ReplayMessageBlock,
    ReplayRequest,
    ReplaySession,
    ReplaySummary,
    ReplayThinkPercentiles,
)
from tests.unit.dataset.loader._weka_replay_builders import (
    build_header,
    build_instance,
    build_request,
    build_session,
)


class TestReplayHeader:
    @pytest.mark.parametrize("schema", ["replay-v3", "replay-v4", "replay-v5"])  # fmt: skip
    def test_schema_versions_parse(self, schema):
        header = ReplayHeader.model_validate(build_header(schema_version=schema))
        assert header.schema_version == schema

    def test_schema_alias(self):
        raw = build_header()
        header = ReplayHeader.model_validate(raw)
        assert "_schema" in raw
        assert header.schema_version == raw["_schema"]

    def test_unknown_schema_rejected(self):
        with pytest.raises(ValidationError):
            ReplayHeader.model_validate(build_header(schema_version="replay-v9"))

    def test_unknown_top_level_field_rejected(self):
        raw = build_header()
        raw["unexpected_field"] = 1
        with pytest.raises(ValidationError):
            ReplayHeader.model_validate(raw)

    def test_summary_timing_percentiles_v4(self):
        raw = build_header(
            schema_version="replay-v4",
            summary={
                "sessions": 10,
                "timing": {
                    "think_ms": {"p50": 1, "p90": 2, "p99": 3},
                    "total_ms": {"p50": 4, "p90": 5, "p99": 6},
                },
            },
        )
        header = ReplayHeader.model_validate(raw)
        assert header.summary.sessions == 10
        assert header.summary.timing.think_ms.p50 == 1
        assert header.summary.timing.think_ms.p99 == 3
        assert header.summary.timing.total_ms.p99 == 6

    def test_summary_permits_unknown_fields(self):
        """ReplaySummary/ReplaySummaryTiming field names are a best-effort
        reconstruction (see weka_replay_models module docstring); they stay
        permissive so an unverified real-world field doesn't block loading."""
        raw = build_header(summary={"totally_unverified_field": 42})
        header = ReplayHeader.model_validate(raw)
        assert header.summary.model_dump(mode="json")["totally_unverified_field"] == 42

    def test_orjson_roundtrip(self):
        raw = build_header(schema_version="replay-v5")
        header = ReplayHeader.model_validate(orjson.loads(orjson.dumps(raw)))
        assert header.schema_version == "replay-v5"


class TestReplaySummary:
    def test_all_fields_optional(self):
        assert ReplaySummary.model_validate({}).sessions is None
        assert ReplayThinkPercentiles.model_validate({}).p50 is None

    def test_realistic_v5_summary_parses(self):
        """Realistic real-shape summary block (verified against the real
        producer's Go source — see weka_replay_models module docstring)."""
        raw = {
            "sessions": 42,
            "instances": 103,
            "requests": 512,
            "spawns": 61,
            "matched_spawns": 58,
            "fan_out_turns": 40,
            "max_fan_out_in_one_turn": 5,
            "repaired_parents": 2,
            "cycles_broken": 0,
            "start_ts": "2024-01-01T00:00:00Z",
            "end_ts": "2024-01-01T01:00:00Z",
            "timing": {
                "requests_with_timing": 500,
                "requests_with_think": 300,
                "think_clamped": 4,
                "think_clamped_percent": 1.33,
                "think_ms": {"n": 300, "p50": 120, "p90": 400, "p99": 900, "max": 1200, "mean": 150},
                "think_ms_by_ref": {
                    "prev": {"n": 200, "p50": 100, "p90": 300, "p99": 800, "max": 1000, "mean": 130},
                },
                "total_ms": {"n": 500, "p50": 220, "p90": 600, "p99": 1200, "max": 2000, "mean": 300},
            },
        }  # fmt: skip
        summary = ReplaySummary.model_validate(raw)
        assert summary.sessions == 42
        assert summary.max_fan_out_in_one_turn == 5
        assert summary.start_ts == "2024-01-01T00:00:00Z"
        assert summary.timing.requests_with_timing == 500
        assert summary.timing.think_ms.p99 == 900
        assert summary.timing.think_ms_by_ref["prev"].mean == 130
        assert summary.timing.total_ms.max == 2000


class TestReplaySession:
    def test_minimal_session_parses(self):
        session = ReplaySession.model_validate(build_session())
        assert session.session_id == "sess1"
        assert session.instances is not None
        assert len(session.instances) == 1

    def test_instances_optional_none(self):
        raw = build_session()
        raw["instances"] = None
        session = ReplaySession.model_validate(raw)
        assert session.instances is None

    def test_unknown_top_level_field_rejected(self):
        raw = build_session()
        raw["bogus"] = True
        with pytest.raises(ValidationError):
            ReplaySession.model_validate(raw)

    def test_ts_stays_raw_string(self):
        """Timestamp parsing is deferred to the IR step, not done here."""
        raw = build_session(start_ts="2024-01-01T00:00:00.123456789Z")
        session = ReplaySession.model_validate(raw)
        assert session.start_ts == "2024-01-01T00:00:00.123456789Z"


class TestReplayInstance:
    def test_minimal_instance_parses(self):
        inst = ReplayInstance.model_validate(build_instance())
        assert inst.instance_id == "sess1:sig:fp"
        assert inst.role == "main"
        assert inst.parent_instance_id is None
        assert inst.parent_spawn_request_id is None
        assert inst.fan_out_group is None
        assert inst.fan_out_position is None

    def test_fan_out_fields(self):
        raw = build_instance(
            fan_out_group="5",
            fan_out_size=3,
            fan_out_position=0,
            parent_spawn_request_id=5,
        )
        inst = ReplayInstance.model_validate(raw)
        assert inst.fan_out_group == "5"
        assert inst.fan_out_size == 3
        assert inst.fan_out_position == 0
        assert inst.parent_spawn_request_id == 5

    def test_fan_out_position_omitempty_absent_means_zero_semantically(self):
        """Go's omitempty drops position 0 from the wire; the field is optional
        here precisely so an absent value can be told apart from a real 0
        at the model layer (the IR step maps absence -> 0)."""
        raw = build_instance()
        raw.pop("fan_out_position", None)
        inst = ReplayInstance.model_validate(raw)
        assert inst.fan_out_position is None

    def test_parent_instance_id_parses_but_is_just_data(self):
        inst = ReplayInstance.model_validate(
            build_instance(parent_instance_id="other:sig:fp")
        )
        assert inst.parent_instance_id == "other:sig:fp"

    def test_unknown_top_level_field_rejected(self):
        raw = build_instance()
        raw["bogus"] = True
        with pytest.raises(ValidationError):
            ReplayInstance.model_validate(raw)


class TestReplayRequest:
    def test_minimal_request_parses(self):
        req = ReplayRequest.model_validate(build_request())
        assert req.request_id == 1
        assert req.stream is False
        assert req.system_blocks == []
        assert req.tools is None
        assert req.messages == []
        assert req.upstream_latency_ms == 0.0
        assert req.total_ms == 0.0

    def test_v4_timing_fields(self):
        raw = build_request(
            total_ms=123.4, upstream_latency_ms=12.0, think_ms=5.0, think_ref="prev"
        )
        req = ReplayRequest.model_validate(raw)
        assert req.total_ms == 123.4
        assert req.upstream_latency_ms == 12.0
        assert req.think_ms == 5.0
        assert req.think_ref == "prev"

    def test_think_ref_rejects_unknown_value(self):
        with pytest.raises(ValidationError):
            ReplayRequest.model_validate(build_request(think_ref="sideways"))

    def test_sampling_params_optional_and_real(self):
        raw = build_request(
            temperature=0.7, top_p=0.9, top_k=40, thinking={"budget_tokens": 1024}
        )
        req = ReplayRequest.model_validate(raw)
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 40
        assert req.thinking == {"budget_tokens": 1024}

    def test_a_9_digit_fraction_timestamp_parses_as_raw_string(self):
        raw = build_request(ts="2024-01-01T00:00:01.123456789Z")
        req = ReplayRequest.model_validate(raw)
        assert req.ts == "2024-01-01T00:00:01.123456789Z"

    def test_unknown_top_level_field_rejected(self):
        raw = build_request()
        raw["bogus"] = True
        with pytest.raises(ValidationError):
            ReplayRequest.model_validate(raw)


class TestReplayContentBlocks:
    def test_block_types_null_parses_as_none(self):
        """block_types can be JSON null (a Go nil slice), not just absent."""
        raw = build_request(
            system_blocks=[
                {
                    "type": "text",
                    "hash": "sha256:abc",
                    "bytes": 10,
                    "tokens": 5,
                    "block_types": None,
                }
            ]
        )
        req = ReplayRequest.model_validate(raw)
        assert req.system_blocks[0].block_types is None

    def test_message_block_seed_hash(self):
        block = ReplayMessageBlock.model_validate(
            {
                "hash": "sha256:abc",
                "bytes": 10,
                "tokens": 5,
                "role": "user",
                "seed_hash": "sha256:def",
            }
        )
        assert block.seed_hash == "sha256:def"
        assert block.role == "user"

    def test_tools_block_count(self):
        raw = build_request(
            tools={"hash": "sha256:t", "bytes": 100, "tokens": 20, "count": 4}
        )
        req = ReplayRequest.model_validate(raw)
        assert req.tools.count == 4

    def test_tools_block_count_required(self):
        """The real producer's ReplayToolsSpec.Count has no omitempty, so it
        is always present on a real tools block — count is a required field,
        not defaulted."""
        raw = build_request(tools={"hash": "sha256:t", "bytes": 100, "tokens": 20})
        with pytest.raises(ValidationError):
            ReplayRequest.model_validate(raw)

    def test_unknown_field_rejected(self):
        raw = build_request(
            system_blocks=[
                {
                    "type": "text",
                    "hash": "sha256:abc",
                    "bytes": 10,
                    "tokens": 5,
                    "bogus": True,
                }
            ]
        )
        with pytest.raises(ValidationError):
            ReplayRequest.model_validate(raw)

    def test_system_block_requires_type(self):
        """The real producer's ReplaySystemBlock.Type has no omitempty, so it
        is always present on a real system block — type is required."""
        raw = build_request(
            system_blocks=[{"hash": "sha256:abc", "bytes": 10, "tokens": 5}]
        )
        with pytest.raises(ValidationError):
            ReplayRequest.model_validate(raw)

    def test_system_block_realistic_shape_parses(self):
        """Realistic real-shape system block: no tokens key at all (the
        producer's omitempty drops it whenever the estimate is exactly 0),
        locking in the tokens-defaults-to-0 fix."""
        raw = build_request(
            system_blocks=[
                {"type": "text", "hash": "sha256:abc123def4567890", "bytes": 1024}
            ]
        )
        req = ReplayRequest.model_validate(raw)
        assert req.system_blocks[0].type == "text"
        assert req.system_blocks[0].tokens == 0


class TestReplayIndex:
    def test_minimal_index_parses(self):
        raw = {
            "_schema": "replay-index-v1",
            "sessions": [
                {
                    "session_id": "sess1",
                    "offset": 100,
                    "length": 200,
                    "start_ts": "2024-01-01T00:00:00Z",
                    "end_ts": "2024-01-01T00:00:10Z",
                    "requests": 3,
                }
            ],
        }
        index = ReplayIndex.model_validate(raw)
        assert index.schema_version == "replay-index-v1"
        assert index.sessions[0].session_id == "sess1"
        assert index.sessions[0].offset == 100

    def test_wrong_schema_rejected(self):
        raw = {"_schema": "replay-v5", "sessions": []}
        with pytest.raises(ValidationError):
            ReplayIndex.model_validate(raw)
