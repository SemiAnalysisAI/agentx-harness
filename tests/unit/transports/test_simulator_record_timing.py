# SPDX-License-Identifier: Apache-2.0
"""Native generation bounds must be independent of physical SSE coalescing."""

from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
    SSEMessage,
    TextResponseData,
)
from aiperf.common.models.record_models import TokenCounts
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.decode_duration_metric import FullDecodeDurationMetric
from aiperf.metrics.types.e2e_output_throughput_metric import (
    E2EOutputTokenThroughputMetric,
)
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from aiperf.metrics.types.inter_token_latency_metric import (
    FullResponseInterTokenLatencyMetric,
    InterTokenLatencyMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.time_to_first_output_token_metric import (
    TimeToFirstOutputTokenMetric,
)
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.metrics.types.ttst_metric import TTSTMetric
from aiperf.simulation.identity import VIRTUAL_TIMESTAMP_ORIGIN_NS
from aiperf.simulation.record_timing import apply_simulator_timing
from tests.unit.metrics.conftest import run_simple_metrics_pipeline


def snapshot(**updates: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "rid": "request",
        "arrival_ns": 0,
        "first_token_ns": 10,
        "last_token_ns": 30,
        "commit_ns": 30,
        "terminal_ns": 30,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cached_tokens": 2,
        **updates,
    }


def sse(data: Any, event: str | None = None, perf_ns: int = 999999) -> SSEMessage:
    prefix = f"event: {event}\n" if event else ""
    body = data if isinstance(data, str) else orjson.dumps(data).decode()
    return SSEMessage.parse(f"{prefix}data: {body}", perf_ns)


def timing(value: dict[str, Any]) -> SSEMessage:
    return sse(value, "sglang_simulator")


def usage(**updates: Any) -> SSEMessage:
    return sse(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 2},
                **updates,
            },
        }
    )


def record_with(snapshot_value: dict[str, Any] | None = None) -> RequestRecord:
    return RequestRecord(
        status=200,
        responses=[
            sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            timing(snapshot() if snapshot_value is None else snapshot_value),
            sse({"choices": [{"delta": {"content": "abc"}}]}),
            usage(),
            sse("[DONE]"),
        ],
    )


def parsed(record: RequestRecord, timestamps: list[int]) -> ParsedResponseRecord:
    return ParsedResponseRecord(
        request=record,
        responses=[
            ParsedResponse(perf_ns=ts, data=TextResponseData(text="abc"))
            for ts in timestamps
        ],
    )


@pytest.mark.parametrize("timestamps", [[30], [10, 30], [10, 20, 30]])
def test_generation_metrics_do_not_depend_on_content_chunking(
    timestamps: list[int],
) -> None:
    record = apply_simulator_timing(record_with(), "request")
    result = parsed(record, timestamps)
    metrics = MetricRecordDict()
    assert TTFTMetric()._parse_record(result, metrics) == 10
    assert RequestLatencyMetric()._parse_record(result, metrics) == 30
    assert FullDecodeDurationMetric()._parse_record(result, metrics) == 20
    assert record.timestamp_ns == VIRTUAL_TIMESTAMP_ORIGIN_NS
    assert record.start_perf_ns == 0 and record.end_perf_ns == 30
    assert record.recv_start_perf_ns is None and record.trace_data is None
    assert all(response.perf_ns == 30 for response in record.responses)
    assert result.valid


@pytest.mark.parametrize(
    "metric", [TTSTMetric, InterChunkLatencyMetric, TimeToFirstOutputTokenMetric]
)
def test_chunk_dependent_metrics_are_unavailable(metric: type) -> None:
    record = apply_simulator_timing(record_with(), "request")
    with pytest.raises(NoMetricValue):
        metric()._parse_record(parsed(record, [10, 20, 30]), MetricRecordDict())


@pytest.mark.parametrize(
    "updates",
    [
        param({"version": 2}, id="version"),
        param({"rid": "other"}, id="request-id"),
        param({"arrival_ns": True}, id="boolean"),
        param({"first_token_ns": None}, id="missing-first"),
        param({"first_token_ns": 31}, id="unordered"),
        param({"terminal_ns": 31}, id="terminal-boundary"),
        param({"completion_tokens": -1}, id="negative-count"),
        param({"cached_tokens": 8}, id="cache-count"),
        param({"completion_tokens": 1}, id="one-token-bounds"),
    ],
)  # fmt: skip
def test_invalid_snapshot_never_leaks_physical_timing(updates: dict[str, Any]) -> None:
    record = record_with(snapshot(**updates))
    raw = [response.get_raw() for response in record.responses]
    apply_simulator_timing(record, "request")
    assert record.error is not None and record.server_timing is not None
    assert record.server_timing.first_token_perf_ns is None
    assert record.end_perf_ns is None
    assert all(response.perf_ns == 0 for response in record.responses)
    assert [response.get_raw() for response in record.responses] == raw
    for metric in (TTFTMetric(), RequestLatencyMetric(), FullDecodeDurationMetric()):
        with pytest.raises(NoMetricValue):
            metric._parse_record(parsed(record, [999999]), MetricRecordDict())


@pytest.mark.parametrize(
    "missing", ["version", "first_token_ns", "terminal_ns", "cached_tokens"]
)
def test_all_version_one_fields_are_required(missing: str) -> None:
    value = snapshot()
    del value[missing]
    assert apply_simulator_timing(record_with(value), "request").error is not None


def test_additive_unknown_fields_do_not_change_version_one_meaning() -> None:
    assert (
        apply_simulator_timing(record_with(snapshot(future="unused")), "request").error
        is None
    )


@pytest.mark.parametrize("missing", [1, 3, 4])
def test_missing_metadata_usage_or_done_rejects_record(missing: int) -> None:
    record = record_with()
    del record.responses[missing]
    assert apply_simulator_timing(record, "request").error is not None


def test_final_usage_counts_must_match_native_snapshot() -> None:
    record = record_with()
    record.responses[3] = usage(completion_tokens=8)
    assert apply_simulator_timing(record, "request").error is not None


def test_usage_before_terminal_snapshot_is_not_final_usage() -> None:
    record = record_with()
    record.responses.insert(
        1, timing(snapshot(terminal_ns=None, completion_tokens=1, last_token_ns=10))
    )
    record.responses.insert(2, record.responses.pop(4))
    assert apply_simulator_timing(record, "request").error is not None


def test_continuous_usage_does_not_replace_final_usage() -> None:
    record = record_with()
    record.responses.insert(
        1, timing(snapshot(terminal_ns=None, completion_tokens=1, last_token_ns=10))
    )
    record.responses.insert(2, usage(completion_tokens=1))
    assert apply_simulator_timing(record, "request").error is None


@pytest.mark.parametrize(
    "delta", [{"content": "abc"}, {"role": "assistant", "reasoning_content": "think"}]
)
def test_only_empty_role_chunk_may_precede_first_snapshot(
    delta: dict[str, Any],
) -> None:
    record = record_with()
    record.responses[0] = sse({"choices": [{"delta": delta}]})
    assert apply_simulator_timing(record, "request").error is not None


@pytest.mark.parametrize("reasoning_content", [None, ""])
def test_native_empty_optional_fields_may_precede_first_snapshot(
    reasoning_content: str | None,
) -> None:
    record = record_with()
    record.responses[0] = sse(
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": reasoning_content,
                        "role": "assistant",
                        "content": "",
                    }
                }
            ]
        }
    )
    assert apply_simulator_timing(record, "request").error is None


def test_physical_transport_error_keeps_body_and_error_without_timing() -> None:
    record = record_with()
    record.error = ErrorDetails(message="HTTP failed")
    error = record.error
    apply_simulator_timing(record, "request")
    assert record.error is error
    assert record.server_timing is not None and record.end_perf_ns is None
    assert len(record.responses) == 5


def test_snapshot_regression_is_rejected() -> None:
    record = record_with()
    record.responses.insert(
        1,
        timing(
            snapshot(
                terminal_ns=None, last_token_ns=20, commit_ns=20, completion_tokens=2
            )
        ),
    )
    record.responses[2] = timing(snapshot(first_token_ns=11))
    assert apply_simulator_timing(record, "request").error is not None


def test_incremental_snapshot_stamps_only_following_responses() -> None:
    record = record_with()
    record.responses.insert(
        1,
        timing(
            snapshot(
                terminal_ns=None, last_token_ns=10, commit_ns=10, completion_tokens=1
            )
        ),
    )
    record.responses.insert(2, sse({"choices": [{"delta": {"content": "a"}}]}))
    apply_simulator_timing(record, "request")
    assert record.error is None
    assert [response.perf_ns for response in record.responses] == [
        10,
        10,
        10,
        30,
        30,
        30,
        30,
    ]


def test_timing_survives_record_serialization() -> None:
    record = apply_simulator_timing(record_with(), "request")
    restored = RequestRecord.model_validate_json(record.model_dump_json())
    assert restored.server_timing == record.server_timing
    assert TTFTMetric()._parse_record(parsed(restored, [30]), MetricRecordDict()) == 10


def test_original_metric_pipeline_uses_generation_bounds_and_native_counts() -> None:
    result = parsed(apply_simulator_timing(record_with(), "request"), [30])
    result.token_counts = TokenCounts(input=7, output=3)
    metrics = run_simple_metrics_pipeline(
        [result],
        InterTokenLatencyMetric.tag,
        FullResponseInterTokenLatencyMetric.tag,
        E2EOutputTokenThroughputMetric.tag,
    )
    assert metrics[InterTokenLatencyMetric.tag] == [10]
    assert metrics[FullResponseInterTokenLatencyMetric.tag] == [10]
    assert metrics[E2EOutputTokenThroughputMetric.tag] == pytest.approx([100_000_000])


def test_zero_generated_tokens_have_no_generation_metrics() -> None:
    record = record_with(
        snapshot(first_token_ns=None, last_token_ns=None, completion_tokens=0)
    )
    record.responses[3] = usage(completion_tokens=0)
    apply_simulator_timing(record, "request")
    assert record.error is None
    assert record.server_timing.first_token_perf_ns is None
    result = parsed(record, [])
    assert not result.valid
    with pytest.raises(NoMetricValue):
        TTFTMetric()._parse_record(result, MetricRecordDict())


def test_warmup_and_profile_records_share_one_timestamp_origin() -> None:
    record = record_with(
        snapshot(
            arrival_ns=100,
            first_token_ns=110,
            last_token_ns=130,
            commit_ns=130,
            terminal_ns=130,
        )
    )
    apply_simulator_timing(record, "request")
    assert record.timestamp_ns == VIRTUAL_TIMESTAMP_ORIGIN_NS + 100
    assert TTFTMetric()._parse_record(parsed(record, [130]), MetricRecordDict()) == 10
