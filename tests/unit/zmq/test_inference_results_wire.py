# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the msgpack wire codec on the worker -> record-processor channel.

The JSON bus (``Message.to_json_bytes`` -> ``Message.from_json``) is the oracle:
every record must decode to the same message the JSON path produced.
"""

from typing import Any

import msgspec
import pytest

from aiperf.common.enums import CreditPhase, LifecycleState
from aiperf.common.messages import HeartbeatMessage, InferenceResultsMessage, Message
from aiperf.common.models import ErrorDetails, RequestRecord
from aiperf.common.models.record_models import (
    BinaryResponse,
    RecordContext,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.trace_models import AioHttpTraceData
from aiperf.zmq.inference_results_wire import (
    WIRE_TAG,
    decode_inference_results,
    encode_inference_results,
)

_CHUNK = (
    '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1762560000,'
    '"model":"m","choices":[{"index":0,"delta":{"content":" the"},'
    '"logprobs":null,"finish_reason":null}]}'
)


def _message(responses: list[Any], error: ErrorDetails | None = None) -> Any:
    record = RequestRecord(
        request_info=RecordContext(
            credit_num=7,
            credit_phase=CreditPhase.PROFILING,
            conversation_id="conv-1",
            turn_index=2,
            x_request_id="req-1",
            x_correlation_id="corr-1",
            payload_bytes=b'{"model":"m"}',
            max_tokens=64,
        ),
        model_name="m",
        start_perf_ns=1_000,
        timestamp_ns=1_700_000_000_000_000_000,
        end_perf_ns=9_000,
        status=200,
        responses=responses,
        error=error,
        trace_data=AioHttpTraceData(
            response_status_code=200, response_chunks=[(1_100, 64), (1_200, 64)]
        ),
        clock_offset_ns=-42,
    )
    return InferenceResultsMessage(service_id="worker-1", record=record)


def _assert_matches_json_path(message: Any) -> InferenceResultsMessage:
    expected = Message.from_json(message.to_json_bytes())
    actual = decode_inference_results(encode_inference_results(message))

    assert type(actual) is InferenceResultsMessage
    assert actual.message_type == expected.message_type
    assert actual.service_id == expected.service_id
    assert actual.request_ns == expected.request_ns
    assert actual.request_id == expected.request_id
    assert actual.record.model_dump(
        mode="json", exclude={"responses"}
    ) == expected.record.model_dump(mode="json", exclude={"responses"})
    assert [type(r) for r in actual.record.responses] == [
        type(r) for r in expected.record.responses
    ]
    assert actual.record.responses == expected.record.responses
    assert type(actual.record.request_info) is RecordContext
    assert type(actual.record.trace_data) is type(expected.record.trace_data)
    return actual


def test_streamed_sse_record_matches_json_path() -> None:
    responses = [
        SSEMessage(perf_ns=2_000 + i, packets=[SSEField(name="data", value=_CHUNK)])
        for i in range(200)
    ]
    responses.append(
        SSEMessage(perf_ns=5_000, packets=[SSEField(name="data", value="[DONE]")])
    )
    _assert_matches_json_path(_message(responses))


def test_multi_field_sse_message_keeps_every_packet() -> None:
    message = _message(
        [
            SSEMessage(
                perf_ns=2_000,
                packets=[
                    SSEField(name="event", value="error"),
                    SSEField(name="data", value='{"error":"x"}'),
                    SSEField(name="comment", value=None),
                ],
            )
        ]
    )
    actual = _assert_matches_json_path(message)
    assert [p.name for p in actual.record.responses[0].packets] == [
        "event",
        "data",
        "comment",
    ]


def test_mixed_responses_keep_order_and_types() -> None:
    responses = [
        TextResponse(
            perf_ns=2_000, text='{"id":"job"}', content_type="application/json"
        ),
        SSEMessage(perf_ns=2_500, packets=[SSEField(name="data", value=_CHUNK)]),
        BinaryResponse(
            perf_ns=4_000, raw_bytes=b"\x00\xff\x10", content_type="video/mp4"
        ),
    ]
    actual = _assert_matches_json_path(_message(responses))
    assert actual.record.responses[2].raw_bytes == b"\x00\xff\x10"


def test_error_record_without_responses_matches_json_path() -> None:
    message = _message([], error=ErrorDetails(code=500, type="Boom", message="boom"))
    actual = _assert_matches_json_path(message)
    assert actual.record.responses == []
    assert actual.record.error == message.record.error


def test_json_frame_is_rejected_by_msgpack_decoder() -> None:
    message = _message([SSEMessage(perf_ns=2_000, packets=[SSEField("data", _CHUNK)])])
    with pytest.raises(msgspec.DecodeError):
        decode_inference_results(message.to_json_bytes())


def test_msgpack_frame_is_rejected_by_json_decoder() -> None:
    message = _message([SSEMessage(perf_ns=2_000, packets=[SSEField("data", _CHUNK)])])
    with pytest.raises(ValueError):
        Message.from_json(encode_inference_results(message))


def test_other_frame_tag_is_rejected() -> None:
    frame = msgspec.msgpack.encode(
        {"type": WIRE_TAG.replace("v1", "v0"), "service_id": "w", "record": {}}
    )
    with pytest.raises(msgspec.ValidationError):
        decode_inference_results(frame)


def test_frame_with_both_response_lists_is_rejected() -> None:
    frame = msgspec.msgpack.encode(
        {
            "type": WIRE_TAG,
            "service_id": "w",
            "record": {"start_perf_ns": 1, "timestamp_ns": 1},
            "sse": [{"perf_ns": 1, "packets": []}],
            "other": [["text", 1, "x", None]],
        }
    )
    with pytest.raises(ValueError, match="both sse and other"):
        decode_inference_results(frame)


def test_other_message_types_are_rejected() -> None:
    heartbeat = HeartbeatMessage(
        service_id="s",
        state=LifecycleState.RUNNING,
        service_type="test",
        request_id="r",
    )
    with pytest.raises(TypeError, match="only InferenceResultsMessage"):
        encode_inference_results(heartbeat)


def test_unknown_response_type_is_rejected() -> None:
    message = _message([])
    # RequestRecord does not validate assignment, so a foreign object can reach
    # the encoder; it must fail loudly instead of being dropped.
    message.record.responses.append(object())
    with pytest.raises(TypeError, match="Cannot encode response of type object"):
        encode_inference_results(message)
