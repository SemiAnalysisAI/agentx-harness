# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""msgpack wire codec for the worker -> record-processor raw-inference channel.

Every ``InferenceResultsMessage`` carries a ``RequestRecord`` whose
``responses`` list holds one ``SSEMessage`` per streamed chunk. On the JSON bus
each chunk is projected through pydantic (``model_dump(mode="json")``), escaped
into a JSON string by orjson, re-parsed by orjson in the record processor, and
re-validated by pydantic against a four-member union. This codec replaces only
that wire step, mirroring the typed msgpack clients
(``streaming_push_client.py`` / ``streaming_pull_client.py``): a pre-built
``msgspec.msgpack.Encoder`` on the send side and a typed
``msgspec.msgpack.Decoder`` on the receive side.

- Streamed chunks travel as the existing ``SSEMessage``/``SSEField`` slots
  dataclasses, which msgspec encodes and decodes natively in C. The decoded
  objects are the same types the JSON path produced, so the record processor,
  its parsers and the raw-record exporter see identical inputs.
- Non-SSE responses (text, binary, or a mixed list) travel as tagged structs
  in their original order and are rebuilt into their original types.
- The per-record remainder (request context, error, trace data, ...) is still
  dumped with ``mode="json"`` exactly as ``Message.to_json_bytes`` does and is
  re-validated by pydantic, so its semantics are unchanged.

The codec is bound to the channel, not chosen by callers: ``WIRE_CODECS`` maps
(client type, address) to codec kwargs, and ``BaseZMQCommunication.create_client``
injects them. Every PUSH client on ``RAW_INFERENCE_PROXY_FRONTEND`` encodes
msgpack and every PULL client on ``RAW_INFERENCE_PROXY_BACKEND`` decodes it, so
JSON and msgpack can never share that socket. A frame from a mismatched peer
fails decoding loudly in either direction (see ``WIRE_TAG``).
"""

from typing import Any, Final

import msgspec
from msgspec import Struct

from aiperf.common.enums import CommAddress
from aiperf.common.messages import InferenceResultsMessage, Message
from aiperf.common.models.record_models import (
    BinaryResponse,
    RequestRecord,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.plugin.enums import CommClientType

WIRE_TAG: Final[str] = "aiperf.inference_results.v1"
"""Frame tag. Bump the version whenever the frame layout changes.

A JSON frame starts with ``{`` (0x7B), which msgpack reads as the positive
fixint 123, so the typed decoder rejects it before reading a field. A msgpack
frame starts with a fixmap byte (0x80-0x8F), which orjson rejects."""


class _SSEWire(Struct, tag="sse", array_like=True, frozen=True):
    """An ``SSEMessage`` inside a non-homogeneous responses list."""

    perf_ns: int
    packets: list[SSEField]


class _TextWire(Struct, tag="text", array_like=True, frozen=True):
    perf_ns: int
    text: str
    content_type: str | None


class _BinaryWire(Struct, tag="binary", array_like=True, frozen=True):
    perf_ns: int
    raw_bytes: bytes
    content_type: str | None


_ResponseWire = _SSEWire | _TextWire | _BinaryWire


class InferenceResultsFrame(
    Struct, tag=WIRE_TAG, forbid_unknown_fields=True, omit_defaults=True
):
    """One ``InferenceResultsMessage`` on the raw-inference channel.

    Exactly one of ``sse`` and ``other`` is non-empty, unless the record has no
    responses at all (an error record), in which case both are empty.
    """

    service_id: str
    record: dict[str, Any]
    """``RequestRecord`` minus ``responses``, dumped like ``Message.to_json_bytes``."""
    request_ns: int | None = None
    request_id: str | None = None
    sse: list[SSEMessage] = []
    """Fast path: every response is an ``SSEMessage`` (all streamed records)."""
    other: list[_ResponseWire] = []
    """Fallback: any other response list, order preserved."""


# No enc_hook: the frame holds only builtins (the record remainder is already
# dumped with mode="json") and the SSE/text/binary slots types msgspec encodes
# natively, matching the other msgpack clients in this package.
_encoder: Final = msgspec.msgpack.Encoder()
_decoder: Final = msgspec.msgpack.Decoder(InferenceResultsFrame)

_RECORD_DUMP_EXCLUDE: Final[set[str]] = {"responses"}
_RECORD_DUMP_CONTEXT: Final[dict[str, bool]] = {"include_internal": True}


def _to_wire(response: Any) -> _ResponseWire:
    response_type = type(response)
    if response_type is SSEMessage:
        return _SSEWire(response.perf_ns, response.packets)
    if response_type is TextResponse:
        return _TextWire(response.perf_ns, response.text, response.content_type)
    if response_type is BinaryResponse:
        return _BinaryWire(response.perf_ns, response.raw_bytes, response.content_type)
    raise TypeError(
        f"Cannot encode response of type {response_type.__name__} on the "
        "raw-inference channel"
    )


def _from_wire(
    wire: _ResponseWire,
) -> SSEMessage | TextResponse | BinaryResponse:
    wire_type = type(wire)
    if wire_type is _SSEWire:
        return SSEMessage(perf_ns=wire.perf_ns, packets=wire.packets)
    if wire_type is _TextWire:
        return TextResponse(
            perf_ns=wire.perf_ns, text=wire.text, content_type=wire.content_type
        )
    return BinaryResponse(
        perf_ns=wire.perf_ns,
        raw_bytes=wire.raw_bytes,
        content_type=wire.content_type,
    )


def encode_inference_results(message: Message) -> bytes:
    """Encode an ``InferenceResultsMessage`` as one msgpack frame.

    Raises:
        TypeError: ``message`` is not exactly an ``InferenceResultsMessage``, it
            carries undeclared extra fields, or a response has an unknown type.
            The channel carries nothing else, so anything unexpected fails here
            rather than being dropped silently.
    """
    if type(message) is not InferenceResultsMessage:
        raise TypeError(
            "The raw-inference channel carries only InferenceResultsMessage, "
            f"got {type(message).__name__}"
        )
    if message.__pydantic_extra__:
        raise TypeError(
            "InferenceResultsMessage extra fields are not carried on the "
            f"raw-inference channel: {sorted(message.__pydantic_extra__)}"
        )
    record = message.record
    responses = record.responses or []
    record_fields = record.model_dump(
        mode="json",
        exclude_none=True,
        exclude=_RECORD_DUMP_EXCLUDE,
        context=_RECORD_DUMP_CONTEXT,
    )
    if all(type(response) is SSEMessage for response in responses):
        frame = InferenceResultsFrame(
            service_id=message.service_id,
            record=record_fields,
            request_ns=message.request_ns,
            request_id=message.request_id,
            sse=responses,
        )
    else:
        frame = InferenceResultsFrame(
            service_id=message.service_id,
            record=record_fields,
            request_ns=message.request_ns,
            request_id=message.request_id,
            other=[_to_wire(response) for response in responses],
        )
    return _encoder.encode(frame)


def decode_inference_results(data: bytes) -> InferenceResultsMessage:
    """Decode one msgpack frame back into an ``InferenceResultsMessage``.

    The record remainder is validated by pydantic exactly as on the JSON path;
    the responses are already typed by msgspec, so they skip pydantic.

    Raises:
        msgspec.DecodeError: the frame is not a valid ``InferenceResultsFrame``
            (including a JSON frame or a frame with a different ``WIRE_TAG``).
        ValueError: the frame carries both ``sse`` and ``other`` responses.
    """
    frame = _decoder.decode(data)
    if frame.sse and frame.other:
        raise ValueError(
            "Malformed raw-inference frame: both sse and other responses are set"
        )
    record = RequestRecord.model_validate(frame.record)
    # Plain attribute assignment: RequestRecord does not validate assignment
    # (the record processor itself assigns ``record.responses = None``).
    record.responses = (
        frame.sse if frame.sse else [_from_wire(wire) for wire in frame.other]
    )
    return InferenceResultsMessage.model_construct(
        service_id=frame.service_id,
        record=record,
        request_ns=frame.request_ns,
        request_id=frame.request_id,
    )


WIRE_CODECS: Final[dict[tuple[CommClientType, CommAddress], dict[str, Any]]] = {
    (CommClientType.PUSH, CommAddress.RAW_INFERENCE_PROXY_FRONTEND): {
        "encode": encode_inference_results,
    },
    (CommClientType.PULL, CommAddress.RAW_INFERENCE_PROXY_BACKEND): {
        "decode": decode_inference_results,
    },
}
"""Channel-bound wire codecs, injected by ``BaseZMQCommunication.create_client``."""
