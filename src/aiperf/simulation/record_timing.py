# SPDX-License-Identifier: Apache-2.0
"""Validate simulator SSE snapshots before admitting virtual metric samples."""

from typing import Any

import msgspec

from aiperf.common.models import ErrorDetails, RequestRecord, SSEMessage
from aiperf.common.models.server_timing import ServerTiming
from aiperf.simulation.identity import VIRTUAL_TIMESTAMP_ORIGIN_NS
from aiperf.simulation.protocol import SimulationError

TIMING_EVENT = "sglang_simulator"


class TimingSnapshot(msgspec.Struct, frozen=True):
    """Version-one fields; unknown additive fields do not change their meaning."""

    version: int
    rid: str
    arrival_ns: int
    first_token_ns: int | None
    last_token_ns: int | None
    commit_ns: int
    terminal_ns: int | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


def normalize_unavailable_timing(record: RequestRecord) -> None:
    """Keep the raw failure evidence without contributing physical timing."""
    record.error = record.error or ErrorDetails.from_exception(
        SimulationError("simulator generation timing is unavailable")
    )
    record.server_timing = ServerTiming()
    issued = record.request_info.credit_issued_ns if record.request_info else None
    record.timestamp_ns = issued or VIRTUAL_TIMESTAMP_ORIGIN_NS
    record.start_perf_ns = 0
    record.end_perf_ns = None
    record.recv_start_perf_ns = None
    record.cancellation_perf_ns = None
    record.credit_drop_latency = None
    record.trace_data = None
    record._parsed_responses_cache = None
    for response in record.responses:
        response.perf_ns = 0


def _validate_snapshot(
    value: Any, rid: str, previous: TimingSnapshot | None
) -> TimingSnapshot:
    snapshot = msgspec.convert(value, type=TimingSnapshot, strict=True)
    if snapshot.version != 1 or snapshot.rid != rid:
        raise SimulationError("simulator timing version or request ID mismatch")
    first, last = snapshot.first_token_ns, snapshot.last_token_ns
    if not (
        0 <= snapshot.arrival_ns <= snapshot.commit_ns
        and 0 <= snapshot.cached_tokens <= snapshot.prompt_tokens
        and snapshot.completion_tokens >= 0
        and (snapshot.terminal_ns is None or snapshot.terminal_ns == snapshot.commit_ns)
    ):
        raise SimulationError("invalid simulator timing bounds or token counts")
    if snapshot.completion_tokens == 0:
        if first is not None or last is not None:
            raise SimulationError("zero-token snapshot has generation timestamps")
    elif (
        first is None
        or last is None
        or not (snapshot.arrival_ns <= first <= last <= snapshot.commit_ns)
    ):
        raise SimulationError("missing or unordered generation timestamps")
    if snapshot.completion_tokens == 1 and first != last:
        raise SimulationError("single-token snapshot has different generation bounds")
    if previous is not None and (
        previous.terminal_ns is not None
        or snapshot.arrival_ns != previous.arrival_ns
        or snapshot.prompt_tokens != previous.prompt_tokens
        or snapshot.cached_tokens != previous.cached_tokens
        or snapshot.commit_ns < previous.commit_ns
        or snapshot.completion_tokens < previous.completion_tokens
        or (
            snapshot.completion_tokens == previous.completion_tokens
            and snapshot.last_token_ns != previous.last_token_ns
        )
        or (
            previous.first_token_ns is not None
            and snapshot.first_token_ns != previous.first_token_ns
        )
        or (
            previous.last_token_ns is not None
            and (
                snapshot.last_token_ns is None
                or snapshot.last_token_ns < previous.last_token_ns
            )
        )
    ):
        raise SimulationError(
            "simulator timing snapshot moved backwards or changed identity"
        )
    return snapshot


def _validate_usage(usage: Any, snapshot: TimingSnapshot) -> None:
    if not isinstance(usage, dict):
        raise SimulationError("simulator response omitted final usage")
    details = usage.get("prompt_tokens_details")
    actual = (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        details.get("cached_tokens") if isinstance(details, dict) else None,
    )
    expected = (
        snapshot.prompt_tokens,
        snapshot.completion_tokens,
        snapshot.cached_tokens,
    )
    if any(type(count) is not int for count in actual) or actual != expected:
        raise SimulationError(
            "simulator native usage does not match timing token counts"
        )


def _read_snapshots(record: RequestRecord, rid: str) -> TimingSnapshot:
    previous = None
    usage = None
    done = False
    leading = []
    for response in record.responses:
        if not isinstance(response, SSEMessage) or done:
            raise SimulationError("simulator requires a complete SSE response")
        is_timing = any(
            field.name == "event" and field.value == TIMING_EVENT
            for field in response.packets
        )
        data = response.get_json()
        if is_timing:
            previous = _validate_snapshot(data, rid, previous)
            for initial in leading:
                initial.perf_ns = previous.commit_ns
            leading.clear()
        elif response.get_text() == "[DONE]":
            done = True
        elif isinstance(data, dict):
            usage = _read_usage(data, previous) or usage
        if previous is None:
            leading.append(response)
        else:
            response.perf_ns = previous.commit_ns
    if not done or previous is None or previous.terminal_ns is None:
        raise SimulationError("simulator response omitted terminal timing or [DONE]")
    _validate_usage(usage, previous)
    return previous


def _read_usage(data: dict[str, Any], snapshot: TimingSnapshot | None) -> Any:
    if data.get("error"):
        raise SimulationError("simulator returned an SSE error")
    if snapshot is None:
        _validate_initial_role(data)
    if snapshot is not None and snapshot.terminal_ns is not None:
        return data.get("usage")
    return None


def _validate_initial_role(data: dict[str, Any]) -> None:
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise SimulationError("simulator emitted data before its first timing snapshot")
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if (
        not isinstance(delta, dict)
        or delta.get("role") != "assistant"
        or any(value not in (None, "") for key, value in delta.items() if key != "role")
    ):
        raise SimulationError(
            "simulator emitted content before its first timing snapshot"
        )


def apply_simulator_timing(record: RequestRecord, rid: str) -> RequestRecord:
    """Replace physical timestamps only after validating the native response."""
    try:
        if record.error is not None or record.status != 200:
            raise SimulationError(
                "simulator HTTP request did not complete successfully"
            )
        snapshot = _read_snapshots(record, rid)
    except (SimulationError, msgspec.ValidationError, TypeError, ValueError) as exc:
        record.error = record.error or ErrorDetails.from_exception(exc)
        normalize_unavailable_timing(record)
        return record
    record.server_timing = ServerTiming(
        first_token_perf_ns=snapshot.first_token_ns,
        last_token_perf_ns=snapshot.last_token_ns,
    )
    record.timestamp_ns = VIRTUAL_TIMESTAMP_ORIGIN_NS + snapshot.arrival_ns
    record.start_perf_ns = snapshot.arrival_ns
    record.end_perf_ns = snapshot.terminal_ns
    # HTTP establishment and network tracing are physical, outside the model.
    record.recv_start_perf_ns = None
    record.trace_data = None
    record.cancellation_perf_ns = None
    record.credit_drop_latency = None
    record._parsed_responses_cache = None
    return record
