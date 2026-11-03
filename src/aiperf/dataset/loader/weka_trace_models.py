# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Weka KV-cache-tester agentic coding trace format.

See ``artifacts/kv-cache-tester/docs/trace_replay_tester.md`` for the source
format. Each trace file is a single JSON object; ``requests`` is an ordered
list interleaving normal API calls (``type: "n"``), streaming API calls
(``type: "s"``), and subagent markers (``type: "subagent"``) with their own
nested request lists.
"""

from __future__ import annotations

from array import array
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AfterValidator, ConfigDict, Field, PlainSerializer

from aiperf.common.models import AIPerfBaseModel


class HashIds(array):
    """KV-cache block hash IDs packed as int64 instead of boxed Python ints.

    A trace holds ~200k hash IDs; as a ``list[int]`` that is ~7 MB per trace
    (4 GB for the 393-trace corpus held by the dataset manager), as this array
    ~1.5 MB. It is a read-only sequence to every consumer (iteration, index,
    slice, ``len``, membership, ``list(...)``), compares equal to a list with
    the same values, and serializes back to a plain list.
    """

    def __new__(cls, values: Any = ()) -> HashIds:
        return super().__new__(cls, "q", values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, array):
            return array.__eq__(self, other)
        if isinstance(other, (list, tuple)):
            return len(self) == len(other) and all(
                a == b for a, b in zip(self, other, strict=True)
            )
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = None  # type: ignore[assignment]

    def __reduce__(self) -> tuple:
        return (HashIds, (list(self),))


def _pack_hash_ids(values: list[int]) -> list[int] | HashIds:
    try:
        return HashIds(values)
    except OverflowError:
        # Outside int64: keep the list so the values are untouched.
        return values


HashIdList: TypeAlias = Annotated[
    list[int],
    AfterValidator(_pack_hash_ids),
    PlainSerializer(list, return_type=list[int]),
]


class WekaNormalRequest(AIPerfBaseModel):
    """One normal (``type: "n"``) API call in a Weka trace."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    t: float = Field(
        description="Request timestamp in seconds from conversation start."
    )
    type: Literal["n"] = Field(description="Discriminator: normal API call.")
    model: str = Field(description="Model identifier for this request.")
    input_length: int = Field(alias="in", description="Input token count.")
    output_length: int = Field(alias="out", description="Output token count.")
    hash_ids: HashIdList = Field(
        default_factory=list, description="KV-cache block hash IDs."
    )
    input_types: list[str] = Field(
        default_factory=list, description="Content-type annotations for input."
    )
    output_types: list[str] = Field(
        default_factory=list, description="Content-type annotations for output."
    )
    stop: str = Field(
        default="", description="Stop reason: '', 'tool_use', 'end_turn'."
    )
    api_time: float | None = Field(
        default=None, description="Server processing time in seconds."
    )
    think_time: float | None = Field(
        default=None, description="Client delay in seconds before this request."
    )


class WekaStreamingRequest(AIPerfBaseModel):
    """One streaming (``type: "s"``) API call in a Weka trace.

    Structurally identical to :class:`WekaNormalRequest` except for the
    discriminator value and an optional ``ttft`` field (recorded
    time-to-first-token in seconds).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    t: float = Field(
        description="Request timestamp in seconds from conversation start."
    )
    type: Literal["s"] = Field(description="Discriminator: streaming API call.")
    model: str = Field(description="Model identifier for this request.")
    input_length: int = Field(alias="in", description="Input token count.")
    output_length: int = Field(alias="out", description="Output token count.")
    hash_ids: HashIdList = Field(
        default_factory=list, description="KV-cache block hash IDs."
    )
    input_types: list[str] = Field(
        default_factory=list, description="Content-type annotations for input."
    )
    output_types: list[str] = Field(
        default_factory=list, description="Content-type annotations for output."
    )
    stop: str = Field(
        default="", description="Stop reason: '', 'tool_use', 'end_turn'."
    )
    api_time: float | None = Field(
        default=None, description="Server processing time in seconds."
    )
    think_time: float | None = Field(
        default=None, description="Client delay in seconds before this request."
    )
    ttft: float | None = Field(
        default=None, description="Recorded time-to-first-token in seconds."
    )


class WekaSubagentEntry(AIPerfBaseModel):
    """A ``type: "subagent"`` marker with its nested inner requests.

    The parent's next ``WekaNormalRequest`` in the outer list is understood
    to occur after this subagent completes (spec §4.4).
    """

    model_config = ConfigDict(extra="forbid")

    t: float = Field(description="Spawn timestamp in seconds from conversation start.")
    type: Literal["subagent"] = Field(description="Discriminator: subagent marker.")
    agent_id: str = Field(description="Opaque subagent identifier, e.g. 'agent_001'.")
    subagent_type: str = Field(description="Subagent type, e.g. 'Explore'.")
    duration_ms: int | None = Field(
        default=None,
        description="Wall-clock duration of the subagent. None for subagents "
        "with status='async_launched' (telemetry not captured).",
    )
    total_tokens: int | None = Field(
        default=None,
        description="Total tokens across all subagent inner requests. None "
        "for status='async_launched'.",
    )
    tool_use_count: int | None = Field(
        default=None,
        description="Tool calls made by the subagent. None for "
        "status='async_launched'.",
    )
    status: str = Field(description="'completed' or other terminal status.")
    requests: list[
        Annotated[WekaNormalRequest | WekaStreamingRequest, Field(discriminator="type")]
    ] = Field(
        description="Inner requests of the subagent (normal or streaming API calls)."
    )
    models: list[str] = Field(description="Models used by the subagent.")
    tool_tokens: int = Field(
        default=0, description="Subagent's tools prefix token count."
    )
    system_tokens: int = Field(
        default=0, description="Subagent's system prefix token count."
    )


WekaRequest: TypeAlias = Annotated[
    WekaNormalRequest | WekaStreamingRequest | WekaSubagentEntry,
    Field(discriminator="type"),
]


class WekaTrace(AIPerfBaseModel):
    """A single Weka trace file."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Trace identifier (session ID).")
    models: list[str] = Field(description="Models used in the trace.")
    block_size: int = Field(gt=0, description="Cache block size in tokens.")
    hash_id_scope: Literal["local"] = Field(
        description=(
            "Hash ID namespace scope. v1 loader only supports 'local' scope "
            "(hashes scoped per-trace); 'global' scope (cross-trace KV-cache "
            "sharing) would require synthesis-time coordination across files "
            "and is rejected at schema level until implemented."
        )
    )
    tool_tokens: int = Field(default=0, description="Tools prefix token count.")
    system_tokens: int = Field(default=0, description="System prefix token count.")
    requests: list[WekaRequest] = Field(
        description="Interleaved normal requests and subagent markers."
    )
    totals: dict[str, Any] | None = Field(
        default=None, description="Optional trace-level summary; opaque."
    )
