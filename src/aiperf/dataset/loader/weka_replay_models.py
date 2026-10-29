# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic schema for the ``wekai`` router's ``replay-v3``/``v4``/``v5`` capture format.

One model set covers all three schema versions: v4/v5-only fields are optional
and simply absent on v3 input. ``extra="forbid"`` is used on every model whose
fields carry linkage/structural semantics (header, session, instance, request,
content blocks, v5 index) so an unrecognized field fails loudly at load time
rather than being silently ignored — this format's tree reconstruction depends
on knowing every structural field that exists.

Deviation from that rule, deliberately: ``ReplaySummary``/``ReplayTimingSummary``/
``ReplayThinkPercentiles`` (the header's informational counts/percentiles
block) are left permissive (inheriting :class:`AIPerfBaseModel`'s own
``extra="allow"``) rather than ``extra="forbid"``. Their field names are now
verified against the real producer's Go source (``ReplaySummary`` in
``command_router_replay_prepare.go``; ``ReplayTimingSummary`` and
``ThinkPercentiles`` in ``replay_think.go``), but ``extra="allow"`` is kept
anyway: the summary block carries no linkage semantics — nothing here is used
for tree/causal reconstruction — so staying lenient about a field the verified
structs don't cover (e.g. one added to a real v5 header after this check)
doesn't compromise the reason the rest of this schema is strict.

This module is pure schema: no parsing/IO logic, no timestamp parsing (``ts``/
``start_ts`` remain raw strings; parsing into epoch-ms happens in
``_weka_replay_ir.py``, since pydantic's datetime parser isn't guaranteed to
accept every RFC3339Nano shape this format emits).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from aiperf.common.models import AIPerfBaseModel


class ReplayThinkPercentiles(AIPerfBaseModel):
    """One percentile-stats block (sample count plus p50/p90/p99/max/mean).

    Verified against the real producer's ``ThinkPercentiles`` Go struct
    (``replay_think.go``): reused for both the ``think_ms``/``think_ms_by_ref``
    and ``total_ms`` stats nested in :class:`ReplayTimingSummary`. All values
    are plain integer milliseconds, not floats, and there is deliberately no
    ``p95`` — the real struct only has p50/p90/p99.
    """

    n: int | None = Field(default=None, ge=0, description="Number of samples.")
    p50: int | None = Field(default=None, ge=0, description="p50 in milliseconds.")
    p90: int | None = Field(default=None, ge=0, description="p90 in milliseconds.")
    p99: int | None = Field(default=None, ge=0, description="p99 in milliseconds.")
    max: int | None = Field(default=None, ge=0, description="Maximum in milliseconds.")
    mean: int | None = Field(default=None, ge=0, description="Mean in milliseconds.")


class ReplayTimingSummary(AIPerfBaseModel):
    """v4+'s nested ``summary.timing`` block.

    Verified against the real producer's ``ReplayTimingSummary`` Go struct
    (``replay_think.go``). Permissive (``extra="allow"``, inherited from
    :class:`AIPerfBaseModel`) — see the module docstring.
    """

    requests_with_timing: int | None = Field(
        default=None,
        ge=0,
        description="Number of requests in the capture that carry real timing data.",
    )
    requests_with_think: int | None = Field(
        default=None,
        ge=0,
        description="Number of requests in the capture that carry a think-time measurement.",
    )
    think_clamped: int | None = Field(
        default=None,
        ge=0,
        description="Number of think-time measurements that were clamped.",
    )
    think_clamped_percent: float | None = Field(
        default=None,
        ge=0,
        description="Percentage of think-time measurements that were clamped.",
    )
    think_ms: ReplayThinkPercentiles | None = Field(
        default=None, description="Think-time percentiles across the capture."
    )
    think_ms_by_ref: dict[str, ReplayThinkPercentiles] | None = Field(
        default=None,
        description="Think-time percentiles broken down by think_ref "
        "('prev'/'spawn'/'session_start').",
    )
    total_ms: ReplayThinkPercentiles | None = Field(
        default=None,
        description="Total request-duration percentiles across the capture.",
    )


class ReplaySummary(AIPerfBaseModel):
    """Header's ``summary`` block: corpus-level counts plus, on v4+, timing percentiles.

    Verified against the real producer's ``ReplaySummary`` Go struct
    (``command_router_replay_prepare.go``). Permissive (``extra="allow"``,
    inherited from :class:`AIPerfBaseModel`) — see the module docstring.
    """

    sessions: int | None = Field(
        default=None, ge=0, description="Total number of sessions in this capture."
    )
    instances: int | None = Field(
        default=None,
        ge=0,
        description="Total number of instances across all sessions.",
    )
    requests: int | None = Field(
        default=None, ge=0, description="Total number of requests across all instances."
    )
    spawns: int | None = Field(
        default=None, ge=0, description="Total number of spawn events in this capture."
    )
    matched_spawns: int | None = Field(
        default=None,
        ge=0,
        description="Number of spawn events successfully matched to a spawning turn.",
    )
    fan_out_turns: int | None = Field(
        default=None,
        ge=0,
        description="Number of turns that spawned at least one sub-agent.",
    )
    max_fan_out_in_one_turn: int | None = Field(
        default=None,
        ge=0,
        description="Largest number of sub-agents spawned from a single turn.",
    )
    repaired_parents: int | None = Field(
        default=None,
        ge=0,
        description="Number of instances whose parent linkage was repaired by the producer.",
    )
    cycles_broken: int | None = Field(
        default=None,
        ge=0,
        description="Number of causal-link cycles the producer broke while building this capture.",
    )
    start_ts: str | None = Field(
        default=None,
        description="RFC3339Nano timestamp of the earliest event in this capture "
        "(raw string; not parsed at this layer).",
    )
    end_ts: str | None = Field(
        default=None,
        description="RFC3339Nano timestamp of the latest event in this capture "
        "(raw string; not parsed at this layer).",
    )
    timing: ReplayTimingSummary | None = Field(
        default=None,
        description="v4+ only: request/think-time percentiles across the capture.",
    )


class ReplayHeader(AIPerfBaseModel):
    """Line 1 of a replay-vN capture: file-level metadata."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["replay-v3", "replay-v4", "replay-v5"] = Field(
        alias="_schema",
        description="Capture format version: 'replay-v3', 'replay-v4', or 'replay-v5'.",
    )
    name: str = Field(description="Human-readable name of this capture.")
    generated_at: str = Field(
        description="RFC3339Nano timestamp of when this capture was generated "
        "(raw string; not parsed at this layer)."
    )
    source: str = Field(
        description="Identifier of the system/process that produced this capture."
    )
    summary: ReplaySummary = Field(
        description="Corpus-level counts and (v4+) timing percentiles."
    )


class ReplayBlockBase(AIPerfBaseModel):
    """Fields shared by every content block (system block, tools block, message).

    ``hash`` is a 64-bit truncated sha256 (``"sha256:" + 16 hex chars``)
    computed over canonical JSON *including* ``cache_control`` — this is the
    source of the cache-control churn problem handled in a later layer, not
    this one. ``tokens`` is the producer's own per-request proportional-fill
    token *estimate*, not an exact count; it is re-estimated by the IR layer
    only when this field is non-positive.
    """

    model_config = ConfigDict(extra="forbid")

    hash: str = Field(
        description="'sha256:' + 16 hex chars (64-bit truncated sha256) over "
        "this block's canonical JSON, including cache_control."
    )
    bytes: int = Field(
        ge=0, description="Exact byte count of this block's canonical JSON."
    )
    tokens: int = Field(
        default=0,
        ge=0,
        description="Producer's per-request proportional-fill token estimate. "
        "Drifts request-to-request for unchanged content because it is "
        "re-derived from this request's total input_tokens, not frozen per id. "
        "The producer's own field is `omitempty`, so this key is absent on "
        "disk whenever the estimate is exactly 0 -- default must be 0, not "
        "required, or a real (low-token) block fails to parse.",
    )
    cache_control: str | None = Field(
        default=None,
        description="Cache-control breakpoint marker, if any, on this block.",
    )
    block_types: list[str] | None = Field(
        default=None,
        description="Content block type tags (e.g. 'text', 'tool_use', "
        "'tool_result'). Can be JSON null (a Go nil slice), not just absent.",
    )
    tool_use_ids: list[str] | None = Field(
        default=None, description="Tool-use ids referenced within this block, if any."
    )
    tool_result_ids: list[str] | None = Field(
        default=None,
        description="Tool-result ids referenced within this block, if any.",
    )


class ReplaySystemBlock(ReplayBlockBase):
    """One system-prompt content block."""

    type: str = Field(
        description="Content block type, always 'text' in the real corpus. "
        "Required (no omitempty on the producer's side -- always present)."
    )


class ReplayToolsBlock(ReplayBlockBase):
    """The request's tools definition, represented as a single aggregated block."""

    count: int = Field(
        ge=0,
        description="Number of tool definitions folded into this block. Wire "
        "field name is 'count' (verified against the real producer's "
        "ReplayToolsSpec struct); the IR layer's own BlockSlotIR.tool_count "
        "is populated from this field, not renamed at this layer.",
    )


class ReplayMessageBlock(ReplayBlockBase):
    """One conversation message content block."""

    role: str = Field(
        description="Message role as recorded by the producer (user/assistant/tool/...)."
    )
    seed_hash: str | None = Field(
        default=None,
        description="User, text-only messages only: hash of the message's text "
        "with <system-reminder> wrappers stripped before hashing. Unlike "
        "``hash``, this does not depend on cache_control.",
    )


class ReplayRequest(AIPerfBaseModel):
    """One request (one API call) within an instance."""

    model_config = ConfigDict(extra="forbid")

    request_id: int = Field(
        ge=0,
        description="Corpus-global counter. 0 is a sentinel value, never a real "
        "parent reference.",
    )
    ts: str = Field(
        description="RFC3339Nano request start time (raw string; not parsed at "
        "this layer — see module docstring)."
    )
    model: str = Field(description="Model name as requested, before router rewrite.")
    stream: bool = Field(description="Whether this request used streaming.")
    max_tokens: int | None = Field(
        default=None, ge=0, description="Requested max_tokens, if recorded."
    )
    input_tokens: int = Field(
        ge=0,
        description="prefill_tokens + cache_read_tokens + cache_creation_tokens, already summed.",
    )
    prefill_tokens: int = Field(
        ge=0, description="Exact, server-reported prefill token count."
    )
    cache_read_tokens: int = Field(
        ge=0, description="Exact, server-reported cache-read token count."
    )
    cache_creation_tokens: int = Field(
        ge=0, description="Exact, server-reported cache-creation token count."
    )
    output_tokens: int = Field(
        ge=0, description="Exact, server-reported output token count."
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Real sampling temperature, if recorded. Upper bound of 2.0 "
        "matches the widest range seen across common inference APIs (e.g. "
        "OpenAI's 0-2); no stricter precedent exists elsewhere in this codebase.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Real sampling top_p, if recorded. Standard nucleus-sampling "
        "range.",
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Real sampling top_k, if recorded. 0 is accepted as some "
        "APIs use it to mean 'disabled'.",
    )
    thinking: dict[str, Any] | None = Field(
        default=None,
        description="Real 'thinking' sampling config, passed through verbatim.",
    )
    system_blocks: list[ReplaySystemBlock] = Field(
        default_factory=list, description="Ordered system-prompt content blocks."
    )
    tools: ReplayToolsBlock | None = Field(
        default=None, description="Tools definition block, if any tools were offered."
    )
    messages: list[ReplayMessageBlock] = Field(
        default_factory=list, description="Ordered message content blocks."
    )
    stop_reason: str | None = Field(
        default=None, description="Recorded stop reason, if any."
    )
    upstream_latency_ms: float = Field(
        default=0.0,
        ge=0,
        description="Upstream latency in milliseconds. Always 0 in v3; real in v4+.",
    )
    total_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total request duration in milliseconds. Always 0 in v3; real in v4+.",
    )
    think_ms: float | None = Field(
        default=None,
        ge=0,
        description="v4+ only: recorded think time in milliseconds.",
    )
    think_ref: Literal["prev", "spawn", "session_start"] | None = Field(
        default=None, description="v4+ only: what think_ms is measured relative to."
    )


class ReplayInstance(AIPerfBaseModel):
    """One instance (one agent persona's run) within a session."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(description="Format 'session:sig:fp'.")
    role: str = Field(
        description="Free-form role string as recorded by the producer. Mapped "
        "to an enum in the IR step so an unrecognized value warns instead of "
        "failing the load."
    )
    parent_instance_id: str | None = Field(
        default=None,
        description="Parsed but never used for structure — not a reliable "
        "cross-session pointer (see plan).",
    )
    parent_spawn_request_id: int | None = Field(
        default=None,
        ge=0,
        description="request_id of the turn that spawned this instance, if any. "
        "0 is a legitimate request_id (see ReplayRequest.request_id) but is "
        "treated as a falsy 'no parent' sentinel by the IR layer.",
    )
    fan_out_group: str | None = Field(
        default=None,
        description="The spawning request id as a decimal string. A real causal "
        "link holds fan_out_group == str(parent_spawn_request_id).",
    )
    fan_out_size: int | None = Field(
        default=None,
        ge=0,
        description="Number of sibling instances in this fan-out group.",
    )
    fan_out_position: int | None = Field(
        default=None,
        ge=0,
        description="0-indexed position within the fan-out group. Go's "
        "omitempty means position 0 is absent on disk, not literal 0 — "
        "absence means 0, not missing data.",
    )
    requests: list[ReplayRequest] = Field(
        description="Ordered requests made by this instance."
    )


class ReplaySession(AIPerfBaseModel):
    """One line (after the header) of a replay-vN capture: one session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Unique session identifier.")
    start_ts: str = Field(
        description="RFC3339Nano session start time (raw string; not parsed at "
        "this layer — see module docstring)."
    )
    instances: list[ReplayInstance] | None = Field(
        default=None, description="Instances within this session."
    )


class ReplayIndexSession(AIPerfBaseModel):
    """One entry in the v5 trailing index: byte location of one session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        description="Session identifier, matching the session line's own."
    )
    offset: int = Field(
        ge=0, description="Byte offset of the session line's JSON content."
    )
    length: int = Field(
        ge=0, description="Byte length of the session line's JSON content."
    )
    start_ts: str = Field(description="RFC3339Nano session start time (raw string).")
    end_ts: str = Field(description="RFC3339Nano session end time (raw string).")
    requests: int = Field(ge=0, description="Number of requests in this session.")


class ReplayIndex(AIPerfBaseModel):
    """The v5 trailing index line: ``{"_schema": "replay-index-v1", "sessions": [...]}``.

    Always the last line of the file when present; sessions are listed in
    ``start_ts`` order.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["replay-index-v1"] = Field(
        alias="_schema", description="Discriminator: always 'replay-index-v1'."
    )
    sessions: list[ReplayIndexSession] = Field(
        description="Byte location of every session in the file, in start_ts order."
    )
