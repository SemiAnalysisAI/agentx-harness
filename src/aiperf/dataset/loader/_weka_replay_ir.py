# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-function normalization of one raw :class:`ReplaySession` into an IR.

``normalize_session`` parses timestamps, sorts requests, classifies instance
roles and causal/implicit spawn links, and fills token estimates. It has no
dependency on AIPerf's ``Conversation``/``Turn`` models or any other AIPerf
framework type — only stdlib plus the two sibling modules
(``weka_replay_models`` for the raw schema, ``_weka_replay_reader`` for
:class:`WekaReplayLoadError`) — so it is importable and testable in total
isolation from the rest of AIPerf.

This module does **not** drop or filter instances for policy reasons (role
allow-lists, ephemeral/orphan handling) — that's the tree-planning layer's
job. The one exception is a structural necessity: an instance with zero
requests has no ``first_ts_ms``/``last_end_ms`` to report and cannot be
represented as a valid :class:`InstanceIR`, so it is dropped here and counted
in :class:`SessionDiagnostics`, distinct from policy-driven filtering.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

from aiperf.dataset.loader._weka_replay_reader import WekaReplayLoadError

if TYPE_CHECKING:
    from aiperf.dataset.loader.weka_replay_models import (
        ReplayMessageBlock,
        ReplayRequest,
        ReplaySession,
        ReplaySystemBlock,
        ReplayToolsBlock,
    )

    _ReplayBlock = ReplaySystemBlock | ReplayToolsBlock | ReplayMessageBlock

_EXAMPLE_CAP = 5


class ReplayRole(StrEnum):
    """Instance role, trusted as recorded (never recomputed — see plan §4.1)."""

    MAIN = "main"
    SUB_AGENT = "sub-agent"
    HELPER = "helper-or-isolated"
    ORPHAN_SUB_AGENT = "orphan-sub-agent"
    EPHEMERAL = "ephemeral (no system)"
    UNKNOWN = "unknown"


class LinkKind(StrEnum):
    """How an instance relates to the rest of its session's tree."""

    CAUSAL = "causal"
    IMPLICIT = "implicit"
    NONE = "none"
    CYCLE_BROKEN = "cycle_broken"
    ANOMALOUS = "anomalous"


class BlockKind(IntEnum):
    """Which part of a request a :class:`BlockSlotIR` represents."""

    TOOLS = 0
    SYSTEM = 1
    MESSAGE = 2


_ROLE_MAP: dict[str, ReplayRole] = {
    "main": ReplayRole.MAIN,
    "sub-agent": ReplayRole.SUB_AGENT,
    "helper-or-isolated": ReplayRole.HELPER,
    "orphan-sub-agent": ReplayRole.ORPHAN_SUB_AGENT,
    "ephemeral (no system)": ReplayRole.EPHEMERAL,
}

# RFC3339 with an optional fractional-second component of ANY digit count
# (Go's RFC3339Nano trims trailing zeros, so it isn't always 0/3/6/9 digits)
# and either 'Z' or a numeric UTC offset.
_RFC3339_RE = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<tz>Z|[+-]\d{2}:\d{2})$"
)


@dataclass(frozen=True, slots=True)
class BlockSlotIR:
    """One normalized content block (a system block, the tools block, or one message)."""

    kind: BlockKind
    slot_index: int
    role: str
    hash: str
    byte_count: int
    tokens_est: int
    cache_control: str | None
    block_types: tuple[str, ...]
    tool_use_ids: tuple[str, ...]
    tool_result_ids: tuple[str, ...]
    seed_hash: str | None
    tool_count: int


@dataclass(frozen=True, slots=True)
class SamplingIR:
    """Real sampling parameters, passed through from the raw request."""

    temperature: float | None
    top_p: float | None
    top_k: int | None
    thinking: dict | None


@dataclass(frozen=True, slots=True)
class RequestIR:
    """One normalized request within an instance."""

    request_id: int
    ts_ms: float
    model: str
    stream: bool
    max_tokens_recorded: int
    input_tokens: int
    prefill_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    output_tokens: int
    sampling: SamplingIR
    stop_reason: str | None
    api_time_ms: float | None
    upstream_latency_ms: float | None
    think_ms: float | None
    think_ref: str | None
    system: tuple[BlockSlotIR, ...]
    tools: BlockSlotIR | None
    messages: tuple[BlockSlotIR, ...]


@dataclass(frozen=True, slots=True)
class LinkIR:
    """How one instance is linked to the rest of its session's tree."""

    kind: LinkKind
    parent_request_id: int | None
    fan_out_group: str | None
    fan_out_size: int | None
    fan_out_position: int
    reason: str


@dataclass(frozen=True, slots=True)
class InstanceIR:
    """One normalized instance within a session."""

    instance_id: str
    role: ReplayRole
    raw_role: str
    link: LinkIR
    requests: tuple[RequestIR, ...]
    first_ts_ms: float
    last_end_ms: float


@dataclass(frozen=True, slots=True)
class SessionIR:
    """One normalized session: every instance, classified and linked."""

    session_id: str
    schema_version: str
    start_ts_ms: float
    orphan_shape: str
    instances: tuple[InstanceIR, ...]
    request_owner: Mapping[int, tuple[int, int]]


@dataclass(slots=True)
class SessionDiagnostics:
    """Plain, mutable counters surfaced to the caller after normalization.

    Each ``*_examples``/``anomalous_links`` list is capped at
    :data:`_EXAMPLE_CAP` entries — enough to spot-check a problem without
    unbounded growth on a pathological file.
    """

    dropped_instance_count: int = 0
    dropped_instance_examples: list[str] = field(default_factory=list)

    requests_resorted: int = 0

    unknown_role_count: int = 0
    unknown_role_examples: list[str] = field(default_factory=list)

    duplicate_request_id_count: int = 0
    duplicate_request_id_examples: list[str] = field(default_factory=list)

    cycle_broken_count: int = 0
    anomalous_count: int = 0
    anomalous_links: list[str] = field(default_factory=list)

    token_estimate_fill_count: int = 0
    input_tokens_mismatch_count: int = 0
    billing_header_dropped_count: int = 0

    # --- Populated by weka_replay.py: cross-checks this loader's own
    # timestamp-derived delay/branch-start timing against the capture's own
    # recorded think_ms, for the same reference ("prev"/"spawn"). Validation
    # only; think_ms itself is never used to drive behavior (see
    # _weka_replay_timing.think_ms_disagrees). ---
    think_prev_compared_count: int = 0
    think_prev_disagreement_count: int = 0
    think_spawn_compared_count: int = 0
    think_spawn_disagreement_count: int = 0

    # --- Populated by the tree-planning layer (_weka_replay_tree.py), not by
    # normalize_session itself - kept on this one shared dataclass per the
    # project convention of a single diagnostics object per session rather
    # than a second, parallel one. See plan §4.2-§4.4. ---
    orphan_session_dropped: int = 0
    empty_session_after_filtering: int = 0
    dropped_for_max_context_length: int = 0

    ephemeral_dropped_count: int = 0
    role_filtered_dropped_count: int = 0

    cascade_dropped_count: int = 0
    cascade_dropped_examples: list[str] = field(default_factory=list)

    dropped_for_max_isl_count: int = 0
    dropped_for_max_isl_examples: list[str] = field(default_factory=list)

    causal_parent_dropped_by_max_isl_count: int = 0
    causal_parent_dropped_by_max_isl_examples: list[str] = field(default_factory=list)

    root_fallback_used_count: int = 0


def _append_capped(examples: list[str], item: str, cap: int = _EXAMPLE_CAP) -> None:
    if len(examples) < cap:
        examples.append(item)


def _parse_ts_ms(ts: str, *, context: str, lineno: int) -> float:
    """Parse an RFC3339Nano timestamp string to epoch milliseconds.

    ``datetime.fromisoformat`` only accepts a 0/3/6-digit fractional-second
    component, but Go's RFC3339Nano trims trailing zeros from the fraction
    (so e.g. a 1-, 2-, 5-, 7-, 8-, or 9-digit fraction is common on real
    data). On a ``fromisoformat`` failure, fall back to normalizing the
    fraction to exactly 6 digits (right-pad short fractions with zeros —
    correct, since the trimmed digits were zeros; truncate long ones) and
    'Z' to '+00:00', then retry.
    """
    try:
        return datetime.fromisoformat(ts).timestamp() * 1000.0
    except ValueError:
        pass
    match = _RFC3339_RE.match(ts)
    if match is None:
        raise WekaReplayLoadError(
            f"weka_replay line {lineno}: {context}: invalid timestamp {ts!r}: "
            "not a recognizable RFC3339 timestamp"
        )
    base = match.group("base")
    tz = match.group("tz")
    if tz == "Z":
        tz = "+00:00"
    frac = (match.group("frac") or "").ljust(6, "0")[:6]
    normalized = f"{base}.{frac}{tz}"
    try:
        return datetime.fromisoformat(normalized).timestamp() * 1000.0
    except ValueError as exc:
        raise WekaReplayLoadError(
            f"weka_replay line {lineno}: {context}: invalid timestamp {ts!r}: {exc}"
        ) from exc


def classify_orphan_shape(session_id: str) -> str:
    if session_id == "orphan:no-session-id":
        return "shared_bucket"
    if session_id.startswith("orphan:no-session-id:"):
        return "per_record"
    return "none"


def _map_role(raw_role: str, diag: SessionDiagnostics) -> ReplayRole:
    role = _ROLE_MAP.get(raw_role)
    if role is not None:
        return role
    diag.unknown_role_count += 1
    _append_capped(diag.unknown_role_examples, raw_role)
    return ReplayRole.UNKNOWN


def _block_tuple(values: list[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else ()


@dataclass(slots=True)
class _RawSlot:
    kind: BlockKind
    slot_index: int
    role: str
    block: _ReplayBlock
    tool_count: int


_BILLING_HEADER_MAX_BYTES = 200


def _effective_system_blocks(
    blocks: list[ReplaySystemBlock],
) -> list[ReplaySystemBlock]:
    """Drop the capture's per-request billing/preamble header block
    (``system_blocks[0]`` when its ``bytes < 200``).

    Mirrors wekai's own ``effectiveSystemBlocks``
    (``benchmark/replay_router_wire.go``): that header carries a near-unique
    hash on essentially every request -- unlike Anthropic's real
    ``cache_control`` breakpoints, which sit on the shared system blocks that
    follow it, this header is never itself cache-marked. Left in, it poisons
    this loader's positional content-identity matching for every block after
    it: confirmed against the real corpus, 91% of all request-to-request
    wire-divergence events trace to exactly this slot, even though >99% of
    individual blocks resolve to reused content. Slot indices for the
    remaining system blocks are renumbered from 0 (not left at their raw
    array position) so cross-turn churn matching keyed by ``slot_index``
    stays stable regardless of whether this header was present that turn.
    """
    if blocks and blocks[0].bytes < _BILLING_HEADER_MAX_BYTES:
        return blocks[1:]
    return blocks


def _collect_raw_slots(req: ReplayRequest, diag: SessionDiagnostics) -> list[_RawSlot]:
    slots: list[_RawSlot] = []
    system_blocks = _effective_system_blocks(req.system_blocks)
    if len(system_blocks) < len(req.system_blocks):
        diag.billing_header_dropped_count += 1
    for i, sb in enumerate(system_blocks):
        slots.append(_RawSlot(BlockKind.SYSTEM, i, "system", sb, 0))
    if req.tools is not None:
        slots.append(_RawSlot(BlockKind.TOOLS, 0, "tools", req.tools, req.tools.count))
    for j, mb in enumerate(req.messages):
        slots.append(_RawSlot(BlockKind.MESSAGE, j, mb.role, mb, 0))
    return slots


def _build_block_slots(
    req: ReplayRequest, diag: SessionDiagnostics
) -> tuple[BlockSlotIR, ...]:
    """Build every content-block slot for one request, filling token estimates.

    Recorded ``tokens > 0`` is kept as-is (frozen). For the remaining blocks
    (``tokens <= 0``), the request's exact ``input_tokens`` minus whatever was
    already recorded is spread across just those blocks in proportion to
    their byte size, with the remainder on the last such block — so the
    block-level total always matches ``input_tokens`` exactly whenever a fill
    happens. When no fill is needed, a mismatch between the already-recorded
    total and ``input_tokens`` is counted (not corrected) as producer drift.
    """
    raw_slots = _collect_raw_slots(req, diag)
    recorded_sum = sum(s.block.tokens for s in raw_slots if s.block.tokens > 0)
    needs_fill = [s for s in raw_slots if s.block.tokens <= 0]
    remaining = max(0, req.input_tokens - recorded_sum)

    fill_values: dict[tuple[BlockKind, int], int] = {}
    if needs_fill:
        total_bytes = sum(s.block.bytes for s in needs_fill)
        allocated = 0
        if total_bytes > 0:
            for s in needs_fill[:-1]:
                share = int(remaining * s.block.bytes / total_bytes)
                fill_values[(s.kind, s.slot_index)] = share
                allocated += share
        else:
            base = remaining // len(needs_fill)
            for s in needs_fill[:-1]:
                fill_values[(s.kind, s.slot_index)] = base
                allocated += base
        last = needs_fill[-1]
        fill_values[(last.kind, last.slot_index)] = remaining - allocated
        diag.token_estimate_fill_count += len(needs_fill)

    result: list[BlockSlotIR] = []
    for s in raw_slots:
        tokens_est = (
            s.block.tokens
            if s.block.tokens > 0
            else fill_values[(s.kind, s.slot_index)]
        )
        result.append(
            BlockSlotIR(
                kind=s.kind,
                slot_index=s.slot_index,
                role=s.role,
                hash=s.block.hash,
                byte_count=s.block.bytes,
                tokens_est=tokens_est,
                cache_control=s.block.cache_control,
                block_types=_block_tuple(s.block.block_types),
                tool_use_ids=_block_tuple(s.block.tool_use_ids),
                tool_result_ids=_block_tuple(s.block.tool_result_ids),
                seed_hash=getattr(s.block, "seed_hash", None),
                tool_count=s.tool_count,
            )
        )

    if not needs_fill and sum(b.tokens_est for b in result) != req.input_tokens:
        diag.input_tokens_mismatch_count += 1

    return tuple(result)


def _build_request_ir(
    req: ReplayRequest, ts_ms: float, diag: SessionDiagnostics
) -> RequestIR:
    blocks = _build_block_slots(req, diag)
    system = tuple(b for b in blocks if b.kind == BlockKind.SYSTEM)
    tools = next((b for b in blocks if b.kind == BlockKind.TOOLS), None)
    messages = tuple(b for b in blocks if b.kind == BlockKind.MESSAGE)

    return RequestIR(
        request_id=req.request_id,
        ts_ms=ts_ms,
        model=req.model,
        stream=req.stream,
        max_tokens_recorded=req.max_tokens or 0,
        input_tokens=req.input_tokens,
        prefill_tokens=req.prefill_tokens,
        cache_read_tokens=req.cache_read_tokens,
        cache_creation_tokens=req.cache_creation_tokens,
        output_tokens=req.output_tokens,
        sampling=SamplingIR(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            thinking=req.thinking,
        ),
        stop_reason=req.stop_reason,
        api_time_ms=req.total_ms if req.total_ms > 0 else None,
        upstream_latency_ms=req.upstream_latency_ms
        if req.upstream_latency_ms > 0
        else None,
        think_ms=req.think_ms,
        think_ref=req.think_ref,
        system=system,
        tools=tools,
        messages=messages,
    )


@dataclass(slots=True)
class _PendingInstance:
    instance_id: str
    raw_role: str
    fan_out_group: str | None
    parent_spawn_request_id: int | None
    fan_out_size: int | None
    fan_out_position: int | None
    requests: tuple[RequestIR, ...]
    first_ts_ms: float
    last_end_ms: float


def _sort_requests(
    requests: list[ReplayRequest], *, session_id: str, instance_id: str, lineno: int
) -> tuple[list[tuple[ReplayRequest, float]], bool]:
    parsed = [
        (
            req,
            _parse_ts_ms(
                req.ts,
                context=f"session '{session_id}' instance '{instance_id}' request {req.request_id}",
                lineno=lineno,
            ),
        )
        for req in requests
    ]
    original_ids = [req.request_id for req, _ in parsed]
    parsed_sorted = sorted(parsed, key=lambda pair: (pair[1], pair[0].request_id))
    resorted = [req.request_id for req, _ in parsed_sorted] != original_ids
    return parsed_sorted, resorted


def _classify_link(
    *,
    fan_out_group: str | None,
    parent_id: int | None,
    fan_out_size: int | None,
    fan_out_position_raw: int | None,
    self_idx: int,
    first_ts_ms: float,
    request_owner: Mapping[int, tuple[int, int]],
    request_ts_by_id: Mapping[int, float],
    diag: SessionDiagnostics,
    instance_label: str,
) -> LinkIR:
    fan_out_position = fan_out_position_raw or 0

    if fan_out_group is not None:
        if not parent_id:
            diag.cycle_broken_count += 1
            _append_capped(diag.anomalous_links, f"{instance_label}:cycle_broken")
            return LinkIR(
                LinkKind.CYCLE_BROKEN,
                None,
                fan_out_group,
                fan_out_size,
                fan_out_position,
                "cycle_broken",
            )
        if fan_out_group != str(parent_id):
            diag.anomalous_count += 1
            _append_capped(diag.anomalous_links, f"{instance_label}:group_mismatch")
            return LinkIR(
                LinkKind.ANOMALOUS,
                parent_id,
                fan_out_group,
                fan_out_size,
                fan_out_position,
                "group_mismatch",
            )
        owner = request_owner.get(parent_id)
        if owner is None:
            diag.anomalous_count += 1
            _append_capped(diag.anomalous_links, f"{instance_label}:unowned")
            return LinkIR(
                LinkKind.ANOMALOUS,
                parent_id,
                fan_out_group,
                fan_out_size,
                fan_out_position,
                "unowned",
            )
        owner_idx, _owner_turn_idx = owner
        if owner_idx == self_idx:
            diag.anomalous_count += 1
            _append_capped(diag.anomalous_links, f"{instance_label}:self_spawn")
            return LinkIR(
                LinkKind.ANOMALOUS,
                parent_id,
                fan_out_group,
                fan_out_size,
                fan_out_position,
                "self_spawn",
            )
        owner_ts = request_ts_by_id[parent_id]
        if first_ts_ms < owner_ts:
            diag.anomalous_count += 1
            _append_capped(diag.anomalous_links, f"{instance_label}:time_inversion")
            return LinkIR(
                LinkKind.ANOMALOUS,
                parent_id,
                fan_out_group,
                fan_out_size,
                fan_out_position,
                "time_inversion",
            )
        return LinkIR(
            LinkKind.CAUSAL,
            parent_id,
            fan_out_group,
            fan_out_size,
            fan_out_position,
            "ok",
        )

    if parent_id:
        return LinkIR(LinkKind.IMPLICIT, parent_id, None, None, fan_out_position, "ok")

    return LinkIR(LinkKind.NONE, None, None, None, fan_out_position, "ok")


def normalize_session(
    raw: ReplaySession, *, schema_version: str, lineno: int, diag: SessionDiagnostics
) -> SessionIR:
    """Normalize one raw session into a :class:`SessionIR`.

    Does not drop or filter instances for policy reasons — see the module
    docstring. Raises :class:`WekaReplayLoadError` only for genuine data
    integrity problems (an unparsable timestamp, or a duplicated
    ``request_id`` that is also referenced by some instance's
    ``parent_spawn_request_id``, which would make linkage ambiguous).
    """
    start_ts_ms = _parse_ts_ms(
        raw.start_ts, context=f"session '{raw.session_id}'", lineno=lineno
    )
    orphan_shape = classify_orphan_shape(raw.session_id)

    pending: list[_PendingInstance] = []
    request_owner: dict[int, tuple[int, int]] = {}
    request_ts_by_id: dict[int, float] = {}
    duplicated_ids: set[int] = set()
    referenced_parent_ids: set[int] = set()

    for inst in raw.instances or []:
        if not inst.requests:
            diag.dropped_instance_count += 1
            _append_capped(
                diag.dropped_instance_examples, f"{inst.instance_id}:empty_requests"
            )
            continue

        if inst.parent_spawn_request_id:
            referenced_parent_ids.add(inst.parent_spawn_request_id)

        parsed_sorted, resorted = _sort_requests(
            inst.requests,
            session_id=raw.session_id,
            instance_id=inst.instance_id,
            lineno=lineno,
        )
        if resorted:
            diag.requests_resorted += 1

        instance_idx = len(pending)
        request_irs: list[RequestIR] = []
        for turn_idx, (req, ts_ms) in enumerate(parsed_sorted):
            if req.request_id in request_owner:
                diag.duplicate_request_id_count += 1
                duplicated_ids.add(req.request_id)
                _append_capped(
                    diag.duplicate_request_id_examples,
                    f"{req.request_id}@{inst.instance_id}",
                )
            else:
                request_owner[req.request_id] = (instance_idx, turn_idx)
                request_ts_by_id[req.request_id] = ts_ms
            request_irs.append(_build_request_ir(req, ts_ms, diag))

        first_ts_ms = request_irs[0].ts_ms
        last_req = request_irs[-1]
        last_end_ms = last_req.ts_ms + (last_req.api_time_ms or 0.0)

        pending.append(
            _PendingInstance(
                instance_id=inst.instance_id,
                raw_role=inst.role,
                fan_out_group=inst.fan_out_group,
                parent_spawn_request_id=inst.parent_spawn_request_id,
                fan_out_size=inst.fan_out_size,
                fan_out_position=inst.fan_out_position,
                requests=tuple(request_irs),
                first_ts_ms=first_ts_ms,
                last_end_ms=last_end_ms,
            )
        )

    ambiguous = duplicated_ids & referenced_parent_ids
    if ambiguous:
        bad_id = sorted(ambiguous)[0]
        raise WekaReplayLoadError(
            f"weka_replay line {lineno}: session '{raw.session_id}': request_id "
            f"{bad_id} is duplicated across instances and also referenced by a "
            "parent_spawn_request_id; linkage would be ambiguous"
        )

    instances: list[InstanceIR] = []
    for idx, p in enumerate(pending):
        link = _classify_link(
            fan_out_group=p.fan_out_group,
            parent_id=p.parent_spawn_request_id,
            fan_out_size=p.fan_out_size,
            fan_out_position_raw=p.fan_out_position,
            self_idx=idx,
            first_ts_ms=p.first_ts_ms,
            request_owner=request_owner,
            request_ts_by_id=request_ts_by_id,
            diag=diag,
            instance_label=p.instance_id,
        )
        instances.append(
            InstanceIR(
                instance_id=p.instance_id,
                role=_map_role(p.raw_role, diag),
                raw_role=p.raw_role,
                link=link,
                requests=p.requests,
                first_ts_ms=p.first_ts_ms,
                last_end_ms=p.last_end_ms,
            )
        )

    return SessionIR(
        session_id=raw.session_id,
        schema_version=schema_version,
        start_ts_ms=start_ts_ms,
        orphan_shape=orphan_shape,
        instances=tuple(instances),
        request_owner=request_owner,
    )
