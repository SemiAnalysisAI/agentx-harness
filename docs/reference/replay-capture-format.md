---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Replay Capture Format (replay-vN)
---

# The `replay-v3`/`v4`/`v5` Capture Format

This is a field-by-field specification of the JSONL format `wekai`'s router produces via `wekai router replay-prepare` (`cli/command_router_replay_prepare.go` in the `wekai` repository, a separate codebase from this one) and that AIPerf's [`weka_replay`](../tutorials/weka-replay.md) loader consumes directly.

**No standalone prose spec of this format exists in the `wekai` repository itself** (checked as of this writing) — the closest things are its root `README.md`'s "Replay benchmark" section (CLI usage and behavior, not a field reference) and `.ainav/router-testing/replay-notes.md` (a short file-layout note). The authoritative source is the Go struct definitions and their doc comments in `wekai`, primarily `cli/command_router_replay_prepare.go`, `benchmark/replay_index.go`, and `cli/replay_think.go`. This document distills those into a reference, cross-checked against AIPerf's own `weka_replay_models.py` (which independently verified every field against that same Go source while building the loader) and against empirical observation of a real 9.5GB production capture.

## File layout

A `replay-vN` file is newline-delimited JSON:

```
line 1:        ReplayHeader
line 2..N:     one ReplaySession object per line
line N+1:      (replay-v5 only) the trailing ReplayIndex, schema "replay-index-v1"
```

Sessions are written one per line specifically so a producer can stream output, and a consumer can stream input with a bounded queue, without holding the whole file — or its decoded objects — in memory at once. Lines after the header are otherwise unordered with respect to file position; only the optional v5 trailing index guarantees an easy way to seek to a specific session (see below).

## Version differences

| | v3 | v4 | v5 |
|---|---|---|---|
| Per-request structured content spec (`system_blocks`/`tools`/`messages`, hash-identified) | ✓ (introduced in v3, vs. v2) | ✓ | ✓ |
| `upstream_latency_ms` / `total_ms` | always `0` | real, backfilled from the capture | real |
| `think_ms` / `think_ref` per request | absent | ✓ | ✓ |
| Header `summary.timing` (corpus-wide think/duration percentiles) | absent | ✓ | ✓ |
| Trailing session index (`replay-index-v1`) | absent | absent | ✓ |

A reader that only understands v3 semantics can still parse v4/v5 input (the new fields are simply present and unused); `weka_replay`'s own schema is a single pydantic model set covering all three versions, with v4/v5-only fields typed `Optional` and absent on v3 input.

## Header (line 1)

```jsonc
{
  "_schema": "replay-v5",          // "replay-v3" | "replay-v4" | "replay-v5"
  "name": "...",                   // human-readable capture name
  "generated_at": "2026-...",      // RFC3339Nano
  "source": "...",                 // producer/process identifier
  "summary": { /* ReplaySummary */ }
}
```

### `summary` (`ReplaySummary`)

Corpus-level counts, informational only (not needed for tree reconstruction):

| Field | Type | Meaning |
|---|---|---|
| `sessions` | int | Total sessions in this capture. |
| `instances` | int | Total instances across all sessions. |
| `requests` | int | Total requests across all instances. |
| `spawns` | int | Total spawn events. |
| `matched_spawns` | int | Spawn events the producer successfully linked to a spawning turn. |
| `fan_out_turns` | int | Turns that spawned at least one sub-agent. |
| `max_fan_out_in_one_turn` | int | Largest fan-out width from a single turn. |
| `repaired_parents` | int | Instances whose parent linkage the producer had to repair. |
| `cycles_broken` | int | Causal-link cycles the producer broke while building the capture. |
| `start_ts` / `end_ts` | string | RFC3339Nano bounds of the whole capture. |
| `timing` | object, v4+ only | `ReplayTimingSummary` — see below. |

### `summary.timing` (`ReplayTimingSummary`, v4+ only)

| Field | Meaning |
|---|---|
| `requests_with_timing` | Requests carrying a real `total_ms`. |
| `requests_with_think` | Requests carrying a `think_ms`. |
| `think_clamped` / `think_clamped_percent` | How many think-time measurements were negative and clamped to 0 (overlapping requests, or a sub-agent that started before its parent's response finished). |
| `think_ms` | `ThinkPercentiles` over every request with a think time. |
| `think_ms_by_ref` | Same, broken down by `think_ref` (`"prev"`/`"spawn"`/`"session_start"`). |
| `total_ms` | `ThinkPercentiles` over real request durations. |

`ThinkPercentiles` is `{n, p50, p90, p99, max, mean}`, all integer milliseconds — note there is deliberately no `p95`.

## Session (one per line after the header)

```jsonc
{
  "session_id": "...",
  "start_ts": "2026-...",
  "instances": [ /* ReplayInstance */ ]
}
```

`session_id` normally comes from the real `X-Claude-Code-Session-Id` request header. A request with no such header is bucketed under the synthetic id `orphan:no-session-id` (a `wekai`-side constant, `cli/command_router_tree.go`'s `OrphanSessionID`) — every header-less record in the whole capture shares this ONE session, by design: it keeps the real per-session averages from being diluted by many single-record placeholder "sessions," and at replay time the whole bucket consumes one concurrency slot instead of N. There is no real chronological relationship between unrelated users' requests inside this bucket. (`weka_replay` additionally recognizes a per-record-suffixed shape, `orphan:no-session-id:<record-id>`, defensively — this suffixed form was not found in the `wekai` checkout this document was written against, so treat it as a possible alternate producer behavior rather than a confirmed current one.)

## Instance (one per agent persona's run within a session)

```jsonc
{
  "instance_id": "session:sig:fp",
  "role": "main",
  "parent_instance_id": "...",       // present but NOT reliable for structure, see below
  "parent_spawn_request_id": 123,    // 0 / absent = no parent
  "fan_out_group": "123",            // decimal string of the spawning request_id
  "fan_out_size": 2,
  "fan_out_position": 0,             // 0-indexed; omitempty, so position 0 is ABSENT on disk
  "requests": [ /* ReplayRequest */ ]
}
```

### `role`

Assigned by the *producer*, per-instance (not per-persona — the same persona/signature can serve different roles in different sessions), by `cli/command_router_tree.go`'s pass-4 classification:

| Role | Producer's condition |
|---|---|
| `sub-agent` | The instance has a resolved parent (linked by a causal spawn edge). |
| `ephemeral (no system)` | No persona signature at all (no system prompt recorded) — single-token liveness probes, CLI title-gen calls, etc. |
| `orphan-sub-agent` | This instance's own persona shape matches a child the producer successfully linked *elsewhere*, but *this* occurrence couldn't be linked (hash mismatch, context compaction, a missing parent capture). Rendered as a root so it still appears in the session tree, but labeled distinctly so it isn't confused with a real main agent. |
| `main` | No parent, and this persona is observed (anywhere in the corpus, not just this session) to spawn other agents. |
| `helper-or-isolated` | No parent, and this persona is never observed spawning anything. |

A consumer should treat `role` as ground truth — recomputing it would require corpus-global persona behavior (whether a persona *ever* spawns or is spawned *anywhere* in the whole corpus) that a single-session reader can't reconstruct.

### Spawn linkage

- `parent_instance_id` is parsed but is **not** a reliable cross-session structural pointer — do not use it to build the tree.
- The real causal link is `parent_spawn_request_id` (the `request_id` of the turn that spawned this instance) cross-checked by `fan_out_group == str(parent_spawn_request_id)`. A link should only be trusted as causal when, in addition, the referenced request genuinely belongs to a *different* instance in the same session, and this instance's own first request does not start before that parent request. `fan_out_size`/`fan_out_position` describe sibling fan-out width/order and are never load-bearing for structure by themselves.
- Everything short of a fully-verified causal link (a group marker with no resolvable parent id, a dangling/self-referential/time-inverted reference, or simply no spawn markers at all) should be treated as a non-causal stream, not asserted as a spawn — the producer's own linkage can be incomplete or, rarely, cyclic (`summary.cycles_broken`).

## Request (one API call within an instance)

```jsonc
{
  "request_id": 123,
  "ts": "2026-...",                  // RFC3339Nano, REQUEST START time
  "model": "...",                    // pre-router-rewrite name
  "stream": true,
  "max_tokens": 4096,                // original requested max_tokens
  "input_tokens": 1500,              // = prefill + cache_read + cache_creation
  "prefill_tokens": 200,
  "cache_read_tokens": 1250,
  "cache_creation_tokens": 50,
  "output_tokens": 300,              // real completion length
  "temperature": 1.0,
  "top_p": null,
  "top_k": null,
  "thinking": null,
  "system_blocks": [ /* ReplaySystemBlock */ ],
  "tools": { /* ReplayToolsSpec */ },
  "messages": [ /* ReplayMessage */ ],
  "stop_reason": "...",
  "upstream_latency_ms": 0,
  "total_ms": 0,                     // real duration, v4+ only (always 0 on v3)
  "think_ms": 450,                   // v4+ only
  "think_ref": "prev"                // "prev" | "spawn" | "session_start"
}
```

`request_id` is a corpus-global counter; `0` is a sentinel and never a real request to reference. `messages` carries the **full conversation as the original request actually sent it** — every prior turn is re-sent per request (the real wire shape), not just a delta; a consumer building a delta-based replay must diff consecutive requests on the same instance itself.

### `think_ms` / `think_ref` (v4+ only)

The producer's own precomputed measurement of "how long did this agent pause before issuing this request," per `cli/replay_think.go`'s `annotateThink`:

| `think_ref` | Reference point | `think_ms` = |
|---|---|---|
| `"prev"` | The previous request on the *same instance* | `this.ts − (prev.ts + prev.total_ms)` |
| `"spawn"` | This instance's spawning parent request | `this.ts − (parent.ts + parent.total_ms)` — every sibling of a fan-out measures from the same instant |
| `"session_start"` | The session's own `start_ts` | Only used for an instance with no resolvable parent (a true root) |

Negative values (overlapping requests, or a sub-agent that started before its parent's response finished) are clamped to `0` and counted in `summary.timing.think_clamped`. Note `"prev"`'s formula is literally the same end-to-start-idle-gap computation a consumer would derive itself from `ts`/`total_ms` — `think_ms` is a convenience/cross-check, not new information, for that one reference; `"spawn"` and `"session_start"` are the references a consumer can't derive without doing the same cross-request bookkeeping the producer already did.

## Content blocks

Real message/system/tool text is **never recorded** — every block is a hash, a byte count, and an approximate token count, so the replayer (or any other consumer) must synthesize substitute content keyed by hash. The same hash always implies the same real content, so same-hash blocks should decode to the same synthetic text for realistic prefix-cache behavior.

### `ReplaySystemBlock` (one entry per `system_blocks[i]`)

| Field | Type | Notes |
|---|---|---|
| `type` | string | Always `"text"` in practice; required (no `omitempty` on the producer's side). |
| `hash` | string | `"sha256:"` + 16 hex chars (64-bit truncated sha256) over the block's canonical JSON, **including** `cache_control` — this is why the same real system block gets a different hash whenever its `cache_control` presence changes. |
| `bytes` | int | Exact byte count of the canonical JSON. |
| `tokens` | int | Producer's *per-request* proportional-fill token estimate — re-derived from this request's own `input_tokens` each time, so it drifts request-to-request for unchanged content. `omitempty`: absent (not zero) when the estimate is exactly 0. |
| `cache_control` | string, optional | Cache-control breakpoint marker, when present on this block. |

**A real-corpus convention worth knowing**: `system_blocks[0]`, when under ~200 bytes, is a per-request billing/preamble header — not real conversation content — that the producer's own replay tooling (`cli/benchmark/replay_router_wire.go`'s `effectiveSystemBlocks`) already knows to drop before replaying or cache-estimating, because it carries a near-unique hash on essentially every request and never carries `cache_control`. `weka_replay` applies the identical rule (see [`_weka_replay_ir.py`](../tutorials/weka-replay.md)'s `_effective_system_blocks`) — this was empirically confirmed to be responsible for the overwhelming majority of request-to-request content-identity "misses" when first measured against a real corpus, since (unfixed) it mints a fresh content id at the very start of the wire on nearly every turn.

### `ReplayToolsSpec` (the whole `tools` array, as one aggregated block)

| Field | Type | Notes |
|---|---|---|
| `count` | int | Number of tool definitions folded into this one block (wire field name is literally `count`, not `tool_count`). |
| `bytes` / `tokens` | int | Same semantics as a system block. |
| `hash` | string | Fingerprint of the whole tools array — deterministic, so the same tools array hashes identically across replays of the same session. |

### `ReplayMessage` (one entry per `messages[i]`)

| Field | Type | Notes |
|---|---|---|
| `role` | string | As recorded by the producer (`user`/`assistant`/`tool`/...). |
| `hash` | string | Same `"sha256:"+16hex` shape and `cache_control`-inclusive hashing as a system block. |
| `block_types` | `[]string`, nullable | Content block type tags (`text`, `tool_use`, `tool_result`, ...). Can be **JSON `null`** (a Go nil slice), not just absent — a reader must handle both. |
| `bytes` / `tokens` | int | As above. |
| `cache_control` | string, optional | As above. |
| `tool_use_ids` / `tool_result_ids` | `[]string`, optional | Preserves the tool-call reference graph across turns. |
| `seed_hash` | string, optional | **User, text-only messages only.** A hash of the message's real text with `<system-reminder>` wrappers stripped *before* hashing — unlike `hash`, this does **not** depend on `cache_control`, so it's the one signal that survives a `cache_control` breakpoint moving without the underlying user text changing. |

## The v5 trailing index (`replay-index-v1`)

The last line of a v5 file, written only after every session line (so the producer can stream session output and only then emit the table it enables):

```jsonc
{
  "_schema": "replay-index-v1",
  "sessions": [
    {
      "session_id": "...",
      "offset": 123456,      // byte offset of the session line's JSON content
      "length": 2048,        // byte length (excludes the trailing newline)
      "start_ts": "...",     // this session's first-seen time
      "end_ts": "...",       // when its last request finished, in the capture
      "requests": 42
    }
  ]
}
```

Sessions are listed in `start_ts` order. This is what lets a reader seek directly to any session (or sample newest-first) without a full file scan.

**Locating this line efficiently**: `wekai`'s own reader (`benchmark/replay_index.go`'s `readReplayIndex`) reads a tail window of the file looking for the final line's start, **doubling the window if no newline is found**, rather than committing to one fixed-size guess (its own comment: "an index costs ~190 bytes per session... a larger index costs one more read per doubling"). `weka_replay`'s reader (`_weka_replay_reader.py`'s `_find_trailer_newline`) independently arrived at the identical strategy — worth knowing if comparing the two implementations, since neither has a hard size ceiling on the index it can locate.

## Cross-references

- [`weka_replay` loader tutorial](../tutorials/weka-replay.md) — how AIPerf consumes this format, what it drops by default, flag behavior, and known fidelity/timing caveats.
- `weka_replay_models.py` (`src/aiperf/dataset/loader/`) — the pydantic schema this document was cross-checked against; treat it as the second source of truth alongside the `wekai` Go structs themselves.
