# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure timing helpers for ``weka_replay`` tree planning.

No AIPerf types, no I/O - plain floats in/out. Shared by
``_weka_replay_tree.py`` for both spawn-branch anchoring (sync/async children
of a specific parent turn) and co-session anchoring (background streams
attached to the primary root).
"""

from __future__ import annotations

from collections.abc import Sequence


def compute_delay_ms(
    prev_ts_ms: float, prev_api_time_ms: float | None, ts_ms: float
) -> float:
    """End-to-start idle gap between two consecutive requests on one instance.

    ``prev_api_time_ms`` is the previous request's real server-side duration
    (``None`` on replay-v3, which carries no durations at all). When it is
    ``None`` this degrades to the raw start-to-start gap - a deliberate,
    known-lossy approximation on v3 input (see plan §4.5); the caller, not
    this function, is responsible for the one-time warning about that
    degradation, so this function stays pure and warning-free.
    """
    prev_end_ms = prev_ts_ms + (prev_api_time_ms or 0.0)
    return max(0.0, ts_ms - prev_end_ms)


def compute_branch_start_ms(
    parent_request_ts_ms: float,
    parent_request_api_time_ms: float | None,
    child_first_ts_ms_values: Sequence[float],
) -> float | None:
    """Effective start time of a branch of children anchored to one parent turn.

    Returns ``min(parent_end_ms, min(child_first_ts_ms_values))``, where
    ``parent_end_ms`` is the parent turn's recorded end time when its
    duration is known, or - when it is ``None`` (v3, or a co-session anchor
    with no real "end" concept) - simply the earliest child start, so the
    branch is never reported as starting later than its own children
    actually did. Returns ``None`` for an empty child list (defensive; real
    callers never invoke this with no children).
    """
    if not child_first_ts_ms_values:
        return None
    earliest_child_ms = min(child_first_ts_ms_values)
    if parent_request_api_time_ms is not None:
        parent_end_ms = parent_request_ts_ms + parent_request_api_time_ms
    else:
        parent_end_ms = earliest_child_ms
    return min(parent_end_ms, earliest_child_ms)


def think_ms_disagrees(computed_ms: float, recorded_think_ms: float) -> bool:
    """Whether a computed gap (inter-turn delay, or a spawn branch's
    start-to-child-start offset) disagrees meaningfully with the capture's
    own recorded ``think_ms`` for that same reference.

    Validation only (see plan §4.5's "pending cross-check" and
    _weka_replay_identity.py's churn-fragment validation for the same
    pattern elsewhere in this loader) -- never gates behavior, only counted
    by the caller. Threshold is 500ms absolute or 10% relative, whichever is
    larger, to tolerate float/rounding noise without masking a real drift
    between this loader's timestamp-derived timing and the producer's own
    ``think_ms``.
    """
    threshold = max(500.0, 0.1 * recorded_think_ms)
    return abs(computed_ms - recorded_think_ms) > threshold
