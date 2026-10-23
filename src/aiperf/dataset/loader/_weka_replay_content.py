# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary contract between the weka_replay tree-mapping layer and the
content-synthesis layer.

Kept as its own module (rather than living in either side) so both
`_weka_replay_tree.py`/`weka_replay.py` and `_weka_replay_synth.py` can depend
on it without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from aiperf.dataset.loader._weka_replay_ir import InstanceIR, SessionIR


@dataclass(frozen=True, slots=True)
class TurnContent:
    """Synthesized content for a single request/turn.

    ``raw_messages`` is a delta (only the messages new since the previous
    turn on this instance) except on turn 0 or when ``reset_context`` is set,
    in which case it is the full current wire. ``theoretical_prefix_cache_*``
    fields are intentionally absent here: they depend on a session-wide,
    cross-instance cache simulation that only resolves after every instance
    in the session has been built, so they are applied separately by the
    caller from :meth:`SessionContentBuilder.finish`.
    """

    raw_messages: list[dict[str, Any]]
    raw_tools: list[dict[str, Any]] | None
    reset_context: bool


class SessionContentBuilder(Protocol):
    """Synthesizes content for every retained instance of one session.

    Call order across instances does not matter for the correctness of
    ``build_instance_turns`` (exact-hash/seed_hash identity is session-scoped
    and slot-churn identity is per-instance, so instances are independent of
    each other); only ``finish`` needs every instance to have been built
    first, since the theoretical cache-hit metric is computed across the
    whole session's chronological request stream.
    """

    def build_instance_turns(self, instance: InstanceIR) -> list[TurnContent]:
        """Return one :class:`TurnContent` per request in ``instance.requests``, in order."""
        ...

    def finish(self) -> dict[tuple[str, int], tuple[int, int]]:
        """Return ``(instance_id, request_id) -> (hit_blocks, total_blocks)`` for every
        turn built via :meth:`build_instance_turns` across this session.
        """
        ...


class ReplayContentBuilder(Protocol):
    """Factory for a per-session :class:`SessionContentBuilder`.

    One instance of this is constructed per loader run (it owns the shared
    tokenizer/corpus access); ``for_session`` is called once per session,
    establishing that session's local identity scope (see the module docstring
    of ``_weka_replay_synth.py`` for why scope is per-session, not global).
    """

    def for_session(self, session: SessionIR) -> SessionContentBuilder: ...
