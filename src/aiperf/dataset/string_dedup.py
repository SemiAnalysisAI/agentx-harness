# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collapse repeated large strings in a freshly decoded conversation.

Agentic traces repeat whole message bodies verbatim: a Claude Code trajectory
re-sends the same ~230 KB first user message after every context reset, so one
conversation carries 26 equal copies at ``turns[i].raw_messages[0].content``.
JSON decoding builds a separate ``str`` for every occurrence. Measured over 300
conversations of the AgentX Weka corpus, strings of 1 KiB or more held 602 MiB
decoded but only 154 MiB of distinct content.

Most of that content is also non-Latin-1 (a few ``✓``/``—`` characters force
CPython's 2-byte UCS-2 storage), and ``orjson.dumps`` caches a UTF-8 copy on
every such string it serializes. Deduplicating means each distinct body pays
both costs once per conversation instead of once per occurrence.

The walk only rebinds equal values to one shared object, so every consumer
sees identical content and identical wire bytes. Shared strings are safe
because ``str`` is immutable; writers that change a message (cache-bust) build
a new string rather than editing one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.common.models.dataset_models import Conversation

#: Strings shorter than this are left alone: hashing and table entries would
#: cost more than the duplicates they save, and small keys/roles are already
#: shared or tiny.
DEDUP_MIN_CHARS = 256

#: Turn fields holding JSON-shaped payload data decoded as plain dicts/lists.
_RAW_TURN_FIELDS = ("raw_messages", "raw_tools", "raw_system", "raw_payload")


def dedupe_conversation_strings(
    conversation: Conversation, *, min_chars: int = DEDUP_MIN_CHARS
) -> None:
    """Rebind every repeated string of ``min_chars`` or more to one object.

    Mutates the containers of a conversation nothing else references yet; call
    it immediately after decoding.
    """
    seen: dict[str, str] = {}
    for turn in conversation.turns:
        for field in _RAW_TURN_FIELDS:
            value = getattr(turn, field)
            if value is not None:
                _dedupe_container(value, seen, min_chars)
        for media_list in (turn.texts, turn.images, turn.audios, turn.videos):
            for media in media_list:
                _dedupe_list(media.contents, seen, min_chars)


def _dedupe_container(value: Any, seen: dict[str, str], min_chars: int) -> None:
    if type(value) is dict:
        _dedupe_dict(value, seen, min_chars)
    elif type(value) is list:
        _dedupe_list(value, seen, min_chars)


def _dedupe_dict(value: dict, seen: dict[str, str], min_chars: int) -> None:
    for key, item in value.items():
        if type(item) is str:
            if len(item) >= min_chars:
                value[key] = seen.setdefault(item, item)
        else:
            _dedupe_container(item, seen, min_chars)


def _dedupe_list(value: list, seen: dict[str, str], min_chars: int) -> None:
    for index, item in enumerate(value):
        if type(item) is str:
            if len(item) >= min_chars:
                value[index] = seen.setdefault(item, item)
        else:
            _dedupe_container(item, seen, min_chars)
