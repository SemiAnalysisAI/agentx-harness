# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repeated large strings in a decoded conversation collapse to one object."""

from __future__ import annotations

import orjson

from aiperf.common.models.dataset_models import Conversation
from aiperf.dataset.memory_map_utils import MemoryMapDatasetClient
from aiperf.dataset.string_dedup import DEDUP_MIN_CHARS, dedupe_conversation_strings

# Non-Latin-1 on purpose: this is the UCS-2 shape the AgentX corpus decodes to.
_BIG = "context ✓ " * 200
_SHORT = "hi"


def _conversation_bytes() -> bytes:
    def turn(text_part: str) -> dict:
        return {
            "role": "user",
            "raw_messages": [
                {"role": "user", "content": _BIG},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text_part}],
                },
                {"role": "user", "content": _SHORT},
            ],
            "texts": [{"contents": [_BIG]}],
        }

    return orjson.dumps(
        {"session_id": "c", "turns": [turn(_BIG), turn(_BIG), turn("other " * 100)]}
    )


def _decode_without_dedup() -> Conversation:
    return Conversation.model_validate_json(_conversation_bytes())


def test_equal_large_strings_become_one_object() -> None:
    conversation = _decode_without_dedup()
    first, second, third = conversation.turns
    assert first.raw_messages[0]["content"] is not second.raw_messages[0]["content"]

    dedupe_conversation_strings(conversation)

    canonical = first.raw_messages[0]["content"]
    assert canonical == _BIG
    assert second.raw_messages[0]["content"] is canonical
    assert third.raw_messages[0]["content"] is canonical
    # Nested content parts and media contents share the same object too.
    assert first.raw_messages[1]["content"][0]["text"] is canonical
    assert second.raw_messages[1]["content"][0]["text"] is canonical
    assert first.texts[0].contents[0] is canonical
    assert third.raw_messages[1]["content"][0]["text"] == "other " * 100


def test_short_strings_are_left_alone() -> None:
    assert len(_SHORT) < DEDUP_MIN_CHARS
    conversation = _decode_without_dedup()
    before = [turn.raw_messages[2]["content"] for turn in conversation.turns]

    dedupe_conversation_strings(conversation)

    assert [turn.raw_messages[2]["content"] for turn in conversation.turns] == before
    assert all(
        turn.raw_messages[2]["content"] is value
        for turn, value in zip(conversation.turns, before, strict=True)
    )


def test_dedup_does_not_change_content_or_serialization() -> None:
    reference = _decode_without_dedup()
    conversation = _decode_without_dedup()

    dedupe_conversation_strings(conversation)

    assert conversation == reference
    assert conversation.model_dump_json() == reference.model_dump_json()


def test_memory_map_decode_dedupes() -> None:
    conversation = MemoryMapDatasetClient._deserialize_conversation(
        None, _conversation_bytes()
    )
    first, second, _ = conversation.turns
    assert second.raw_messages[0]["content"] is first.raw_messages[0]["content"]
