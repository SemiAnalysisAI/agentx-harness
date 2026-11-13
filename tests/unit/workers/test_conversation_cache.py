# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-worker decoded-conversation cache: session clones stay private, and the
LRU bound follows the worker's live-session count."""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiperf.common.models import Conversation, Image, Text, Turn
from aiperf.workers import worker as worker_module
from aiperf.workers.session_manager import UserSessionManager
from aiperf.workers.worker import Worker, _clone_conversation_for_session


def _conversation() -> Conversation:
    return Conversation(
        session_id="c0",
        turns=[
            Turn(
                raw_messages=[{"role": "user", "content": "hello"}],
                texts=[Text(name="text", contents=["prompt"])],
                images=[Image(name="image", contents=["data:image/png;base64,AAA"])],
                raw_payload={"k": "v"},
            )
        ],
    )


def test_clone_containers_are_private_to_the_session() -> None:
    source = _conversation()
    clone = _clone_conversation_for_session(source)
    turn = clone.turns[0]

    # The slot writes the cache-bust and media-uuid mutators perform.
    turn.raw_messages[0] = {"role": "user", "content": "busted"}
    turn.texts[0].contents[0] = "busted"
    turn.images[:] = [Image(name="image", contents=["other"])]
    turn.raw_payload["k"] = "busted"
    clone.turns.append(Turn())

    original = source.turns[0]
    assert len(source.turns) == 1
    assert original.raw_messages == [{"role": "user", "content": "hello"}]
    assert original.texts[0].contents == ["prompt"]
    assert original.images[0].contents == ["data:image/png;base64,AAA"]
    assert original.raw_payload == {"k": "v"}


def test_clone_shares_the_immutable_payload() -> None:
    source = _conversation()
    clone = _clone_conversation_for_session(source)

    assert clone.turns[0].raw_messages[0] is source.turns[0].raw_messages[0]
    assert clone.turns[0].images[0] is source.turns[0].images[0]


def _cache_owner(live_sessions: int) -> SimpleNamespace:
    session_manager = UserSessionManager()
    for i in range(live_sessions):
        session_manager.store(f"x{i}", SimpleNamespace())
    return SimpleNamespace(
        session_manager=session_manager,
        _conversation_cache=OrderedDict(),
        _retrieve_conversation_for_session=AsyncMock(
            side_effect=lambda **_: _conversation()
        ),
    )


async def _borrow(owner: SimpleNamespace, conversation_id: str) -> Conversation:
    credit = SimpleNamespace(conversation_id=conversation_id, num_turns=1)
    return await Worker._borrow_conversation_for_session(
        owner, credit_context=SimpleNamespace(credit=credit)
    )


async def test_cache_bound_is_the_floor_with_few_live_sessions(monkeypatch) -> None:
    monkeypatch.setattr(worker_module, "_CONVERSATION_CACHE_MIN_ENTRIES", 2)
    owner = _cache_owner(live_sessions=1)

    for cid in ("a", "b", "c"):
        await _borrow(owner, cid)

    assert [key[0] for key in owner._conversation_cache] == ["b", "c"]


async def test_cache_bound_grows_with_live_sessions(monkeypatch) -> None:
    monkeypatch.setattr(worker_module, "_CONVERSATION_CACHE_MIN_ENTRIES", 2)
    owner = _cache_owner(live_sessions=4)

    for cid in ("a", "b", "c", "d", "a"):
        await _borrow(owner, cid)

    assert owner._retrieve_conversation_for_session.await_count == 4
    assert [key[0] for key in owner._conversation_cache] == ["b", "c", "d", "a"]


async def test_cache_shrinks_when_live_sessions_drain(monkeypatch) -> None:
    monkeypatch.setattr(worker_module, "_CONVERSATION_CACHE_MIN_ENTRIES", 2)
    owner = _cache_owner(live_sessions=4)
    for cid in ("a", "b", "c", "d"):
        await _borrow(owner, cid)

    for i in range(4):
        owner.session_manager.evict(f"x{i}")
    await _borrow(owner, "e")

    assert [key[0] for key in owner._conversation_cache] == ["d", "e"]
