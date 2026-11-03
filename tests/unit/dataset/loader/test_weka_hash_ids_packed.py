# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed hash IDs must behave like the list[int] they replace."""

import pickle
import sys

import pytest
from pytest import param

from aiperf.dataset.loader.weka_trace_models import (
    HashIds,
    WekaNormalRequest,
    WekaStreamingRequest,
)


def _request(
    cls: type, hash_ids: list[int]
) -> WekaNormalRequest | WekaStreamingRequest:
    return cls.model_validate(
        {
            "t": 0,
            "model": "m",
            "in": 10,
            "out": 2,
            "hash_ids": hash_ids,
            "type": "n" if cls is WekaNormalRequest else "s",
        }
    )


@pytest.mark.parametrize("cls", [WekaNormalRequest, WekaStreamingRequest])
@pytest.mark.parametrize(
    "ids",
    [
        param([], id="empty"),
        param([0, 1, 2, 30110], id="small"),
        param([2**62, -(2**62), 7], id="int64-edges"),
    ],
)  # fmt: skip
def test_hash_ids_pack_and_behave_like_lists(cls: type, ids: list[int]) -> None:
    req = _request(cls, ids)
    assert isinstance(req.hash_ids, HashIds)
    assert req.hash_ids == ids
    assert (req.hash_ids != ids) is False  # __ne__ is custom too
    assert list(req.hash_ids) == ids and len(req.hash_ids) == len(ids)
    assert [h for h in req.hash_ids] == ids
    assert list(req.hash_ids[1:]) == ids[1:]
    assert all(h in req.hash_ids for h in ids)
    assert set(req.hash_ids) == set(ids)
    assert bool(req.hash_ids) == bool(ids)
    assert req.model_dump()["hash_ids"] == ids
    assert req.model_dump(mode="json")["hash_ids"] == ids
    assert cls.model_validate_json(req.model_dump_json()).hash_ids == ids
    restored = pickle.loads(pickle.dumps(req))
    assert isinstance(restored.hash_ids, HashIds) and restored.hash_ids == ids


def test_hash_ids_outside_int64_keep_the_list() -> None:
    ids = [2**70, 1]
    req = _request(WekaNormalRequest, ids)
    assert type(req.hash_ids) is list and req.hash_ids == ids


def test_hash_ids_are_compact() -> None:
    ids = list(range(100_000))
    packed = _request(WekaNormalRequest, ids).hash_ids
    assert sys.getsizeof(packed) < sys.getsizeof(ids) + 28 * len(ids) // 4
