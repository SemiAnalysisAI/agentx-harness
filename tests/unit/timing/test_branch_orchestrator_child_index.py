# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-parent tracked-child index must always equal the brute-force walk.

``_ChildJoinMap`` keeps ``BranchOrchestrator._tracked_children_by_parent`` in
step with ``_child_to_join`` so ``_finalize_failed_dispatches`` answers "does
this parent still have a tracked child?" in O(1). These tests churn the map
through every mutating path and compare against the brute-force reference
``any_child_tracked_for_parent``.
"""

from __future__ import annotations

import random
from collections import Counter
from unittest.mock import MagicMock

import pytest

from aiperf.common.models import DatasetMetadata
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import (
    BranchOrchestrator,
    ChildJoinEntry,
    _ChildJoinMap,
    any_child_tracked_for_parent,
)


def _entries(parent: str, n: int = 1) -> list[ChildJoinEntry]:
    return [
        ChildJoinEntry(
            parent_correlation_id=parent,
            gated_turn_index=None if i == 0 else i,
            prereq_key=None if i == 0 else f"SPAWN_JOIN:{parent}:{i}",
        )
        for i in range(n)
    ]


def _brute_counts(mapping: dict[str, list[ChildJoinEntry]]) -> dict[str, int]:
    return dict(
        Counter(e.parent_correlation_id for ents in mapping.values() for e in ents)
    )


def _assert_consistent(
    mapping: dict[str, list[ChildJoinEntry]],
    counts: dict[str, int],
    parents: list[str],
) -> None:
    assert counts == _brute_counts(mapping)
    assert all(v > 0 for v in counts.values()), "zero counts must be dropped"
    for parent in parents:
        assert bool(counts.get(parent, 0)) == any_child_tracked_for_parent(
            mapping, parent
        )


def _orch() -> BranchOrchestrator:
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=[], sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    return BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())


PARENTS = ["p0", "p1", "p2"]


def test_orchestrator_wires_index_to_map():
    orch = _orch()
    assert isinstance(orch._child_to_join, _ChildJoinMap)
    assert orch._child_to_join == {}
    orch._child_to_join["c"] = _entries("p0", 2)
    assert orch._tracked_children_by_parent == {"p0": 2}
    orch._child_to_join.clear()
    assert orch._tracked_children_by_parent == {}


def test_index_tracks_each_mutation_path():
    counts: dict[str, int] = {}
    m = _ChildJoinMap(counts)

    m["a"] = _entries("p0")
    m["b"] = _entries("p0", 3)
    m["c"] = _entries("p1", 2)
    _assert_consistent(m, counts, PARENTS)
    assert counts == {"p0": 4, "p1": 2}

    # Overwrite the same child: old contribution removed before new added,
    # including when the overwrite moves it to a different parent.
    m["b"] = _entries("p0", 1)
    _assert_consistent(m, counts, PARENTS)
    m["b"] = _entries("p2", 2)
    _assert_consistent(m, counts, PARENTS)
    assert counts == {"p0": 1, "p1": 2, "p2": 2}

    del m["a"]
    _assert_consistent(m, counts, PARENTS)
    assert "p0" not in counts

    with pytest.raises(KeyError):
        del m["a"]
    with pytest.raises(KeyError):
        m.pop("a")
    assert m.pop("a", None) is None
    sentinel: list[ChildJoinEntry] = []
    assert m.pop("a", sentinel) is sentinel
    _assert_consistent(m, counts, PARENTS)

    popped = m.pop("c")
    assert [e.parent_correlation_id for e in popped] == ["p1", "p1"]
    _assert_consistent(m, counts, PARENTS)

    assert m.pop("b", []) == _entries("p2", 2)
    _assert_consistent(m, counts, PARENTS)
    assert counts == {} and m == {}

    # Empty entry lists contribute nothing on either side.
    m["e"] = []
    _assert_consistent(m, counts, PARENTS)
    del m["e"]

    m.update({"x": _entries("p0"), "y": _entries("p1")}, z=_entries("p1", 2))
    _assert_consistent(m, counts, PARENTS)
    m |= {"x": _entries("p2")}
    _assert_consistent(m, counts, PARENTS)
    assert m.setdefault("x") == _entries("p2")
    m.setdefault("w", _entries("p0"))
    _assert_consistent(m, counts, PARENTS)
    key, _ = m.popitem()
    assert key not in m
    _assert_consistent(m, counts, PARENTS)

    m.clear()
    _assert_consistent(m, counts, PARENTS)
    assert counts == {}


def test_index_equals_brute_force_under_random_churn():
    rng = random.Random(20261108)
    counts: dict[str, int] = {}
    m = _ChildJoinMap(counts)
    parents = [f"p{i}" for i in range(6)]
    children = [f"c{i}" for i in range(40)]
    for step in range(5000):
        op = rng.random()
        child = rng.choice(children)
        if op < 0.45:
            m[child] = _entries(rng.choice(parents), rng.randint(0, 3))
        elif op < 0.65:
            m.pop(child, None)
        elif op < 0.8:
            if child in m:
                del m[child]
        elif op < 0.9:
            if child in m:
                m.pop(child)
        elif op < 0.995:
            m.update({rng.choice(children): _entries(rng.choice(parents), 2)})
        else:
            m.clear()
        if step % 7 == 0:
            _assert_consistent(m, counts, parents)
    _assert_consistent(m, counts, parents)
