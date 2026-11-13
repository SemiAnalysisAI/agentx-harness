# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup credit counts follow the ``trajectories`` list they were computed from."""

from unittest.mock import MagicMock

from aiperf.timing.trajectory_source import (
    ConversationState,
    Trajectory,
    TrajectorySnapshot,
    TrajectorySource,
)
from tests.unit.timing.test_trajectory_source import (
    _make_timestamped_subagent_dataset,
)


def _trajectory(lane: int, warmable: int) -> Trajectory:
    states = tuple(
        ConversationState(
            conversation_id=f"c{lane}",
            x_correlation_id=f"x{lane}-{k}",
            next_turn_index=3 if k < warmable else 0,
        )
        for k in range(3)
    )
    return Trajectory(
        conversation_id=f"c{lane}",
        start_turn_index=0,
        snapshot=TrajectorySnapshot(t_star_ms=0.0, states=states),
    )


def _assembled_source(trajectories: list[Trajectory]) -> TrajectorySource:
    source = TrajectorySource.__new__(TrajectorySource)
    source.trajectories = trajectories
    return source


def _built_source() -> TrajectorySource:
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = ["trace"]
    return TrajectorySource(
        dataset_metadata=_make_timestamped_subagent_dataset(),
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=123,
        start_min_ratio=0.675,
        start_max_ratio=0.675,
    )


def test_warmup_counts_reassign_after_first_read_reflects_new_trajectories() -> None:
    source = _built_source()
    assert source.warmup_credit_count == 2

    # The warmup handoff (``finalize_phase``) replaces the list wholesale.
    source.trajectories = [Trajectory(conversation_id="trace", start_turn_index=0)]

    assert source.warmup_credit_counts_by_lane == (1,)
    assert source.warmup_credit_count == 1


def test_warmup_counts_assembled_source_follows_replacement() -> None:
    source = _assembled_source([_trajectory(0, 1), _trajectory(1, 3)])
    assert source.warmup_credit_counts_by_lane == (1, 3)
    assert source.warmup_credit_count == 4

    source.trajectories = [_trajectory(0, 2)]

    assert source.warmup_credit_counts_by_lane == (2,)
    assert source.warmup_credit_count == 2


def test_warmup_counts_are_computed_once_per_list() -> None:
    trajectories = [_trajectory(0, 1), _trajectory(1, 3)]
    source = _assembled_source(trajectories)
    assert source.warmup_credit_count == 4

    # Only assignment invalidates; an in-place edit is not observed, which
    # shows the read is served from the cache rather than recomputed.
    trajectories.append(_trajectory(2, 2))

    assert source.warmup_credit_counts_by_lane == (1, 3)
    assert source.warmup_credit_count == 4
