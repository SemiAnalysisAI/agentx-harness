# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded per-lane stagger of the first per-trace idle-cap watchdog."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiperf.common.environment import Environment
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy


def _fraction(seed: int, trace_id: str, lane: int) -> float | None:
    strategy = SimpleNamespace(conversation_source=SimpleNamespace(random_seed=seed))
    return AgenticReplayStrategy._initial_idle_budget_fraction(
        strategy, SimpleNamespace(conversation_id=trace_id), lane
    )


def test_stagger_initial_idle_cap_on_by_default() -> None:
    field = type(Environment.AGENTX).model_fields["STAGGER_INITIAL_IDLE_CAP"]
    assert field.default is True


def test_initial_idle_budget_fraction_none_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Environment.AGENTX, "STAGGER_INITIAL_IDLE_CAP", False)
    assert _fraction(42, "trace-a", 0) is None


def test_initial_idle_budget_fraction_is_seeded_and_spread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Environment.AGENTX, "STAGGER_INITIAL_IDLE_CAP", True)
    fractions = [_fraction(42, "trace-a", lane) for lane in range(200)]

    assert fractions == [_fraction(42, "trace-a", lane) for lane in range(200)]
    assert all(0.0 < fraction <= 1.0 for fraction in fractions)
    assert len(set(fractions)) == len(fractions)
    assert min(fractions) < 0.1 and max(fractions) > 0.9
    assert _fraction(7, "trace-a", 0) != _fraction(42, "trace-a", 0)
