# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AIPerfConfig sweep cross-field validators.

Post-redesign, the only envelope-level cross-field validator that remains
on AIPerfConfig is ``validate_sweep_no_dashboard_ui``; the ex-parameter
sweep field validators (same-seed-needs-seed, cooldown-non-neg,
flags-require-sweep) all moved when their fields moved off MultiRunConfig
onto SweepConfig sub-objects, where Pydantic's per-field constraints
(``ge=0``) and the discriminated SweepConfig union enforce them
structurally.
"""

from __future__ import annotations

import re

import pytest
from pytest import param

from aiperf.config.config import AIPerfConfig

_BASE_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {"name": "profiling", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


_ENVELOPE_KEYS = {"sweep", "multi_run", "variables", "random_seed"}


def _make(**overrides) -> AIPerfConfig:
    env_kwargs = {k: overrides.pop(k) for k in list(overrides) if k in _ENVELOPE_KEYS}
    body = {**_BASE_KWARGS, **overrides}
    return AIPerfConfig(benchmark=body, **env_kwargs)


# ---------------------------------------------------------------------------
# validate_sweep_no_dashboard_ui — only AIPerfConfig-scope validator left
# ---------------------------------------------------------------------------


def test_sweep_with_dashboard_ui_rejected() -> None:
    with pytest.raises(ValueError, match="Dashboard UI is incompatible"):
        _make(
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [10, 20]},
            },
            runtime={"ui": "dashboard"},
        )


def test_sweep_with_simple_ui_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [10, 20]},
        },
        runtime={"ui": "simple"},
    )
    assert cfg.sweep is not None


# ---------------------------------------------------------------------------
# _reject_scenario_with_sweep — a fixed-spec scenario lock forbids sweeps.
# A scenario locks ONE configuration; a sweep would fan it into N diverging
# variations, each individually satisfying the lock (the falsification the v1
# list-shaped --concurrency rejection prevented). In v2 magic-list flags are
# hoisted to a sweep block before the config is built, so the rejection lives
# here at the envelope level.
# ---------------------------------------------------------------------------


def test_scenario_with_sweep_rejected() -> None:
    with pytest.raises(ValueError, match="does not support parameter sweeps"):
        _make(
            scenario="inferencex-agentx-mvp",
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1, 2, 3]},
            },
            runtime={"ui": "simple"},
        )


def test_scenario_without_sweep_accepted() -> None:
    cfg = _make(scenario="inferencex-agentx-mvp")
    assert cfg.benchmark.scenario == "inferencex-agentx-mvp"
    assert cfg.sweep is None


def test_sweep_without_scenario_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [1, 2, 3]},
        },
        runtime={"ui": "simple"},
    )
    assert cfg.benchmark.scenario is None
    assert cfg.sweep is not None


def test_scenario_with_sweep_unsafe_override_warns_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        cfg = _make(
            scenario="inferencex-agentx-mvp",
            unsafe_override=True,
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1, 2, 3]},
            },
            runtime={"ui": "simple"},
        )
    assert cfg.sweep is not None
    assert any(
        "does not support parameter sweeps" in r.message
        for r in caplog.records
        if r.levelname == "WARNING"
    )


# ---------------------------------------------------------------------------
# Cooldown / same_seed / iteration_order moved to GridSweep — verified
# structurally there. AIPerfConfig no longer enforces these.
# ---------------------------------------------------------------------------


def test_grid_sweep_negative_cooldown_rejected_by_field_constraint() -> None:
    """``GridSweep.cooldown_seconds`` carries ``ge=0``; bare AIPerfConfig
    construction surfaces the Pydantic field error directly."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        _make(
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [10, 20]},
                "cooldown_seconds": -1.0,
            },
            runtime={"ui": "simple"},
        )


def test_grid_sweep_zero_cooldown_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [10, 20]},
            "cooldown_seconds": 0.0,
        },
        runtime={"ui": "simple"},
    )
    assert isinstance(cfg.sweep, type(cfg.sweep))
    assert cfg.sweep.cooldown_seconds == 0.0


def test_grid_sweep_positive_cooldown_accepted() -> None:
    cfg = _make(
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [10, 20]},
            "cooldown_seconds": 5.0,
        },
        runtime={"ui": "simple"},
    )
    assert cfg.sweep.cooldown_seconds == 5.0


def test_grid_sweep_same_seed_field_round_trips() -> None:
    """``same_seed`` lives on GridSweep; envelope wiring is structural."""
    cfg = _make(
        random_seed=42,
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [10, 20]},
            "same_seed": True,
        },
        runtime={"ui": "simple"},
    )
    assert cfg.sweep.same_seed is True


# ---------------------------------------------------------------------------
# validate_agentic_cache_warmup — ``--agentic-cache-warmup-duration`` is
# consumed only on the AGENTIC_REPLAY path. On any other run the value is
# silently dropped, so the guard hard-raises rather than accept a no-op flag.
# A scenario governs the timing_mode (stamped post-construction by
# apply_scenario, which never re-runs this gate), so the validator resolves the
# scenario's declared timing_mode from its ScenarioSpec and rejects the flag
# when the scenario is not agentic_replay. A no-scenario config is final and
# resolved from the phases directly.
# ---------------------------------------------------------------------------


def _agentic_phase(**phase_overrides) -> list[dict]:
    return [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
            **phase_overrides,
        }
    ]


def test_agentic_cache_warmup_without_scenario_non_agentic_rejected() -> None:
    with pytest.raises(ValueError, match="requires the agentic_replay"):
        _make(
            phases=_agentic_phase(
                agentic_warmup=True, agentic_cache_warmup_duration=30.0
            )
        )


def test_agentic_cache_warmup_with_agentic_scenario_accepted() -> None:
    """The agentic scenario locks AGENTIC_REPLAY, so the flag is accepted."""
    cfg = _make(
        scenario="inferencex-agentx-mvp",
        phases=_agentic_phase(agentic_warmup=True, agentic_cache_warmup_duration=30.0),
    )
    assert cfg.benchmark.scenario == "inferencex-agentx-mvp"


def test_agentic_cache_warmup_with_non_agentic_scenario_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-agentic scenario must reject the flag, not silently drop it.

    The validator resolves the scenario's declared timing_mode rather than
    deferring; ``apply_scenario`` stamps the mode post-construction and never
    re-runs this gate, so a blanket deferral would let a no-op flag through.
    """
    from aiperf.common.scenario import registry
    from aiperf.plugin.enums import TimingMode

    non_agentic = registry.SCENARIOS["inferencex-agentx-mvp"].model_copy(
        update={"name": "test-non-agentic", "timing_mode": TimingMode.REQUEST_RATE}
    )
    monkeypatch.setitem(registry.SCENARIOS, non_agentic.name, non_agentic)

    with pytest.raises(ValueError, match="requires the agentic_replay"):
        _make(
            scenario="test-non-agentic",
            phases=_agentic_phase(
                agentic_warmup=True, agentic_cache_warmup_duration=30.0
            ),
        )


def test_agentic_cache_warmup_with_explicit_agentic_timing_mode_accepted() -> None:
    cfg = _make(
        phases=_agentic_phase(
            agentic_warmup=True,
            agentic_cache_warmup_duration=30.0,
            timing_mode="agentic_replay",
        ),
    )
    assert cfg.benchmark.phases[0].agentic_cache_warmup_duration == 30.0


def test_agentic_cache_warmup_request_budget_without_duration_accepted() -> None:
    cfg = _make(
        phases=_agentic_phase(
            agentic_warmup=True,
            warmup_requests_per_lane=10,
            timing_mode="agentic_replay",
        )
    )
    assert cfg.benchmark.phases[0].warmup_requests_per_lane == 10


def test_agentic_cache_warmup_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make(
            phases=_agentic_phase(
                agentic_warmup=True,
                agentic_cache_warmup_duration=30.0,
                warmup_requests_per_lane=10,
                timing_mode="agentic_replay",
            )
        )


def test_no_agentic_cache_warmup_duration_accepted() -> None:
    cfg = _make(phases=_agentic_phase())
    assert cfg.benchmark.phases[0].agentic_cache_warmup_duration is None
    assert cfg.benchmark.phases[0].agentic_warmup is False


@pytest.mark.parametrize(
    "overrides,flag",
    [
        param({"agentic_cache_warmup_duration": 30.0}, "--agentic-cache-warmup-duration", id="duration"),
        param({"warmup_requests_per_lane": 10}, "--warmup-requests-per-lane", id="per-lane"),
        param({"agentic_warmup_grace_period": 60.0}, "--agentic-warmup-grace-period", id="grace"),
    ],
)  # fmt: skip
def test_agentic_warmup_tuning_without_agentic_warmup_rejected(
    overrides: dict, flag: str
) -> None:
    with pytest.raises(
        ValueError, match=f"Passing {re.escape(flag)}.* requires --agentic-warmup"
    ):
        _make(phases=_agentic_phase(timing_mode="agentic_replay", **overrides))


def test_agentic_warmup_without_tuning_accepted() -> None:
    cfg = _make(
        phases=_agentic_phase(agentic_warmup=True, timing_mode="agentic_replay")
    )
    assert cfg.benchmark.phases[0].agentic_warmup is True


def test_agentic_warmup_on_non_agentic_run_rejected() -> None:
    with pytest.raises(ValueError, match="requires the agentic_replay"):
        _make(phases=_agentic_phase(agentic_warmup=True))


# ---------------------------------------------------------------------------
# validate_window_start_interval — suffix windows need one trailing profiling
# phase and a bounded window count.
# ---------------------------------------------------------------------------

_PROFILING = {"name": "profiling", "type": "concurrency", "concurrency": 1}
_WARMUP = {"name": "warmup", "type": "concurrency", "requests": 5, "concurrency": 1}


def test_window_start_interval_single_trailing_profiling_phase_accepted() -> None:
    cfg = _make(
        phases=[_WARMUP, {**_PROFILING, "duration": 600}],
        artifacts={"window_start_interval": "1m"},
    )
    assert cfg.benchmark.artifacts.window_start_interval == 60.0


@pytest.mark.parametrize(
    "phases",
    [
        param(
            [
                {**_PROFILING, "requests": 10},
                {**_PROFILING, "name": "storm", "kind": "profiling", "requests": 10},
            ],
            id="two-profiling-phases",
        ),
        param(
            [{**_PROFILING, "requests": 10}, {**_WARMUP, "kind": "warmup"}],
            id="profiling-not-last",
        ),
    ],
)  # fmt: skip
def test_window_start_interval_phase_shape_rejected(phases: list[dict]) -> None:
    with pytest.raises(ValueError, match="exactly one profiling phase"):
        _make(phases=phases, artifacts={"window_start_interval": 60})


def test_window_start_interval_too_many_windows_rejected() -> None:
    with pytest.raises(ValueError, match="more than 256 windows"):
        _make(
            phases=[{**_PROFILING, "duration": 3600}],
            artifacts={"window_start_interval": 1},
        )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_window_start_interval_non_positive_or_non_finite_rejected(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        _make(artifacts={"window_start_interval": value})


def test_window_start_interval_window_count_includes_measurement_start() -> None:
    with pytest.raises(ValueError, match="more than 256 windows"):
        _make(
            phases=[{**_PROFILING, "duration": 250, "measurement_start": 10}],
            artifacts={"window_start_interval": 1},
        )


# ---------------------------------------------------------------------------
# validate_measurement_start — the summary cutoff needs the profiling phase to
# be the only, duration-bounded phase, without the agentic warmup.
# ---------------------------------------------------------------------------


def test_measurement_start_on_sole_profiling_phase_accepted() -> None:
    cfg = _make(phases=[{**_PROFILING, "duration": 600, "measurement_start": "12m"}])
    assert cfg.benchmark.phases[0].measurement_start == 720.0


@pytest.mark.parametrize(
    "phases",
    [
        param(
            [_WARMUP, {**_PROFILING, "duration": 600, "measurement_start": 60}],
            id="warmup-phase-before",
        ),
        param(
            [{**_WARMUP, "measurement_start": 60}, {**_PROFILING, "duration": 600}],
            id="on-warmup-phase",
        ),
    ],
)  # fmt: skip
def test_measurement_start_phase_shape_rejected(phases: list[dict]) -> None:
    with pytest.raises(ValueError, match="profiling phase that is the only phase"):
        _make(phases=phases)


def test_measurement_start_without_duration_rejected() -> None:
    with pytest.raises(ValueError, match="requires --benchmark-duration"):
        _make(phases=[{**_PROFILING, "requests": 10, "measurement_start": 60}])


def test_measurement_start_without_duration_under_scenario_deferred() -> None:
    cfg = _make(
        scenario="inferencex-agentx-mvp",
        phases=_agentic_phase(measurement_start=60),
    )
    assert cfg.benchmark.phases[0].measurement_start == 60.0


def test_measurement_start_with_agentic_warmup_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be combined with --agentic-warmup"):
        _make(
            scenario="inferencex-agentx-mvp",
            phases=_agentic_phase(agentic_warmup=True, measurement_start=60),
        )


@pytest.mark.parametrize("value", [0, None])
def test_measurement_start_zero_or_unset_skips_checks(value: float | None) -> None:
    cfg = _make(
        phases=[_WARMUP, {**_PROFILING, "requests": 10, "measurement_start": value}]
    )
    assert not cfg.benchmark.phases[1].measurement_start


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_measurement_start_negative_or_non_finite_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        _make(phases=[{**_PROFILING, "duration": 600, "measurement_start": value}])
