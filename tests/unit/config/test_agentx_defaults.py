# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit choices take precedence over the AgentX CLI preset."""

import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config


def test_explicit_cli_values_override_soft_defaults() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
            benchmark_duration=1200,
            use_server_token_count=False,
            gpu_telemetry=["http://localhost:9400/metrics"],
        )
    )
    assert config.benchmark.get_profiling_phases()[0].duration == 1200
    assert config.benchmark.endpoint.use_server_token_count is False
    assert config.benchmark.gpu_telemetry.enabled is True


def test_duration_warmup_does_not_inject_request_warmup() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
            agentic_warmup=True,
            agentic_cache_warmup_duration=60,
        )
    )
    phase = config.benchmark.get_profiling_phases()[0]
    assert phase.agentic_cache_warmup_duration == 60
    assert phase.warmup_requests_per_lane is None


def test_agentx_defaults_skip_warmup_and_export_windows() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
        )
    )
    phase = config.benchmark.get_profiling_phases()[0]
    assert phase.agentic_warmup is False
    assert phase.warmup_requests_per_lane is None
    assert phase.agentic_warmup_grace_period is None
    assert config.benchmark.artifacts.window_start_interval == 60


def test_agentic_warmup_restores_preset_warmup_tuning() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
            agentic_warmup=True,
        )
    )
    phase = config.benchmark.get_profiling_phases()[0]
    assert phase.agentic_warmup is True
    assert phase.warmup_requests_per_lane == 10
    assert phase.agentic_warmup_grace_period == 1800


def test_explicit_warmup_budget_without_agentic_warmup_raises_error() -> None:
    with pytest.raises(ValueError, match="requires --agentic-warmup"):
        resolve_config(
            CLIConfig(
                scenario="agentx",
                model_names=["test-model"],
                public_dataset="semianalysis_cc_traces_weka_062126",
                warmup_requests_per_lane=1,
            )
        )


@pytest.mark.parametrize(
    "flags",
    [
        {"warmup_duration": 60},
        {"warmup_request_count": 10},
        {"warmup_num_sessions": 5},
    ],
)
def test_generic_warmup_flags_without_agentic_warmup_raise_error(
    flags: dict,
) -> None:
    with pytest.raises(ValueError, match="have no effect under --scenario agentx"):
        resolve_config(
            CLIConfig(
                scenario="agentx",
                model_names=["test-model"],
                public_dataset="semianalysis_cc_traces_weka_062126",
                **flags,
            )
        )


def test_window_start_interval_zero_disables_preset_windows() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
            window_start_interval=0,
        )
    )
    assert config.benchmark.artifacts.window_start_interval is None
