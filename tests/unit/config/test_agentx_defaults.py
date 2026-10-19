# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit choices take precedence over the AgentX CLI preset."""

import pytest
from pytest import param

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


@pytest.mark.parametrize(
    ("cli_overrides", "expected"),
    [
        param({}, None, id="omitted-disables-timeslices"),
        param({"slice_duration": 5.0}, 5.0, id="explicit-enables-timeslices"),
    ],
)
def test_slice_duration_is_opt_in(
    cli_overrides: dict[str, float], expected: float | None
) -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
            **cli_overrides,
        )
    )
    assert config.benchmark.artifacts.slice_duration == expected


def test_duration_warmup_does_not_inject_request_warmup() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
            agentic_cache_warmup_duration=60,
        )
    )
    phase = config.benchmark.get_profiling_phases()[0]
    assert phase.agentic_cache_warmup_duration == 60
    assert phase.warmup_requests_per_lane is None
