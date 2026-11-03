# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario presets respect explicit choices during CLI conversion."""

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.scenario import SCENARIOS, apply_scenario, get_scenario
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.resolution.plan import BenchmarkRun


@pytest.fixture
def preset(monkeypatch: pytest.MonkeyPatch) -> str:
    scenario = get_scenario("inferencex-agentx-mvp").model_copy(
        update={
            "name": "test-preset",
            "cli_defaults": {
                "use_server_token_count": True,
                "no_gpu_telemetry": True,
            },
        }
    )
    monkeypatch.setitem(SCENARIOS, scenario.name, scenario)
    return scenario.name


def test_explicit_cli_values_override_defaults(preset: str) -> None:
    config = resolve_config(
        CLIConfig(
            scenario=preset,
            model_names=["test-model"],
            benchmark_duration=1200,
            use_server_token_count=False,
            gpu_telemetry=["http://localhost:9400/metrics"],
        )
    )
    assert config.benchmark.get_profiling_phases()[0].duration == 1200
    assert config.benchmark.endpoint.use_server_token_count is False
    assert config.benchmark.gpu_telemetry.enabled is True


def test_scenario_duration_prevents_fallback_request_limit(preset: str) -> None:
    config = resolve_config(
        CLIConfig(
            scenario=preset,
            model_names=["test-model"],
        )
    )
    phase = config.benchmark.get_profiling_phases()[0]
    assert phase.duration == get_scenario(preset).default_benchmark_duration_seconds
    assert phase.requests is None


@pytest.mark.parametrize(
    "file_scenario,cli_scenario,expected_scenario",
    [
        param(None, "inferencex-agentx-mvp", "inferencex-agentx-mvp", id="enable"),
        param("agentx", "inferencex-agentx-mvp", "inferencex-agentx-mvp", id="override"),
        param("agentx", None, "agentx", id="preserve"),
        param(None, None, None, id="unset"),
    ],
)  # fmt: skip
def test_resolve_config_file_scenario_cli_precedence(
    tmp_path: Path,
    file_scenario: str | None,
    cli_scenario: str | None,
    expected_scenario: str | None,
) -> None:
    config_path = tmp_path / "benchmark.json"
    config_path.write_bytes(
        orjson.dumps(
            {
                "benchmark": {
                    "models": ["test-model"],
                    "scenario": file_scenario,
                    "endpoint": {"urls": ["http://localhost:8000"]},
                    "datasets": [{"name": "main", "type": "synthetic"}],
                    "phases": [
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "duration": 1200,
                            "concurrency": 1,
                        }
                    ],
                },
            }
        )
    )
    cli = CLIConfig(config_file=config_path)
    if cli_scenario is not None:
        cli = cli.model_copy(update={"scenario": cli_scenario})

    config = resolve_config(cli)

    assert config.benchmark.scenario == expected_scenario


@pytest.mark.parametrize(
    "file_unsafe,cli_unsafe,expected_unsafe",
    [
        param(False, True, True, id="enable"),
        param(True, False, False, id="disable"),
        param(True, None, True, id="preserve-true"),
        param(False, None, False, id="preserve-false"),
    ],
)  # fmt: skip
def test_resolve_config_file_unsafe_override_cli_precedence(
    tmp_path: Path,
    file_unsafe: bool,
    cli_unsafe: bool | None,
    expected_unsafe: bool,
) -> None:
    config_path = tmp_path / "benchmark.json"
    config_path.write_bytes(
        orjson.dumps(
            {
                "benchmark": {
                    "models": ["test-model"],
                    "unsafe_override": file_unsafe,
                    "endpoint": {"urls": ["http://localhost:8000"]},
                    "datasets": [{"name": "main", "type": "synthetic"}],
                    "phases": [
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "duration": 1200,
                            "concurrency": 1,
                        }
                    ],
                },
            }
        )
    )
    cli = CLIConfig(config_file=config_path)
    if cli_unsafe is not None:
        cli = cli.model_copy(update={"unsafe_override": cli_unsafe})

    config = resolve_config(cli)

    assert config.benchmark.unsafe_override is expected_unsafe


def test_resolve_config_file_cli_scenario_injects_ignore_eos(tmp_path: Path) -> None:
    base = resolve_config(
        CLIConfig(
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126_256k",
            benchmark_duration=1200,
        )
    )
    config_path = tmp_path / "benchmark.json"
    config_path.write_bytes(
        orjson.dumps(base.model_dump(mode="json", exclude_unset=True))
    )

    config = resolve_config(
        CLIConfig(
            config_file=config_path,
            scenario="inferencex-agentx-mvp",
        )
    )
    run = BenchmarkRun(
        benchmark_id="scenario-test",
        cfg=config.benchmark,
        artifact_dir=tmp_path / "artifacts",
    )
    apply_scenario(run)

    assert run.cfg.endpoint.extra["ignore_eos"] is True
