# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone command parity, config precedence, and legacy isolation."""

from __future__ import annotations

import asyncio
import copy
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import orjson
import pytest
from pytest import param

from aiperf.cli_commands._agentx import agentx_environment, resolve_profile_config
from aiperf.common.environment import Environment
from aiperf.common.scenario import ScenarioLockError, apply_scenario
from aiperf.common.scenario.registry import get_scenario
from aiperf.config.dataset import FileDataset
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.agentx import AGENTX_DATASET, apply_cli_defaults
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun


def test_short_command_matches_standalone_flags() -> None:
    short = resolve_config(
        CLIConfig(scenario="inferencex-agentx", model_names=["test-model"])
    )
    full = resolve_config(
        CLIConfig(
            scenario="inferencex-agentx-mvp",
            model_names=["test-model"],
            endpoint_type="chat",
            streaming=True,
            concurrency=8,
            benchmark_duration=3600,
            stats_interval=30,
            random_seed=42,
            failed_request_threshold=0.10,
            trajectory_start_min_ratio=0.25,
            trajectory_start_max_ratio=0.75,
            warmup_requests_per_lane=10,
            warmup_grace_period=1800,
            trace_idle_gap_cap_seconds=300,
            use_server_token_count=True,
            no_gpu_telemetry=True,
            conversation_num_dataset_entries=393,
            slice_duration=1.0,
            public_dataset="semianalysis_cc_traces_weka_062126",
        )
    )
    expected = full.model_dump()
    expected["benchmark"]["scenario"] = "inferencex-agentx"
    assert short.model_dump() == expected
    assert short.benchmark.tokenizer.trust_remote_code is False


def test_cli_parser_accepts_exact_short_command() -> None:
    from aiperf.cli import app

    _, bound, _ = app.parse_args(
        ["profile", "--scenario", "inferencex-agentx", "--url", "localhost:8000"]
    )
    with patch(
        "aiperf.cli_commands._agentx.discover_model",
        new_callable=AsyncMock,
        return_value="test-model",
    ) as discover:
        config = resolve_profile_config(bound.arguments["cli_config"])
    assert config.benchmark.models.items[0].name == "test-model"
    assert config.benchmark.phases[0].concurrency == 8
    discover.assert_awaited_once()


def test_explicit_flags_win_without_mutating_caller() -> None:
    cli = CLIConfig(
        scenario="inferencex-agentx",
        model_names=["alias"],
        tokenizer_name="real/tokenizer",
        concurrency=16,
        benchmark_duration=900,
        random_seed=0,
        trajectory_start_min_ratio=0,
        trajectory_start_max_ratio=1,
        warmup_requests_per_lane=1,
        warmup_grace_period=90,
        use_server_token_count=False,
        gpu_telemetry=["http://localhost:9400/metrics"],
        public_dataset="semianalysis_cc_traces_weka_062126_256k",
    )
    original = cli.model_dump()
    config = resolve_config(cli)
    assert cli.model_dump() == original
    phase = config.benchmark.phases[0]
    assert (phase.concurrency, phase.duration) == (16, 900)
    assert (phase.trajectory_start_min_ratio, phase.trajectory_start_max_ratio) == (
        0,
        1,
    )
    assert phase.warmup_requests_per_lane == 1
    assert phase.agentic_warmup_grace_period == 90
    assert config.random_seed == 0
    assert config.benchmark.gpu_telemetry.enabled
    assert config.benchmark.endpoint.use_server_token_count is False
    assert config.benchmark.tokenizer.name == "real/tokenizer"
    assert config.benchmark.datasets[0].dataset.endswith("_256k")


@pytest.mark.parametrize("scenario", [None, "inferencex-agentx-mvp"])
def test_legacy_flags_are_untouched(scenario: str | None) -> None:
    cli = CLIConfig(scenario=scenario, model_names=["test-model"])
    assert apply_cli_defaults(cli) is cli
    with patch(
        "aiperf.cli_commands._agentx.discover_model", new_callable=AsyncMock
    ) as discover:
        config = resolve_profile_config(cli)
    discover.assert_not_called()
    assert config.random_seed is None
    assert config.benchmark.endpoint.use_server_token_count is False
    assert (
        get_scenario("inferencex-agentx-mvp").default_benchmark_duration_seconds == 1800
    )


def test_explicit_model_skips_discovery() -> None:
    with patch(
        "aiperf.cli_commands._agentx.discover_model", new_callable=AsyncMock
    ) as discover:
        config = resolve_profile_config(
            CLIConfig(
                scenario="inferencex-agentx",
                model_names=["alias"],
                tokenizer_name="real/tokenizer",
            )
        )
    discover.assert_not_called()
    assert config.benchmark.models.items[0].name == "alias"


def test_yaml_preset_and_cli_precedence() -> None:
    data = {
        "benchmark": {
            "scenario": "inferencex-agentx",
            "model": "test-model",
            "endpoint": {"urls": ["http://localhost:9000"]},
            "profiling": {"type": "concurrency", "concurrency": 16},
        }
    }
    before = copy.deepcopy(data)
    config = resolve_config(CLIConfig(concurrency=32), config_dict=data)
    assert data == before
    assert config.benchmark.phases[0].concurrency == 32
    assert config.benchmark.phases[0].duration == 3600
    assert config.benchmark.phases[0].warmup_requests_per_lane == 10
    assert config.benchmark.datasets[0].dataset == AGENTX_DATASET
    assert config.benchmark.datasets[0].trace_idle_gap_cap_seconds == 300
    assert config.random_seed == 42


def test_yaml_explicit_values_and_dataset_are_preserved() -> None:
    data = {
        "random_seed": 0,
        "benchmark": {
            "scenario": "inferencex-agentx",
            "model": "test-model",
            "endpoint": {
                "urls": ["http://localhost:9000"],
                "use_server_token_count": False,
            },
            "dataset": {"type": "public", "dataset": AGENTX_DATASET, "entries": 7},
            "profiling": {"type": "concurrency", "concurrency": 2, "duration": 900},
            "gpu_telemetry": {"enabled": True},
            "artifacts": {"slice_duration": 2.0},
        },
    }
    config = resolve_config(CLIConfig(), config_dict=data)
    assert config.random_seed == 0
    assert config.benchmark.endpoint.use_server_token_count is False
    assert config.benchmark.datasets[0].entries == 7
    assert config.benchmark.datasets[0].trace_idle_gap_cap_seconds is None
    assert config.benchmark.phases[0].concurrency == 2
    assert config.benchmark.phases[0].duration == 900
    assert config.benchmark.gpu_telemetry.enabled
    assert config.benchmark.artifacts.slice_duration == 2.0


def test_yaml_discovery_uses_effective_endpoint(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "benchmark:\n  scenario: inferencex-agentx\n  endpoint:\n    urls: [http://server:9000]\n    api_key: test-secret\n    headers:\n      X-Base: keep\n"
    )
    with patch(
        "aiperf.cli_commands._agentx.discover_model",
        new_callable=AsyncMock,
        return_value="test-model",
    ) as discover:
        config = resolve_profile_config(
            CLIConfig(config_file=path, headers=[("X-Test", "yes")])
        )
    endpoint = discover.await_args.args[0]
    assert endpoint["urls"] == ["http://server:9000"]
    assert endpoint["api_key"] == "test-secret"
    assert endpoint["headers"] == {"X-Base": "keep", "X-Test": "yes"}
    assert config.benchmark.models.items[0].name == "test-model"


@pytest.mark.parametrize(
    "overrides",
    [
        param({"streaming": False}, id="streaming"),
        param({"benchmark_duration": 60}, id="duration"),
        param({"ignore_trace_delays": True}, id="trace-delays"),
        param({"extra_inputs": [("ignore_eos", False)]}, id="eos"),
        param({"public_dataset": "sharegpt"}, id="wrong-corpus"),
    ],
)  # fmt: skip
def test_preset_preserves_scenario_locks(overrides: dict[str, Any]) -> None:
    config = resolve_config(
        CLIConfig(scenario="inferencex-agentx", model_names=["test-model"], **overrides)
    )
    run = BenchmarkRun(
        benchmark_id="test", cfg=config.benchmark, artifact_dir=Path("/tmp/agentx-test")
    )
    with pytest.raises(ScenarioLockError):
        apply_scenario(run)


def test_unsafe_override_stays_invalid() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="inferencex-agentx",
            model_names=["test-model"],
            streaming=False,
            unsafe_override=True,
        )
    )
    run = BenchmarkRun(
        benchmark_id="test", cfg=config.benchmark, artifact_dir=Path("/tmp/agentx-test")
    )
    assert apply_scenario(run).submission_valid is False


def test_new_scenario_keeps_mvp_invariants() -> None:
    old = get_scenario("inferencex-agentx-mvp").model_dump()
    new = get_scenario("inferencex-agentx").model_dump()
    for field in (
        "name",
        "default_benchmark_duration_seconds",
        "default_trajectory_start_min_ratio",
        "default_trajectory_start_max_ratio",
    ):
        old.pop(field)
        new.pop(field)
    assert old == new


def test_environment_defaults_restore_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for group, field in (
        ("DATASET", "CONFIGURATION_TIMEOUT"),
        ("SERVICE", "PROFILE_CONFIGURE_TIMEOUT"),
        ("HTTP", "TCP_USER_TIMEOUT"),
        ("UI", "REALTIME_METRICS_ENABLED"),
    ):
        monkeypatch.delenv(f"AIPERF_{group}_{field}", raising=False)
        monkeypatch.setattr(Environment, group, type(getattr(Environment, group))())
    original = Environment.model_dump()
    with pytest.raises(RuntimeError), agentx_environment("inferencex-agentx"):
        assert Environment.DATASET.CONFIGURATION_TIMEOUT == 1800
        assert Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT == 1800
        assert Environment.HTTP.TCP_USER_TIMEOUT == 900000
        assert Environment.UI.REALTIME_METRICS_ENABLED is True
        assert os.environ["AIPERF_HTTP_TCP_USER_TIMEOUT"] == "900000"
        raise RuntimeError("stop")
    assert Environment.model_dump() == original
    assert "AIPERF_HTTP_TCP_USER_TIMEOUT" not in os.environ


def test_environment_explicit_values_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIPERF_HTTP_TCP_USER_TIMEOUT", "12345")
    monkeypatch.setattr(Environment, "HTTP", type(Environment.HTTP)())
    with agentx_environment("inferencex-agentx"):
        assert Environment.HTTP.TCP_USER_TIMEOUT == 12345
    assert os.environ["AIPERF_HTTP_TCP_USER_TIMEOUT"] == "12345"


def test_legacy_environment_untouched() -> None:
    before = Environment.model_dump()
    with agentx_environment("inferencex-agentx-mvp"):
        assert Environment.model_dump() == before


def test_artifact_directory_is_unique_and_overridable(tmp_path: Path) -> None:
    cli = CLIConfig(scenario="inferencex-agentx", model_names=["test-model"])
    first = resolve_profile_config(cli)
    second = resolve_profile_config(cli)
    assert first.benchmark.artifacts.dir != second.benchmark.artifacts.dir
    assert first.benchmark.artifacts.dir.parent == Path("artifacts")
    explicit = resolve_profile_config(
        cli.model_copy(update={"artifact_directory": tmp_path})
    )
    assert explicit.benchmark.artifacts.dir == tmp_path


def test_warmup_duration_override_does_not_add_request_budget() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="inferencex-agentx",
            model_names=["test-model"],
            agentic_cache_warmup_duration=30,
        )
    )
    assert config.benchmark.phases[0].agentic_cache_warmup_duration == 30
    assert config.benchmark.phases[0].warmup_requests_per_lane is None


def test_yaml_cli_overrides_do_not_conflict_with_injected_defaults() -> None:
    data = {
        "random_seed": 99,
        "benchmark": {
            "scenario": "inferencex-agentx",
            "model": "test-model",
            "endpoint": {"urls": ["http://localhost:9000"]},
        },
    }
    config = resolve_config(
        CLIConfig(
            random_seed=0,
            agentic_cache_warmup_duration=30,
            hf_weka_dataset="untrusted/corpus",
        ),
        config_dict=data,
    )
    assert config.random_seed == 0
    assert config.benchmark.datasets[0].random_seed == 0
    assert config.benchmark.datasets[0].dataset == "weka_hf"
    assert config.benchmark.phases[0].warmup_requests_per_lane is None


def test_yaml_defaults_match_cli_defaults() -> None:
    cli = resolve_config(
        CLIConfig(scenario="inferencex-agentx", model_names=["test-model"])
    )
    yaml = resolve_config(
        CLIConfig(),
        config_dict={
            "benchmark": {
                "scenario": "inferencex-agentx",
                "model": "test-model",
                "endpoint": {"urls": ["http://localhost:8000"]},
            }
        },
    )
    # UI selection is CLI/TTY-specific; workload and reporting settings must agree.
    for section in (
        "models",
        "endpoint",
        "datasets",
        "phases",
        "gpu_telemetry",
        "artifacts",
    ):
        assert getattr(cli.benchmark, section) == getattr(yaml.benchmark, section)


def test_new_name_rejects_unpinned_hf_corpus() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="inferencex-agentx",
            model_names=["test-model"],
            hf_weka_dataset="untrusted/corpus",
        )
    )
    run = BenchmarkRun(
        benchmark_id="test", cfg=config.benchmark, artifact_dir=Path("/tmp/agentx-test")
    )
    with pytest.raises(ScenarioLockError, match="hf_weka_dataset"):
        apply_scenario(run)


def test_new_name_rejects_local_unpinned_corpus() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="inferencex-agentx",
            model_names=["test-model"],
        )
    )
    run = BenchmarkRun(
        benchmark_id="test", cfg=config.benchmark, artifact_dir=Path("/tmp/agentx-test")
    )
    run.cfg.datasets = [
        FileDataset(name="main", type="file", path="/unused", format="weka_trace")
    ]
    run.resolved.dataset_types = {"main": "weka_trace"}
    with pytest.raises(ScenarioLockError, match="cannot verify corpus identity"):
        apply_scenario(run)


def test_new_name_rejects_sweeps() -> None:
    with pytest.raises(ValueError, match="sweep"):
        resolve_config(
            CLIConfig(
                scenario="inferencex-agentx",
                model_names=["test-model"],
                concurrency=[1, 2],
            )
        )


def test_profile_passes_resolved_plan_and_environment_to_runner() -> None:
    from aiperf.cli_commands.profile import profile

    def check_plan(plan: BenchmarkPlan) -> None:
        config = plan.configs[0]
        assert config.scenario == "inferencex-agentx"
        assert config.models.items[0].name == "test-model"
        assert Environment.DATASET.CONFIGURATION_TIMEOUT == 1800
        run = BenchmarkRun(
            benchmark_id="test", cfg=config, artifact_dir=Path("/tmp/agentx-test")
        )
        assert apply_scenario(run).submission_valid is True

    with (
        patch(
            "aiperf.cli_commands._agentx.discover_model",
            new_callable=AsyncMock,
            return_value="test-model",
        ),
        patch("aiperf.cli_runner.run_benchmark", side_effect=check_plan) as runner,
    ):
        profile(
            cli_config=CLIConfig(scenario="inferencex-agentx", urls=["localhost:8000"])
        )
    runner.assert_called_once()


@pytest.mark.asyncio
async def test_environment_defaults_reach_child_process() -> None:
    with agentx_environment("inferencex-agentx"):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from aiperf.common.environment import Environment; import orjson; "
            "print(orjson.dumps([Environment.DATASET.CONFIGURATION_TIMEOUT, "
            "Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT, "
            "Environment.HTTP.TCP_USER_TIMEOUT, Environment.UI.REALTIME_METRICS_ENABLED]).decode())",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()
    assert orjson.loads(stdout) == [1800, 1800, 900000, True]
