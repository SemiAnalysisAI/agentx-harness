# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the branded CLI through upstream parsing and configuration."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pytest import param

import agentx_cli
from aiperf.cli import app as aiperf_app


@pytest.fixture
def benchmark_runner() -> Iterator[Mock]:
    """Stop after the real CLI parser and benchmark plan builder."""
    with patch("aiperf.cli_runner.run_benchmark") as runner:
        yield runner


def _run(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        agentx_cli.main(argv)
    assert exc.value.code == 0


def test_profile_defaults_to_agentx_and_preserves_explicit_flags(
    benchmark_runner: Mock,
) -> None:
    _run(["profile", "--model", "test-model", "--benchmark-duration", "1200"])

    benchmark_runner.assert_called_once()
    config = benchmark_runner.call_args.args[0].configs[0]
    assert config.scenario == "agentx"
    assert config.get_profiling_phases()[0].duration == 1200


@pytest.mark.parametrize(
    "scenario_args",
    [
        param(["--scenario", "inferencex-agentx-mvp"], id="separate"),
        param(["--scenario=inferencex-agentx-mvp"], id="equals"),
    ],
)  # fmt: skip
def test_profile_preserves_explicit_scenario(
    scenario_args: list[str], benchmark_runner: Mock
) -> None:
    _run(["profile", "--model", "test-model", *scenario_args])

    config = benchmark_runner.call_args.args[0].configs[0]
    assert config.scenario == "inferencex-agentx-mvp"


@pytest.mark.parametrize(
    "config_flag",
    [
        param("--config", id="long"),
        param("-f", id="short"),
        param("--config=", id="equals"),
        param("-f=", id="short-equals"),
        param("-fjoined", id="short-joined"),
    ],
)  # fmt: skip
@pytest.mark.parametrize("scenario", [None, "inferencex-agentx-mvp"])
def test_profile_preserves_yaml_scenario(
    config_flag: str,
    scenario: str | None,
    tmp_path: Path,
    benchmark_runner: Mock,
) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        "benchmark:\n"
        "  models: [test-model]\n"
        "  endpoint: {urls: [http://localhost:8000]}\n"
        "  datasets: [{name: default, type: synthetic, entries: 1}]\n"
        "  phases: [{name: profiling, type: concurrency, requests: 1, concurrency: 1}]\n"
        f"  scenario: {scenario or 'null'}\n"
    )
    config_args = (
        [f"{config_flag.removesuffix('joined')}{config_path}"]
        if config_flag.endswith(("=", "joined"))
        else [config_flag, str(config_path)]
    )
    _run(["profile", *config_args])

    config = benchmark_runner.call_args.args[0].configs[0]
    assert config.scenario == scenario


@pytest.mark.parametrize("argv", [[], ["--help"], ["profile", "--help"]])
def test_help_uses_agentx_command(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    _run(argv)

    output = capsys.readouterr().out
    assert "Usage: agentx" in output
    if "profile" not in argv:
        assert "AgentX Harness" in output


def test_version_uses_agentx_distribution(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("agentx_cli.version", return_value="1.2.3") as get_version:
        _run(["--version"])

    get_version.assert_called_once_with("agentx-harness")
    assert capsys.readouterr().out.strip() == "1.2.3"


def test_compatibility_cli_keeps_upstream_defaults(benchmark_runner: Mock) -> None:
    with pytest.raises(SystemExit) as exc:
        aiperf_app(["profile", "--model", "test-model", "--request-count", "1"])
    assert exc.value.code == 0

    config = benchmark_runner.call_args.args[0].configs[0]
    assert config.scenario is None
    assert aiperf_app.name == ("aiperf",)
