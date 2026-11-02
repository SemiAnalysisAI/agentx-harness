# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--measurement-start`` reaches the profiling phase from the CLI and over YAML."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config

_YAML = textwrap.dedent("""\
benchmark:
  models:
    - test-model
  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
  datasets:
    - name: default
      type: synthetic
      entries: 100
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: profiling
      type: concurrency
      concurrency: 1
      duration: 600
      measurementStart: 60
""")

_YAML_WITH_WARMUP = _YAML.replace(
    "  phases:\n",
    "  phases:\n"
    "    - name: warmup\n"
    "      type: concurrency\n"
    "      concurrency: 1\n"
    "      requests: 5\n",
)


def _cli(**kwargs) -> CLIConfig:
    """Build a CLIConfig with only the given fields marked explicitly set."""
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_build_profiling_routes_measurement_start() -> None:
    prof = build_profiling(
        _cli(
            url="http://localhost:8000/test",
            model_names=["test-model"],
            benchmark_duration=600,
            measurement_start=120,
            concurrency=1,
        )
    )
    assert prof["measurement_start"] == 120
    assert prof["duration"] == 600


def test_build_profiling_omits_unset_measurement_start() -> None:
    prof = build_profiling(
        _cli(
            url="http://localhost:8000/test",
            model_names=["test-model"],
            benchmark_duration=600,
            concurrency=1,
        )
    )
    assert "measurement_start" not in prof


def test_cli_measurement_start_overrides_yaml(tmp_path: Path) -> None:
    cfg = resolve_config(_cli(measurement_start=120), _write(tmp_path, _YAML))
    assert cfg.benchmark.phases[0].measurement_start == 120.0


def test_yaml_measurement_start_kept_without_cli_flag(tmp_path: Path) -> None:
    cfg = resolve_config(CLIConfig(), _write(tmp_path, _YAML))
    assert cfg.benchmark.phases[0].measurement_start == 60.0


def test_cli_zero_rescues_yaml_measurement_start_with_warmup_phase(
    tmp_path: Path,
) -> None:
    """The cross-field check runs on the merged config, not the YAML alone."""
    cfg = resolve_config(_cli(measurement_start=0), _write(tmp_path, _YAML_WITH_WARMUP))
    assert cfg.benchmark.phases[1].measurement_start == 0.0


def test_yaml_measurement_start_with_warmup_phase_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="profiling phase that is the only phase"):
        resolve_config(CLIConfig(), _write(tmp_path, _YAML_WITH_WARMUP))
