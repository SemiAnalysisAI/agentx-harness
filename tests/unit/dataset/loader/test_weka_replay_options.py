# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the env-driven ``weka_replay`` options (``_weka_replay_options.py``).

Covers each ``AIPERF_DATASET_WEKA_REPLAY_*`` knob's default and override, the
``session_mode`` validation, comma-list parsing for ``include_roles`` /
``sampling_passthrough``, and ``resolve_options`` reading
``--synthesis-max-isl`` / ``--max-context-length`` off a real
``BenchmarkRun``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.environment import Environment
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader._weka_replay_options import (
    WekaReplayOptions,
    resolve_options,
)
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli


def _write_trace(tmp_path: Path) -> Path:
    p = tmp_path / "trace.jsonl"
    p.write_text(
        '{"session_id": "s1", "timestamp": 0, "input_length": 8, "output_length": 4}\n'
    )
    return p


def _run_with(tmp_path: Path, **cli_kwargs: object):
    trace = _write_trace(tmp_path)
    cli = CLIConfig(
        model_names=["test-model"],
        input_file=str(trace),
        custom_dataset_type=CustomDatasetType.WEKA_TRACE,
        **cli_kwargs,
    )
    return make_run_from_cli(cli)


class TestEnvDefaults:
    def test_session_mode_default(self) -> None:
        assert Environment.DATASET.WEKA_REPLAY_SESSION_MODE == "tree"

    def test_include_orphans_default(self) -> None:
        assert Environment.DATASET.WEKA_REPLAY_INCLUDE_ORPHANS is False

    def test_include_ephemeral_default(self) -> None:
        assert Environment.DATASET.WEKA_REPLAY_INCLUDE_EPHEMERAL is False

    def test_include_roles_default(self) -> None:
        assert Environment.DATASET.WEKA_REPLAY_INCLUDE_ROLES == ""

    def test_sampling_passthrough_default(self) -> None:
        assert (
            Environment.DATASET.WEKA_REPLAY_SAMPLING_PASSTHROUGH == "temperature,top_p"
        )


class TestEnvOverrides:
    def test_session_mode_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            Environment.DATASET, "WEKA_REPLAY_SESSION_MODE", "independent"
        )
        assert Environment.DATASET.WEKA_REPLAY_SESSION_MODE == "independent"

    def test_include_orphans_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_INCLUDE_ORPHANS", True)
        assert Environment.DATASET.WEKA_REPLAY_INCLUDE_ORPHANS is True

    def test_include_ephemeral_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_INCLUDE_EPHEMERAL", True)
        assert Environment.DATASET.WEKA_REPLAY_INCLUDE_EPHEMERAL is True

    def test_include_roles_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            Environment.DATASET, "WEKA_REPLAY_INCLUDE_ROLES", "main,sub-agent"
        )
        assert Environment.DATASET.WEKA_REPLAY_INCLUDE_ROLES == "main,sub-agent"

    def test_sampling_passthrough_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.DATASET, "WEKA_REPLAY_SAMPLING_PASSTHROUGH", "temperature"
        )
        assert Environment.DATASET.WEKA_REPLAY_SAMPLING_PASSTHROUGH == "temperature"


class TestResolveOptionsValidation:
    def test_invalid_session_mode_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_SESSION_MODE", "bogus")
        run = _run_with(tmp_path)
        with pytest.raises(ValueError, match="WEKA_REPLAY_SESSION_MODE"):
            resolve_options(run)

    def test_valid_session_modes_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _run_with(tmp_path)
        for mode in ("tree", "independent"):
            monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_SESSION_MODE", mode)
            assert resolve_options(run).session_mode == mode


class TestIncludeRolesParsing:
    def test_empty_string_yields_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_INCLUDE_ROLES", "")
        run = _run_with(tmp_path)
        assert resolve_options(run).include_roles is None

    def test_populated_string_yields_frozenset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.DATASET, "WEKA_REPLAY_INCLUDE_ROLES", "main, sub-agent"
        )
        run = _run_with(tmp_path)
        assert resolve_options(run).include_roles == frozenset({"main", "sub-agent"})


class TestSamplingPassthroughParsing:
    def test_default_yields_two_element_frozenset(self, tmp_path: Path) -> None:
        run = _run_with(tmp_path)
        assert resolve_options(run).sampling_passthrough == frozenset(
            {"temperature", "top_p"}
        )

    def test_empty_string_yields_empty_frozenset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_SAMPLING_PASSTHROUGH", "")
        run = _run_with(tmp_path)
        assert resolve_options(run).sampling_passthrough == frozenset()


class TestResolveOptionsFromRun:
    def test_reads_synthesis_max_isl_and_max_context_length(
        self, tmp_path: Path
    ) -> None:
        run = _run_with(tmp_path, synthesis_max_isl=4096, max_context_length=128000)
        options = resolve_options(run)
        assert options.synthesis_max_isl == 4096
        assert options.max_context_length == 128000

    def test_defaults_to_none_when_unset(self, tmp_path: Path) -> None:
        run = _run_with(tmp_path)
        options = resolve_options(run)
        assert options.synthesis_max_isl is None
        assert options.max_context_length is None

    def test_full_options_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.DATASET, "WEKA_REPLAY_SESSION_MODE", "independent"
        )
        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_INCLUDE_ORPHANS", True)
        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_INCLUDE_EPHEMERAL", True)
        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_INCLUDE_ROLES", "main")
        monkeypatch.setattr(
            Environment.DATASET, "WEKA_REPLAY_SAMPLING_PASSTHROUGH", "temperature,top_k"
        )
        run = _run_with(tmp_path, synthesis_max_isl=100, max_context_length=200)
        assert resolve_options(run) == WekaReplayOptions(
            session_mode="independent",
            include_orphans=True,
            include_ephemeral=True,
            include_roles=frozenset({"main"}),
            sampling_passthrough=frozenset({"temperature", "top_k"}),
            synthesis_max_isl=100,
            max_context_length=200,
        )
