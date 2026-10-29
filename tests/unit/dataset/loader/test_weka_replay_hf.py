# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``WekaReplayHFLoader``: HF download plus delegation to ``WekaReplayLoader``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.weka_replay import WekaReplayLoader
from aiperf.dataset.loader.weka_replay_hf import WekaReplayHFLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    CustomDatasetType,
    DatasetSamplingStrategy,
    PluginType,
    PublicDatasetType,
)
from tests.unit.conftest import make_run_from_cli
from tests.unit.dataset.loader._weka_replay_builders import (
    FakePromptGenerator,
    build_header,
    build_instance,
    build_request,
    build_session,
    write_lines,
)

_REPO = "wekaio/replay-40d-oct14-337a03810637"
_FILENAME = "replay-40d-oct14-337a03810637.jsonl"
_REVISION = "5afaa89882fc2c24a3dcb8bcdcd0485770c5021d"
_ALIAS = "weka_replay_40d_102026"


@pytest.fixture
def replay_file(tmp_path: Path) -> Path:
    session = build_session(
        session_id="sessA",
        instances=[
            build_instance(
                instance_id="sessA:main",
                requests=[
                    build_request(request_id=1, ts="2024-01-01T00:00:01Z"),
                    build_request(request_id=2, ts="2024-01-01T00:00:05Z"),
                ],
            )
        ],
    )
    return write_lines(tmp_path, [build_header("replay-v5"), session])


@pytest.fixture
def run(replay_file: Path):
    return make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(replay_file),
            custom_dataset_type=CustomDatasetType.WEKA_REPLAY,
        )
    )


def _make_loader(run, **overrides) -> WekaReplayHFLoader:
    kwargs = {
        "run": run,
        "hf_dataset_name": _REPO,
        "hf_filename": _FILENAME,
        "hf_revision": _REVISION,
        "hf_split": "train",
        "prompt_generator": FakePromptGenerator(),
        "default_block_size": 64,
    }
    kwargs.update(overrides)
    return WekaReplayHFLoader(**kwargs)


@pytest.mark.asyncio
class TestLoadDataset:
    async def test_load_dataset_calls_hf_hub_download_with_pinned_args(
        self, run, replay_file: Path
    ) -> None:
        loader = _make_loader(run)
        with patch(
            "aiperf.dataset.loader.weka_replay_hf.hf_hub_download",
            return_value=str(replay_file),
        ) as mock_dl:
            data = await loader.load_dataset()

        mock_dl.assert_called_once_with(
            repo_id=_REPO,
            filename=_FILENAME,
            repo_type="dataset",
            revision=_REVISION,
        )
        assert data == {"path": replay_file}

    async def test_load_dataset_without_revision_passes_none(
        self, run, replay_file: Path
    ) -> None:
        loader = _make_loader(run, hf_revision=None)
        with patch(
            "aiperf.dataset.loader.weka_replay_hf.hf_hub_download",
            return_value=str(replay_file),
        ) as mock_dl:
            await loader.load_dataset()

        assert mock_dl.call_args.kwargs["revision"] is None

    @pytest.mark.parametrize(
        "exc",
        [
            param(OSError("offline"), id="oserror"),
            param(RuntimeError("boom"), id="runtime"),
        ],
    )  # fmt: skip
    async def test_load_dataset_download_failure_raises_dataset_loader_error(
        self, run, exc: Exception
    ) -> None:
        loader = _make_loader(run)
        with (
            patch(
                "aiperf.dataset.loader.weka_replay_hf.hf_hub_download",
                side_effect=exc,
            ),
            pytest.raises(DatasetLoaderError) as ei,
        ):
            await loader.load_dataset()

        message = str(ei.value)
        assert _REPO in message
        assert "HF_HUB_OFFLINE" in message
        assert "HF_HOME" in message
        assert ei.value.__cause__ is exc


@pytest.mark.asyncio
class TestDelegation:
    async def test_conversations_match_direct_file_loader(
        self, run, replay_file: Path
    ) -> None:
        loader = _make_loader(run)
        with patch(
            "aiperf.dataset.loader.weka_replay_hf.hf_hub_download",
            return_value=str(replay_file),
        ):
            data = await loader.load_dataset()
            via_hf = await loader.convert_to_conversations(data)

        direct = WekaReplayLoader(
            filename=replay_file,
            run=run,
            prompt_generator=FakePromptGenerator(),
            default_block_size=64,
        )
        expected = direct.convert_to_conversations(direct.load_dataset())

        assert len(via_hf) == len(expected) > 0
        assert [c.session_id for c in via_hf] == [c.session_id for c in expected]
        assert [len(c.turns) for c in via_hf] == [len(c.turns) for c in expected]

    async def test_convert_forwards_block_size_and_prompt_generator(
        self, run, replay_file: Path
    ) -> None:
        pg = FakePromptGenerator()
        loader = _make_loader(run, prompt_generator=pg, default_block_size=32)
        with patch("aiperf.dataset.loader.weka_replay_hf.WekaReplayLoader") as mock_cls:
            mock_cls.return_value.convert_to_conversations.return_value = []
            await loader.convert_to_conversations({"path": replay_file})

        mock_cls.assert_called_once_with(
            filename=replay_file,
            run=run,
            prompt_generator=pg,
            default_block_size=32,
        )

    async def test_tokenizer_kwarg_is_accepted_and_not_forwarded(
        self, run, replay_file: Path
    ) -> None:
        loader = _make_loader(run, tokenizer=object())
        with patch("aiperf.dataset.loader.weka_replay_hf.WekaReplayLoader") as mock_cls:
            mock_cls.return_value.convert_to_conversations.return_value = []
            await loader.convert_to_conversations({"path": replay_file})

        assert "tokenizer" not in mock_cls.call_args.kwargs


class TestClassLevelDelegation:
    def test_sampling_strategy_matches_weka_replay_loader(self) -> None:
        assert (
            WekaReplayHFLoader.get_preferred_sampling_strategy()
            == WekaReplayLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_default_context_mode_matches_weka_replay_loader(self) -> None:
        assert (
            WekaReplayHFLoader.get_default_context_mode()
            == WekaReplayLoader.get_default_context_mode()
            == ConversationContextMode.DELTAS_WITH_RESPONSES
        )


class TestPluginRegistration:
    def test_alias_resolves_to_hf_loader_class(self) -> None:
        cls = plugins.get_class(
            PluginType.PUBLIC_DATASET_LOADER, PublicDatasetType(_ALIAS)
        )
        assert cls is WekaReplayHFLoader

    def test_alias_metadata_pins_repo_file_and_revision(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(_ALIAS)
        assert meta.hf_dataset_name == _REPO
        assert meta.hf_filename == _FILENAME
        assert meta.hf_revision == _REVISION
        assert meta.is_trace is True
        assert meta.default_block_size == 64
        assert meta.default_prompt_corpus == "coding"

    def test_existing_entries_have_no_filename_or_revision(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(
            "semianalysis_cc_traces_weka_062126_256k"
        )
        assert meta.hf_filename is None
        assert meta.hf_revision is None
