# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin-registration / ``can_load`` collision-safety tests for ``weka_replay``.

Mirrors ``test_can_load.py``'s pattern of iterating every registered custom
loader and asserting only the loader under test claims its own header shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader._weka_replay_reader import WekaReplayLoadError
from aiperf.dataset.loader.weka_replay import WekaReplayLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy, PluginType
from tests.unit.conftest import make_run_from_cli
from tests.unit.dataset.loader._weka_replay_builders import (
    FakePromptGenerator,
    build_header,
    build_instance,
    build_request,
    build_session,
    write_lines,
)


def _all_loader_classes() -> list[type]:
    return [
        LoaderClass
        for _entry, LoaderClass in plugins.iter_all(PluginType.CUSTOM_DATASET_LOADER)
    ]


def test_weka_replay_registered_as_custom_dataset_loader():
    assert plugins.has_entry(
        PluginType.CUSTOM_DATASET_LOADER, CustomDatasetType.WEKA_REPLAY
    )
    LoaderClass = plugins.get_class(
        PluginType.CUSTOM_DATASET_LOADER, CustomDatasetType.WEKA_REPLAY
    )
    assert LoaderClass is WekaReplayLoader


def test_weka_replay_custom_dataset_type_enum_value():
    assert CustomDatasetType.WEKA_REPLAY.value == "weka_replay"


def test_weka_replay_preferred_sampling_and_context_mode():
    assert (
        WekaReplayLoader.get_preferred_sampling_strategy()
        == DatasetSamplingStrategy.SEQUENTIAL
    )
    assert (
        WekaReplayLoader.get_default_context_mode()
        == ConversationContextMode.DELTAS_WITH_RESPONSES
    )


@pytest.mark.parametrize(
    "schema_version",
    [
        param("replay-v3", id="v3"),
        param("replay-v4", id="v4"),
        param("replay-v5", id="v5"),
    ],
)  # fmt: skip
def test_only_weka_replay_claims_a_replay_header(schema_version):
    data = build_header(schema_version)
    matched = [
        LoaderClass
        for LoaderClass in _all_loader_classes()
        if LoaderClass.can_load(data=data) is True
    ]
    assert matched == [WekaReplayLoader], (
        f"expected only WekaReplayLoader to claim a {schema_version!r} header, "
        f"got {[c.__name__ for c in matched]}"
    )


@pytest.mark.parametrize(
    "data,expected",
    [
        param(
            {"_schema": "replay-v3"}, True, id="v3_header_dict"
        ),
        param(
            {"_schema": "replay-v4"}, True, id="v4_header_dict"
        ),
        param(
            {"_schema": "replay-v5"}, True, id="v5_header_dict"
        ),
        param(
            {"_schema": "replay-v9"}, True, id="unsupported_version_still_broadly_matched"
        ),
        param(
            {"_schema": "replay-index-v1", "sessions": []},
            False,
            id="v5_index_trailer_rejected",
        ),
        param(
            {"session_id": "s1", "start_ts": "2024-01-01T00:00:00Z"},
            False,
            id="session_line_rejected",
        ),
        param(None, False, id="none_data"),
        param({"foo": "bar"}, False, id="garbage_dict"),
        param("not a dict", False, id="non_dict_garbage"),
    ],
)  # fmt: skip
def test_can_load_data_probe(data, expected):
    assert WekaReplayLoader.can_load(data=data) is expected


def test_can_load_from_filename_v3(tmp_path: Path):
    path = write_lines(tmp_path, [build_header("replay-v3")])
    assert WekaReplayLoader.can_load(data=None, filename=path) is True


def test_can_load_from_filename_rejects_directory(tmp_path: Path):
    assert WekaReplayLoader.can_load(data=None, filename=tmp_path) is False


def test_can_load_from_filename_missing_file_returns_false(tmp_path: Path):
    assert (
        WekaReplayLoader.can_load(data=None, filename=tmp_path / "nope.jsonl") is False
    )


def test_can_load_none_and_no_filename_returns_false():
    assert WekaReplayLoader.can_load(data=None, filename=None) is False


def _build_loader(tmp_path: Path, lines: list[dict]) -> WekaReplayLoader:
    path = write_lines(tmp_path, lines)
    cli = CLIConfig(
        model_names=["test-model"],
        input_file=str(path),
        custom_dataset_type=CustomDatasetType.WEKA_REPLAY,
    )
    run = make_run_from_cli(cli)
    return WekaReplayLoader(
        filename=path, run=run, prompt_generator=FakePromptGenerator()
    )


def test_unsupported_schema_version_claimed_by_can_load_but_rejected_by_load_dataset(
    tmp_path: Path,
):
    header = build_header("replay-v9")
    session = build_session(instances=[build_instance(requests=[build_request()])])
    assert WekaReplayLoader.can_load(data=header) is True

    loader = _build_loader(tmp_path, [header, session])
    with pytest.raises(WekaReplayLoadError, match="replay-v9"):
        loader.load_dataset()
