# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``public_dataset_provenance`` HF file/revision fields."""

from __future__ import annotations

from aiperf.config.config import BenchmarkConfig
from aiperf.dataset.provenance import public_dataset_provenance


def _cfg(dataset: str) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "type": "chat",
            },
            "datasets": [{"name": "main", "type": "public", "dataset": dataset}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 8,
                    "duration": 60,
                }
            ],
        }
    )


def test_public_dataset_provenance_replay_entry_includes_file_and_revision() -> None:
    provenance = public_dataset_provenance(_cfg("weka_replay_40d_102026"))

    assert provenance is not None
    assert provenance["hf_dataset_name"] == "wekaio/replay-40d-oct14-337a03810637"
    assert provenance["hf_filename"] == "replay-40d-oct14-337a03810637.jsonl"
    assert provenance["hf_revision"] == "5afaa89882fc2c24a3dcb8bcdcd0485770c5021d"


def test_public_dataset_provenance_existing_entry_omits_file_and_revision() -> None:
    provenance = public_dataset_provenance(
        _cfg("semianalysis_cc_traces_weka_062126_256k")
    )

    assert provenance is not None
    assert "hf_filename" not in provenance
    assert "hf_revision" not in provenance
