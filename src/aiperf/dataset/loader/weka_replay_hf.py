# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF-hosted ``replay-v3``/``v4``/``v5`` capture loader.

Downloads one JSONL file from a HuggingFace dataset repo and delegates
reconstruction to :class:`WekaReplayLoader`, so ``--public-dataset
weka_replay_*`` and ``--custom-dataset-type weka_replay --input-file`` run the
exact same code. Which file is fetched is governed by the ``hf_dataset_name`` /
``hf_filename`` / ``hf_revision`` plugin metadata fields; the loader itself is
corpus-agnostic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from huggingface_hub import hf_hub_download

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from aiperf.dataset.loader.weka_replay import WekaReplayLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class WekaReplayHFLoader(BasePublicDatasetLoader):
    """Download a replay capture from HuggingFace, then delegate to ``WekaReplayLoader``.

    Not a ``BaseHFDatasetLoader``: ``datasets`` would parse the multi-GB JSONL
    (header and trailer lines included) into Arrow, and
    ``BasePublicDatasetLoader._load_dataset`` would buffer the whole body in RAM.
    ``hf_hub_download`` streams to a resumable, content-addressed cache instead.
    """

    tag: ClassVar[str] = "WekaReplayHF"

    def __init__(
        self,
        *,
        run: BenchmarkRun | None = None,
        hf_dataset_name: str,
        hf_filename: str,
        hf_revision: str | None = None,
        hf_split: str = "train",
        prompt_generator: PromptGenerator | None = None,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(run=run, **kwargs)
        self.hf_dataset_name = hf_dataset_name
        self.hf_filename = hf_filename
        self.hf_revision = hf_revision
        self.hf_split = hf_split
        self._prompt_generator = prompt_generator
        self._default_block_size = default_block_size

    async def load_dataset(self) -> dict[str, Path]:
        """Download (or reuse from the HF cache) the replay file and return its path."""
        self.info(
            f"Fetching {self.hf_dataset_name}/{self.hf_filename} "
            f"(revision={self.hf_revision or 'main'}) from the HuggingFace cache; "
            "the first run downloads the full file (~8.8 GB for the 40d corpus)"
        )
        try:
            path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=self.hf_dataset_name,
                filename=self.hf_filename,
                repo_type="dataset",
                revision=self.hf_revision,
            )
        except Exception as e:
            raise DatasetLoaderError(
                f"Failed to download {self.hf_dataset_name}/{self.hf_filename} "
                f"from HuggingFace: {e!r}. If the machine is offline, unset "
                "HF_HUB_OFFLINE or pre-populate the cache; point HF_HOME at a "
                "volume with enough free disk space (the file is several GB)."
            ) from e
        self.info(f"Replay file available at {path}")
        return {"path": Path(path)}

    async def convert_to_conversations(
        self, data: dict[str, Path]
    ) -> list[Conversation]:
        """Run the file-based loader off the event loop (the conversion is long and CPU-bound)."""
        inner = WekaReplayLoader(
            filename=data["path"],
            run=self.run,
            prompt_generator=self._prompt_generator,
            default_block_size=self._default_block_size,
        )

        def _convert() -> list[Conversation]:
            return inner.convert_to_conversations(inner.load_dataset())

        return await asyncio.to_thread(_convert)

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return WekaReplayLoader.get_preferred_sampling_strategy()

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode:
        return WekaReplayLoader.get_default_context_mode()
