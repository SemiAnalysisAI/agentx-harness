# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Env-driven options for the ``weka_replay`` loader.

:func:`resolve_options` is the single place that reads the
``AIPERF_DATASET_WEKA_REPLAY_*`` env knobs (declared on ``_DatasetSettings`` in
``aiperf.common.environment``) plus the CLI-resolved dataset fields
(``--synthesis-max-isl`` / ``--synthesis-max-osl`` / ``--max-context-length``)
off a ``BenchmarkRun``, and produces a :class:`WekaReplayOptions` the loader
converts into a ``_weka_replay_tree.TreeFilterOptions`` and reuses directly
for ``Turn.max_tokens`` capping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from aiperf.common.environment import Environment

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_SESSION_MODES: tuple[Literal["tree"], Literal["independent"]] = ("tree", "independent")


@dataclass(frozen=True, slots=True)
class WekaReplayOptions:
    """Resolved weka_replay knobs: env-driven filtering/passthrough settings
    plus the two CLI-resolved dataset fields that bound synthesis/selection.
    """

    session_mode: Literal["tree", "independent"]
    include_orphans: bool
    include_ephemeral: bool
    include_roles: (
        frozenset[str] | None
    )  # parsed from the comma list; None = all allowed
    sampling_passthrough: frozenset[str]
    synthesis_max_isl: int | None  # --synthesis-max-isl; None = no limit
    max_context_length: int | None  # --max-context-length; None = no limit
    synthesis_max_osl: int | None = None  # --synthesis-max-osl; None = no limit


def _parse_comma_list(value: str) -> frozenset[str] | None:
    """Parse a comma-separated env string into a frozenset, or ``None`` for
    an empty string (meaning "no restriction" for an allow-list)."""
    stripped = value.strip()
    if not stripped:
        return None
    return frozenset(item.strip() for item in stripped.split(",") if item.strip())


def resolve_options(run: BenchmarkRun) -> WekaReplayOptions:
    """Resolve :class:`WekaReplayOptions` from env knobs and ``run.cfg``.

    ``synthesis_max_isl`` / ``synthesis_max_osl`` / ``max_context_length`` are
    read off the run's default dataset (``FileDataset.synthesis.max_isl`` /
    ``FileDataset.synthesis.max_osl`` / ``FileDataset.max_context_length``),
    not off raw CLI args, so this reflects whatever config-resolution layer
    (CLI flags or a YAML config file) actually produced. All are
    ``getattr``-guarded since not every dataset type carries them (e.g. a
    synthetic dataset has none).
    """
    settings = Environment.DATASET

    session_mode_raw = settings.WEKA_REPLAY_SESSION_MODE
    if session_mode_raw not in _SESSION_MODES:
        raise ValueError(
            f"AIPERF_DATASET_WEKA_REPLAY_SESSION_MODE must be one of "
            f"{_SESSION_MODES!r}, got {session_mode_raw!r}."
        )
    session_mode: Literal["tree", "independent"] = session_mode_raw  # type: ignore[assignment]

    dataset = run.cfg.get_default_dataset()
    synthesis = getattr(dataset, "synthesis", None)
    synthesis_max_isl = getattr(synthesis, "max_isl", None)
    synthesis_max_osl = getattr(synthesis, "max_osl", None)
    max_context_length = getattr(dataset, "max_context_length", None)

    return WekaReplayOptions(
        session_mode=session_mode,
        include_orphans=settings.WEKA_REPLAY_INCLUDE_ORPHANS,
        include_ephemeral=settings.WEKA_REPLAY_INCLUDE_EPHEMERAL,
        include_roles=_parse_comma_list(settings.WEKA_REPLAY_INCLUDE_ROLES),
        sampling_passthrough=_parse_comma_list(
            settings.WEKA_REPLAY_SAMPLING_PASSTHROUGH
        )
        or frozenset(),
        synthesis_max_isl=synthesis_max_isl,
        max_context_length=max_context_length,
        synthesis_max_osl=synthesis_max_osl,
    )
