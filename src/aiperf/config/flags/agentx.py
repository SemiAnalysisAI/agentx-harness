# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in standalone AgentX defaults, separate from the frozen MVP recipe."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig

AGENTX_SCENARIO = "inferencex-agentx"
AGENTX_DATASET = "semianalysis_cc_traces_weka_062126"

CLI_DEFAULTS: dict[str, Any] = {
    "endpoint_type": "chat",
    "streaming": True,
    "concurrency": 8,
    "benchmark_duration": 3600,
    "stats_interval": 30,
    "random_seed": 42,
    "failed_request_threshold": 0.10,
    "trajectory_start_min_ratio": 0.25,
    "trajectory_start_max_ratio": 0.75,
    "warmup_requests_per_lane": 10,
    "warmup_grace_period": 1800,
    "trace_idle_gap_cap_seconds": 300,
    "use_server_token_count": True,
    "no_gpu_telemetry": True,
    "conversation_num_dataset_entries": 393,
    "slice_duration": 1.0,
    "public_dataset": AGENTX_DATASET,
}


def apply_cli_defaults(cli: CLIConfig) -> CLIConfig:
    """Fill omitted flags without mutating the caller or hiding explicit conflicts."""
    if cli.scenario != AGENTX_SCENARIO:
        return cli
    defaults = {
        key: value
        for key, value in CLI_DEFAULTS.items()
        if key not in cli.model_fields_set
    }
    if cli.input_file or cli.hf_weka_dataset:
        defaults.pop("public_dataset", None)
    if "gpu_telemetry" in cli.model_fields_set:
        defaults.pop("no_gpu_telemetry", None)
    if cli.agentic_cache_warmup_duration is not None:
        defaults.pop("warmup_requests_per_lane", None)
    return type(cli).model_validate({**defaults, **cli.model_dump(exclude_unset=True)})


def apply_yaml_defaults(data: dict[str, Any], cli: CLIConfig) -> dict[str, Any]:
    """Fill YAML omissions before CLI overrides; preserve authored lists and values."""
    scenario = (
        cli.scenario
        if "scenario" in cli.model_fields_set
        else data.get("benchmark", {}).get("scenario")
    )
    if scenario != AGENTX_SCENARIO:
        return data
    data = copy.deepcopy(data)
    if "random_seed" in cli.model_fields_set:
        data["random_seed"] = cli.random_seed
    else:
        data.setdefault("random_seed", 42)
    benchmark = data.setdefault("benchmark", {})
    benchmark["scenario"] = scenario
    for section, defaults in {
        "endpoint": {"type": "chat", "streaming": True, "use_server_token_count": True},
        "gpu_telemetry": {"enabled": False},
        "artifacts": {"slice_duration": 1.0},
        "runtime": {"stats_interval": 30},
    }.items():
        target = benchmark.setdefault(section, {})
        if isinstance(target, dict):
            for key, value in defaults.items():
                target.setdefault(key, value)
    if "datasets" not in benchmark:
        from aiperf.config.flags._converter_dataset import build_dataset

        dataset_cli = apply_cli_defaults(
            cli.model_copy(
                update={"scenario": scenario, "random_seed": data["random_seed"]}
            )
        )
        benchmark["datasets"] = [{"name": "main", **build_dataset(dataset_cli)}]
    benchmark.setdefault(
        "phases", [{"name": "profiling", "kind": "profiling", "type": "concurrency"}]
    )
    _apply_yaml_phase_defaults(benchmark["phases"], cli)
    return data


def _apply_yaml_phase_defaults(phases: list[Any], cli: CLIConfig) -> None:
    """Default profiling phases without changing authored warmup phases."""
    for phase in phases:
        if not isinstance(phase, dict) or phase.get("kind", "profiling") != "profiling":
            continue
        for key, value in {
            "concurrency": 8,
            "duration": 3600,
            "grace_period": cli.benchmark_grace_period,
            "failed_request_threshold": 0.10,
            "trajectory_start_min_ratio": 0.25,
            "trajectory_start_max_ratio": 0.75,
            "warmup_requests_per_lane": 10,
            "agentic_warmup_grace_period": 1800,
        }.items():
            if key == "warmup_requests_per_lane" and (
                phase.get("agentic_cache_warmup_duration") is not None
                or cli.agentic_cache_warmup_duration is not None
            ):
                continue
            phase.setdefault(key, value)
