# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported timing paths before starting a coordinated run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.plugin.enums import EndpointType, TimingMode
from aiperf.simulation.protocol import validate_endpoint

if TYPE_CHECKING:
    from aiperf.common.models import EndpointInfo
    from aiperf.config.endpoint import EndpointConfig
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.timing.config import CreditPhaseConfig, TimingConfig


def validate_simulator_endpoint(endpoint: EndpointInfo | EndpointConfig) -> None:
    """Require the endpoint whose native generation metadata is supported."""
    if endpoint.type != EndpointType.CHAT or not endpoint.streaming:
        raise NotImplementedError("simulation: streaming chat is required")
    if not endpoint.use_server_token_count:
        raise NotImplementedError("simulation: use_server_token_count must be enabled")
    choices = dict(endpoint.extra).get("n", 1)
    if type(choices) is not int or choices != 1:
        raise NotImplementedError("simulation: only n=1 is supported")


def validate_simulation(
    run: BenchmarkRun, timing: TimingConfig, endpoint: str | None
) -> None:
    """Keep the initial integration limited to completion-driven agentic replay."""
    if not endpoint or run.cfg.endpoint.transport != "sglang_simulator":
        raise ValueError(
            "simulation requires both --transport sglang_simulator and "
            "AIPERF_TIMING_SIMULATOR_COORDINATOR"
        )
    validate_endpoint(endpoint)
    validate_simulator_endpoint(run.cfg.endpoint)
    for name in ("server_metrics", "gpu_telemetry", "network_latency"):
        if getattr(run.cfg, name).enabled:
            raise NotImplementedError(f"simulation: {name} must be disabled")
    if len(timing.urls) != 1:
        raise NotImplementedError("simulation: exactly one server URL is required")
    if run.cfg.endpoint.wait_for_model_timeout > 0:
        raise NotImplementedError(
            "simulation: inference readiness probes are unsupported"
        )
    dataset = run.cfg.get_default_dataset()
    if getattr(dataset, "trace_idle_gap_cap_seconds", None) is not None:
        raise NotImplementedError("simulation: per-trace idle cap is not supported")
    for phase in timing.phase_configs:
        _validate_phase(phase)


def _validate_phase(phase: CreditPhaseConfig) -> None:
    location = f"simulation phase {phase.phase_index!r} ({phase.phase})"
    if phase.timing_mode != TimingMode.AGENTIC_REPLAY:
        raise NotImplementedError(f"{location}: only agentic replay is supported")
    for name in (
        "seamless",
        "prefill_concurrency",
        "concurrency_ramp_duration_sec",
        "prefill_concurrency_ramp_duration_sec",
        "request_rate_ramp_duration_sec",
        "request_rate_series",
        "agentic_cache_warmup_duration_sec",
        "warmup_requests_per_lane",
    ):
        if getattr(phase, name):
            raise NotImplementedError(f"{location}: {name} is not supported")
    if phase.request_cancellation.rate:
        raise NotImplementedError(f"{location}: randomized cancellation is unsupported")
