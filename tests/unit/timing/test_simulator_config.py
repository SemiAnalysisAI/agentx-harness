# SPDX-License-Identifier: Apache-2.0
"""Reject configurations whose workload timing is not coordinated."""

from types import SimpleNamespace
from typing import Any

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import TimingMode
from aiperf.simulation.config import validate_simulation
from aiperf.timing.config import CreditPhaseConfig, TimingConfig


def configuration(**phase_fields: Any) -> tuple[Any, TimingConfig]:
    run = SimpleNamespace(
        cfg=SimpleNamespace(
            endpoint=SimpleNamespace(
                transport="sglang_simulator",
                wait_for_model_timeout=0,
                type="chat",
                streaming=True,
                use_server_token_count=True,
                extra=[],
            ),
            server_metrics=SimpleNamespace(enabled=False),
            gpu_telemetry=SimpleNamespace(enabled=False),
            network_latency=SimpleNamespace(enabled=False),
            get_default_dataset=lambda: SimpleNamespace(
                trace_idle_gap_cap_seconds=None
            ),
        )
    )
    phase = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        **phase_fields,
    )
    return run, TimingConfig(phase_configs=[phase], urls=["http://localhost:30000"])


def test_snapshot_warmup_and_infinite_grace_remain_supported() -> None:
    run, timing = configuration(grace_period_sec=float("inf"))
    warmup = timing.phase_configs[0].model_copy(
        update={"phase": CreditPhase.WARMUP, "phase_index": None}
    )
    timing = timing.model_copy(
        update={"phase_configs": [warmup, *timing.phase_configs]}
    )
    validate_simulation(run, timing, "tcp://127.0.0.1:30001")


@pytest.mark.parametrize(
    "transport,endpoint",
    [
        param("sglang_simulator", None, id="missing-coordinator"),
        param("http", "tcp://127.0.0.1:30001", id="missing-transport"),
    ],
)  # fmt: skip
def test_both_clock_and_transport_are_required(
    transport: str, endpoint: str | None
) -> None:
    run, timing = configuration()
    run.cfg.endpoint.transport = transport
    with pytest.raises(ValueError, match="requires both"):
        validate_simulation(run, timing, endpoint)


@pytest.mark.parametrize(
    "field,value",
    [
        param("prefill_concurrency", 1, id="first-token-admission"),
        param("agentic_cache_warmup_duration_sec", 1.0, id="accelerated-warmup"),
        param("seamless", True, id="overlapping-phases"),
    ],
)  # fmt: skip
def test_uncoordinated_timing_is_rejected(field: str, value: Any) -> None:
    run, timing = configuration(**{field: value})
    with pytest.raises(NotImplementedError, match=field):
        validate_simulation(run, timing, "tcp://127.0.0.1:30001")


def test_readiness_inference_cannot_enter_the_input_barrier() -> None:
    run, timing = configuration()
    run.cfg.endpoint.wait_for_model_timeout = 5
    with pytest.raises(NotImplementedError, match="readiness probes"):
        validate_simulation(run, timing, "tcp://127.0.0.1:30001")


@pytest.mark.parametrize("name", ["server_metrics", "gpu_telemetry", "network_latency"])
def test_physical_measurements_are_rejected(name: str) -> None:
    run, timing = configuration()
    getattr(run.cfg, name).enabled = True
    with pytest.raises(NotImplementedError, match=name):
        validate_simulation(run, timing, "tcp://127.0.0.1:30001")


@pytest.mark.parametrize(
    "field,value,error",
    [
        param("type", "completions", "streaming chat", id="completions"),
        param("streaming", False, "streaming chat", id="nonstreaming"),
        param("use_server_token_count", False, "server_token_count", id="client-tokens"),
        param("extra", [("n", 2)], "n=1", id="multiple-choices"),
    ],
)  # fmt: skip
def test_generation_timing_requires_supported_endpoint(
    field: str, value: Any, error: str
) -> None:
    run, timing = configuration()
    setattr(run.cfg.endpoint, field, value)
    with pytest.raises(NotImplementedError, match=error):
        validate_simulation(run, timing, "tcp://127.0.0.1:30001")
