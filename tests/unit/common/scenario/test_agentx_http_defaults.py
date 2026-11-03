# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.environment import Environment
from aiperf.common.scenario import get_scenario


def test_agentx_keepalive_outlives_warmup_pause_and_idle_cap() -> None:
    """Pooled connections must survive the 900 s warmup cache-pressure pause and
    the 300 s profiling idle cap, or every lane reconnects at once and the closed
    sockets' TIME_WAIT exhausts the ephemeral ports to a single frontend."""
    http_defaults = get_scenario("agentx").environment_defaults["HTTP"]
    assert http_defaults["KEEPALIVE_TIMEOUT"] == 3600
    assert http_defaults["KEEPALIVE_TIMEOUT"] > 900
    assert http_defaults["KEEPALIVE_TIMEOUT"] > 300


def test_agentx_keepalive_default_applies_to_environment() -> None:
    defaults = get_scenario("agentx").environment_defaults
    with Environment.defaults(defaults):
        assert Environment.HTTP.KEEPALIVE_TIMEOUT == 3600
    assert Environment.HTTP.KEEPALIVE_TIMEOUT == 300
