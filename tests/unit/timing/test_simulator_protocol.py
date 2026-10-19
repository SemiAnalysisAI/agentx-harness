# SPDX-License-Identifier: Apache-2.0
"""Exercise the simulator control contract over an actual loopback socket."""

import asyncio
from collections.abc import Iterator
from typing import Any

import orjson
import pytest
import zmq
import zmq.asyncio
from pytest import param

from aiperf.common.constants import IS_WINDOWS
from aiperf.simulation.protocol import (
    CoordinatorClient,
    Ready,
    SimulationError,
    Submit,
    validate_endpoint,
)


@pytest.fixture(scope="module")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if IS_WINDOWS:
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def connection() -> Iterator[tuple[CoordinatorClient, zmq.asyncio.Socket]]:
    context = zmq.asyncio.Context()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    server.bind("tcp://127.0.0.1:*")
    endpoint = server.getsockopt_string(zmq.LAST_ENDPOINT)
    client = CoordinatorClient(endpoint)
    try:
        yield client, server
    finally:
        client.close()
        server.close(linger=0)
        context.term()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override,error",
    [
        param({}, None, id="valid"),
        param({"run_id": "other"}, "does not match", id="run"),
        param({"round": 1}, "does not match", id="round"),
        param({"protocol_version": 2}, "does not match", id="version"),
        param({"now_ns": -1}, "does not match", id="time"),
        param({"status": "ERROR", "error": "payload missing"}, "payload missing", id="error"),
        param({"unknown": True}, "invalid simulator Grant", id="unknown-field"),
        param(
            {"terminals": [{"rid": "req", "kind": "COMPLETED", "at_ns": 9}]},
            "timestamp differs",
            id="terminal-time",
        ),
    ],
)  # fmt: skip
async def test_control_validates_round_and_time_before_releasing_result(
    connection: tuple[CoordinatorClient, zmq.asyncio.Socket],
    override: dict[str, Any],
    error: str | None,
) -> None:
    client, server = connection

    async def respond() -> None:
        ready = orjson.loads(await server.recv())
        assert ready["operations"] == [{"type": "submit", "rid": "req"}]
        assert ready["protocol_version"] == 1
        await server.send(
            orjson.dumps(
                {
                    "run_id": "run",
                    "round": 0,
                    "now_ns": 10,
                    "terminals": [],
                    **override,
                }
            )
        )

    response = asyncio.create_task(respond())
    try:
        if error is not None:
            with pytest.raises(SimulationError, match=error):
                await client.request(Ready("run", 0, [Submit("req")]))
        else:
            grant = await client.request(Ready("run", 0, [Submit("req")]))
            assert grant.now_ns == 10
        await response
    finally:
        if not response.done():
            response.cancel()
        await asyncio.gather(response, return_exceptions=True)


@pytest.mark.parametrize(
    "endpoint",
    [
        param("tcp://0.0.0.0:1234", id="wildcard"),
        param("tcp://127.0.0.1:0", id="zero-port"),
        param("tcp://127.0.0.1:1234/path", id="path"),
        param("tcp://localhost:1234", id="implicit-address"),
        param("http://127.0.0.1:1234", id="http"),
    ],
)  # fmt: skip
def test_control_requires_explicit_loopback_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        validate_endpoint(endpoint)
