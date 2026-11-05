# SPDX-License-Identifier: Apache-2.0
"""Version-one Ready/Grant wire contract for an external simulator."""

from __future__ import annotations

import ipaddress
from typing import Literal
from urllib.parse import urlsplit

import msgspec
import orjson
import zmq
import zmq.asyncio


class SimulationError(RuntimeError):
    """A coordinated run cannot preserve its timing contract."""


class Submit(msgspec.Struct, tag="submit", frozen=True):
    rid: str


class Cancel(msgspec.Struct, tag="cancel", frozen=True):
    rid: str
    reason: str = "phase cancelled"


class Ready(msgspec.Struct, frozen=True):
    run_id: str
    round: int
    operations: list[Submit | Cancel]
    next_wakeup_ns: int | None = None
    finished: bool = False
    protocol_version: int = 1


class Terminal(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    rid: str
    kind: Literal["COMPLETED", "CANCELLED", "ERROR"]
    at_ns: int
    reason: str | None = None


class Grant(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    run_id: str
    round: int
    now_ns: int
    terminals: list[Terminal]
    status: Literal["CONTINUE", "FINISHED", "ERROR"] = "CONTINUE"
    error: str | None = None
    protocol_version: int = 1


def validate_endpoint(endpoint: str) -> str:
    """Keep the first driver limited to a local coordinator."""
    try:
        parsed = urlsplit(endpoint)
        valid = (
            parsed.scheme == "tcp"
            and ipaddress.ip_address(parsed.hostname or "").is_loopback
            and parsed.port is not None
            and 0 < parsed.port < 65536
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("simulator coordinator must be tcp://<loopback IP>:<port>")
    return endpoint


class CoordinatorClient:
    """A single sequential control connection; ambiguous exchanges are not retried."""

    def __init__(self, endpoint: str) -> None:
        endpoint = validate_endpoint(endpoint)
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IPV6, 1)
        self._socket.connect(endpoint)
        self._now_ns = 0

    async def request(self, ready: Ready) -> Grant:
        """Exchange one round; the driver owns liveness and failure monitoring."""
        await self._socket.send(orjson.dumps(msgspec.to_builtins(ready)))
        try:
            grant = msgspec.convert(
                orjson.loads(await self._socket.recv()), type=Grant, strict=True
            )
        except (orjson.JSONDecodeError, msgspec.ValidationError) as exc:
            raise SimulationError(f"invalid simulator Grant: {exc}") from exc
        if (
            grant.protocol_version != 1
            or grant.run_id != ready.run_id
            or grant.round != ready.round
            or grant.now_ns < self._now_ns
        ):
            raise SimulationError("simulator Grant does not match Ready")
        if grant.status == "ERROR":
            raise SimulationError(grant.error or "simulator reported a failure")
        if any(event.at_ns != grant.now_ns for event in grant.terminals):
            raise SimulationError("terminal timestamp differs from its Grant")
        self._now_ns = grant.now_ns
        return grant

    def close(self) -> None:
        """Close without attempting to resume a partially acknowledged round."""
        self._socket.close(linger=0)
        self._context.term()
