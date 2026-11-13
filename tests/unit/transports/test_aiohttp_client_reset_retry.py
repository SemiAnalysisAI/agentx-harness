# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retry of a POST whose reused keep-alive connection is reset before any response."""

import asyncio
import errno
import socket
import struct
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import aiohttp
import pytest
from pytest import param

from aiperf.common.models import AioHttpTraceData, RequestRecord
from aiperf.transports.aiohttp_client import (
    AioHttpClient,
    _is_reset_reused_connection,
    create_tcp_connector,
)

HEADERS = {"Content-Type": "application/json"}


def _write_failure() -> aiohttp.ClientOSError:
    exc = aiohttp.ClientOSError(None, "Can not write request body for http://x/")
    exc.__cause__ = aiohttp.ClientConnectionResetError(
        "Cannot write to closing transport"
    )
    return exc


def _record(
    *, reused: bool = True, headers_received: bool = False, status: int | None = None
) -> RequestRecord:
    trace = AioHttpTraceData(
        connection_reused_perf_ns=1 if reused else None,
        response_headers_received_perf_ns=2 if headers_received else None,
    )
    return RequestRecord(trace_data=trace, status=status)


class TestIsResetReusedConnection:
    @pytest.mark.parametrize(
        "exc",
        [
            param(_write_failure(), id="body-write-failure"),
            param(aiohttp.ClientOSError(errno.ECONNRESET, "Connection reset by peer"), id="econnreset"),
            param(aiohttp.ClientOSError(errno.EPIPE, "Broken pipe"), id="epipe"),
        ],
    )  # fmt: skip
    def test_reset_on_reused_connection_before_response_is_retryable(
        self, exc: BaseException
    ) -> None:
        assert _is_reset_reused_connection(exc, "POST", b"{}", _record())

    @pytest.mark.parametrize(
        "exc,method,data,record",
        [
            param(aiohttp.ServerDisconnectedError(), "POST", b"{}", _record(), id="clean-close"),
            param(TimeoutError(), "POST", b"{}", _record(), id="timeout"),
            param(aiohttp.ClientOSError(errno.ECONNREFUSED, "refused"), "POST", b"{}", _record(), id="other-errno"),
            param(_write_failure(), "POST", b"{}", _record(reused=False), id="fresh-connection"),
            param(_write_failure(), "POST", b"{}", _record(headers_received=True), id="response-started"),
            param(_write_failure(), "POST", b"{}", _record(status=200), id="status-set"),
            param(_write_failure(), "GET", None, _record(), id="get"),
            param(_write_failure(), "POST", aiohttp.FormData(), _record(), id="multipart"),
        ],
    )  # fmt: skip
    def test_other_failures_are_not_retryable(
        self, exc: BaseException, method: str, data: object, record: RequestRecord
    ) -> None:
        assert not _is_reset_reused_connection(exc, method, data, record)


@dataclass
class _ServerStats:
    connections: int = 0
    served: int = 0


async def _read_request(reader: asyncio.StreamReader) -> None:
    head = await reader.readuntil(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
    await reader.readexactly(length)


async def _respond(writer: asyncio.StreamWriter) -> None:
    body = b'{"ok":true}'
    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
    )
    await writer.drain()


def _reset(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    writer.transport.abort()


Handler = Callable[
    [asyncio.StreamReader, asyncio.StreamWriter, _ServerStats], Awaitable[None]
]


async def _serve_then_reset_next(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, stats: _ServerStats
) -> None:
    """Serve one request, then reset the connection when the next one starts arriving."""
    await _read_request(reader)
    stats.served += 1
    await _respond(writer)
    await reader.read(1)
    _reset(writer)


async def _reset_before_reading(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, stats: _ServerStats
) -> None:
    await reader.read(1)
    _reset(writer)


async def _serve_then_read_next_and_close(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, stats: _ServerStats
) -> None:
    """Serve one request, then read the whole next request and close without responding."""
    await _read_request(reader)
    stats.served += 1
    await _respond(writer)
    await _read_request(reader)
    stats.served += 1
    writer.close()


@pytest.fixture
async def http_server(
    request: pytest.FixtureRequest,
) -> AsyncIterator[tuple[str, _ServerStats]]:
    handler: Handler = request.param
    stats = _ServerStats()

    async def on_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        stats.connections += 1
        try:
            if stats.connections == 1:
                await handler(reader, writer, stats)
            else:
                while True:
                    await _read_request(reader)
                    stats.served += 1
                    await _respond(writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(on_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions", stats
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def client() -> AsyncIterator[AioHttpClient]:
    client = AioHttpClient(timeout=30)
    try:
        yield client
    finally:
        await client.close()


class TestResetRetryOverSockets:
    @pytest.mark.parametrize("http_server", [_serve_then_reset_next], indirect=True)
    @pytest.mark.parametrize(
        "body_size",
        [
            param(2_000, id="small-body"),
            param(2_500_000, id="large-body"),
        ],
    )  # fmt: skip
    async def test_post_reset_on_reused_connection_retries_on_new_connection(
        self,
        http_server: tuple[str, _ServerStats],
        client: AioHttpClient,
        body_size: int,
    ) -> None:
        url, stats = http_server
        first = await client.post_request(url, b"{}", HEADERS)
        second = await client.post_request(url, b"x" * body_size, HEADERS)

        assert first.status == 200
        assert second.error is None
        assert second.status == 200
        assert second.trace_data.connection_reused_perf_ns is None
        assert second.trace_data.tcp_connect_start_perf_ns is not None
        assert stats.connections == 2
        assert stats.served == 2

    @pytest.mark.parametrize("http_server", [_serve_then_reset_next], indirect=True)
    async def test_post_reset_on_caller_connector_retries_on_that_connector(
        self, http_server: tuple[str, _ServerStats], client: AioHttpClient
    ) -> None:
        """A connector the caller keeps (a sticky-session lease) carries the retry,
        so the following request reuses the retry's connection."""
        url, stats = http_server
        lease = create_tcp_connector(limit=1)

        async def post(body: bytes) -> RequestRecord:
            return await asyncio.wait_for(
                client.post_request(
                    url, body, HEADERS, connector=lease, connector_owner=False
                ),
                timeout=5,
            )

        try:
            first = await post(b"{}")
            retried = await post(b"x" * 2_000)
            third = await post(b"{}")
            assert not lease.closed
        finally:
            await lease.close()

        assert first.status == 200
        assert retried.error is None
        assert retried.status == 200
        assert retried.trace_data.connection_reused_perf_ns is None
        assert third.status == 200
        assert third.trace_data.connection_reused_perf_ns is not None
        assert stats.connections == 2
        assert stats.served == 3

    @pytest.mark.parametrize("http_server", [_reset_before_reading], indirect=True)
    async def test_post_reset_on_new_connection_is_not_retried(
        self, http_server: tuple[str, _ServerStats], client: AioHttpClient
    ) -> None:
        url, stats = http_server
        record = await client.post_request(url, b"{}", HEADERS)

        assert record.error is not None
        assert stats.connections == 1
        assert stats.served == 0

    @pytest.mark.parametrize(
        "http_server", [_serve_then_read_next_and_close], indirect=True
    )
    async def test_post_closed_after_server_read_it_is_not_retried(
        self, http_server: tuple[str, _ServerStats], client: AioHttpClient
    ) -> None:
        url, stats = http_server
        await client.post_request(url, b"{}", HEADERS)
        record = await client.post_request(url, b"{}", HEADERS)

        assert record.error is not None
        assert record.error.type == "ServerDisconnectedError"
        assert stats.connections == 1
        assert stats.served == 2
