# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discovery uses a real loopback HTTP server, without inference or HF downloads."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web
from pytest import param

from aiperf.cli_commands._agentx import discover_model, models_url


@pytest.fixture
async def model_server() -> AsyncIterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] = {
        "payload": {"data": [{"id": "real/model"}]},
        "status": 200,
        "requests": [],
    }

    async def models(request: web.Request) -> web.Response:
        state["requests"].append(request)
        if "body" in state:
            return web.Response(text=state["body"], status=state["status"])
        return web.json_response(state["payload"], status=state["status"])

    app = web.Application()
    app.router.add_get("/proxy/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/proxy", state
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_discovery_propagates_auth_and_headers(model_server: tuple) -> None:
    url, state = model_server
    assert (
        await discover_model(
            {
                "urls": [url],
                "api_key": "test-secret",
                "headers": {"X-Tenant": "test", "authorization": "old-secret"},
            }
        )
        == "real/model"
    )
    request = state["requests"][0]
    assert request.headers.getall("Authorization") == ["Bearer test-secret"]
    assert request.headers["X-Tenant"] == "test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        param({"data": []}, id="empty"),
        param({"data": [{"id": "one"}, {"id": "two"}]}, id="multiple"),
        param({"data": [{"id": ""}]}, id="empty-id"),
        param({"data": [{"id": 42}]}, id="non-string-id"),
        param({"data": ["model"]}, id="bad-entry"),
        param({"data": {}}, id="bad-data"),
        param([], id="bad-payload"),
    ],
)  # fmt: skip
async def test_discovery_rejects_ambiguous_payload(
    model_server: tuple, payload: Any
) -> None:
    url, state = model_server
    state["payload"] = payload
    with pytest.raises(ValueError, match="pass --model explicitly"):
        await discover_model({"urls": [url]})


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 401, 403, 404, 500])
async def test_discovery_http_errors_are_actionable(
    model_server: tuple, status: int
) -> None:
    url, state = model_server
    state["status"] = status
    state["body"] = "sensitive response body"
    with pytest.raises(ValueError, match=f"HTTP {status}") as exc:
        await discover_model({"urls": [url], "api_key": "test-secret"})
    assert "test-secret" not in str(exc.value)
    assert "sensitive" not in str(exc.value)
    assert len(state["requests"]) == 1


@pytest.mark.asyncio
async def test_discovery_invalid_json(model_server: tuple) -> None:
    url, state = model_server
    state["body"] = "<html>not JSON</html>"
    with pytest.raises(ValueError, match="could not read"):
        await discover_model({"urls": [url]})


@pytest.mark.asyncio
async def test_discovery_bounds_response_size(model_server: tuple) -> None:
    url, state = model_server
    state["body"] = "x" * (1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        await discover_model({"urls": [url]})


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, aiohttp.ClientConnectionError])
async def test_discovery_transport_error_redacted(error: type[Exception]) -> None:
    with (
        patch("aiohttp.ClientSession.get", side_effect=error("secret-value")),
        pytest.raises(ValueError, match="pass --model explicitly") as exc,
    ):
        await discover_model({"urls": ["http://localhost:9000"]})
    assert "secret-value" not in str(exc.value)


@pytest.mark.asyncio
async def test_discovery_multiple_urls_requires_model() -> None:
    with pytest.raises(ValueError, match="one --url"):
        await discover_model({"urls": ["http://a", "http://b"]})


@pytest.mark.parametrize(
    "url, expected",
    [
        param("localhost:8000", "http://localhost:8000/v1/models", id="bare"),
        param("http://host/", "http://host/v1/models", id="root"),
        param("http://host/v1/", "http://host/v1/models", id="v1"),
        param("https://host/proxy/v1/chat/completions", "https://host/proxy/v1/models", id="full"),
        param("http://[::1]:8000", "http://[::1]:8000/v1/models", id="ipv6"),
    ],
)  # fmt: skip
def test_models_url(url: str, expected: str) -> None:
    assert models_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host",
        "http://user:secret@host",
        "https://host?key=secret",
        "https://host#fragment",
    ],
)
def test_models_url_rejects_unsafe_or_unsupported_urls(url: str) -> None:
    with pytest.raises(ValueError) as exc:
        models_url(url)
    assert "secret" not in str(exc.value)
