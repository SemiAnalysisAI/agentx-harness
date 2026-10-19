# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from aiohttp import web
from pytest import param

from aiperf.common.enums import ConnectionReuseStrategy, CreditPhase, RequestContentType
from aiperf.common.models import (
    ErrorDetails,
    ModelEndpointInfo,
    RequestInfo,
    RequestRecord,
    SSEMessage,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TransportType
from aiperf.simulation.identity import request_id
from aiperf.transports.aiohttp_transport import AioHttpTransport
from aiperf.transports.sglang_simulator_transport import SGLangSimulatorTransport
from aiperf.workers.inference_client import InferenceClient, detect_transport_from_url
from tests.unit.transports.conftest import create_model_endpoint_info


@pytest.fixture
def model_endpoint() -> ModelEndpointInfo:
    result = create_model_endpoint_info(streaming=True)
    result.endpoint.use_server_token_count = True
    return result


def response_body(rid: str) -> str:
    timing = {
        "version": 1,
        "rid": rid,
        "arrival_ns": 0,
        "first_token_ns": 10,
        "last_token_ns": 10,
        "commit_ns": 10,
        "terminal_ns": 10,
        "prompt_tokens": 0,
        "completion_tokens": 1,
        "cached_tokens": 0,
    }
    usage = {
        "choices": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
    return (
        f"event: sglang_simulator\ndata: {orjson.dumps(timing).decode()}\n\n"
        'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        f"data: {orjson.dumps(usage).decode()}\n\ndata: [DONE]\n\n"
    )


def payload() -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.fixture
def request_info(model_endpoint: ModelEndpointInfo) -> RequestInfo:
    return RequestInfo(
        model_endpoint=model_endpoint,
        credit_num=3,
        credit_phase=CreditPhase.PROFILING,
        phase_index=2,
        conversation_id="conversation-1",
        turn_index=1,
        x_request_id="request-1",
        x_correlation_id="correlation-1",
    )


@pytest.fixture
async def transport(
    model_endpoint: ModelEndpointInfo,
) -> AsyncIterator[SGLangSimulatorTransport]:
    transport = SGLangSimulatorTransport(
        model_endpoint=model_endpoint,
        benchmark_id="benchmark-1",
        tcp_kwargs={"limit": 1, "limit_per_host": 1},
    )
    await transport.initialize()
    try:
        yield transport
    finally:
        await transport.stop()


def test_plugin_is_explicit_and_http_detection_is_unchanged() -> None:
    assert (
        plugins.get_class(PluginType.TRANSPORT, TransportType.SGLANG_SIMULATOR)
        is SGLangSimulatorTransport
    )
    assert SGLangSimulatorTransport.metadata().url_schemes == []
    assert detect_transport_from_url("http://localhost:8000") == TransportType.HTTP
    assert detect_transport_from_url("https://localhost:8000") == TransportType.HTTP


@pytest.mark.asyncio
async def test_pool_limit_override_is_local_to_simulator(
    transport: SGLangSimulatorTransport, model_endpoint: ModelEndpointInfo
) -> None:
    assert transport.aiohttp_client.tcp_connector.limit == 0
    assert transport.aiohttp_client.tcp_connector.limit_per_host == 0
    tcp_kwargs = {"limit": 1, "limit_per_host": 1}
    ordinary = AioHttpTransport(model_endpoint=model_endpoint, tcp_kwargs=tcp_kwargs)
    assert ordinary.tcp_kwargs == tcp_kwargs


@pytest.mark.parametrize(
    "benchmark_id",
    [param(None, id="absent"), param("", id="empty")],
)  # fmt: skip
def test_missing_benchmark_identity_fails(
    model_endpoint: ModelEndpointInfo, benchmark_id: str | None
) -> None:
    with pytest.raises(ValueError, match="benchmark_id"):
        SGLangSimulatorTransport(
            model_endpoint=model_endpoint, benchmark_id=benchmark_id
        )


@pytest.mark.parametrize(
    "strategy",
    [
        param(ConnectionReuseStrategy.NEVER, id="never"),
        param(ConnectionReuseStrategy.STICKY_USER_SESSIONS, id="sticky"),
    ],
)  # fmt: skip
def test_nonpooled_connections_fail(
    model_endpoint: ModelEndpointInfo, strategy: ConnectionReuseStrategy
) -> None:
    model_endpoint.endpoint.connection_reuse_strategy = strategy
    with pytest.raises(ValueError, match="pooled connections"):
        SGLangSimulatorTransport(
            model_endpoint=model_endpoint, benchmark_id="benchmark-1"
        )


def test_multipart_requests_fail(model_endpoint: ModelEndpointInfo) -> None:
    model_endpoint.endpoint.request_content_type = (
        RequestContentType.MULTIPART_FORM_DATA
    )
    with pytest.raises(ValueError, match="JSON requests"):
        SGLangSimulatorTransport(
            model_endpoint=model_endpoint, benchmark_id="benchmark-1"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [param(False, id="dict"), param(True, id="bytes")])  # fmt: skip
async def test_send_preserves_payload_and_callback_with_coordinator_identity(
    transport: SGLangSimulatorTransport, request_info: RequestInfo, encoded: bool
) -> None:
    body = payload()
    body["messages"] = [{"role": "user", "content": "hi"}]
    original = orjson.dumps(body)
    record = RequestRecord(
        status=200,
        responses=[
            SSEMessage.parse(chunk, 999999)
            for chunk in response_body(request_id("benchmark-1", 2, 3))
            .strip()
            .split("\n\n")
        ],
    )
    post = AsyncMock(return_value=record)
    callback = AsyncMock(return_value=True)
    transport.aiohttp_client.post_request = post

    result = await transport.send_request(
        request_info, original if encoded else body, first_token_callback=callback
    )

    assert result is record
    expected = {**body, "rid": request_id("benchmark-1", 2, 3)}
    assert orjson.loads(post.call_args.args[1]) == expected
    assert request_info.payload_bytes == post.call_args.args[1]
    assert post.call_args.kwargs["first_token_callback"] is callback
    assert post.call_args.kwargs["cancel_after_ns"] is None
    assert orjson.dumps(body) == original
    assert result.server_timing.first_token_perf_ns == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, error",
    [
        param({"rid": "user-provided"}, "owns the request rid", id="dict-rid"),
        param(b'{"rid":"user-provided"}', "owns the request rid", id="bytes-rid"),
        param(b"[]", "JSON object", id="array"),
    ],
)  # fmt: skip
async def test_invalid_payload_is_rejected_before_http(
    transport: SGLangSimulatorTransport,
    request_info: RequestInfo,
    payload: dict[str, Any] | bytes,
    error: str,
) -> None:
    post = AsyncMock()
    transport.aiohttp_client.post_request = post
    with pytest.raises(ValueError, match=error):
        await transport.send_request(request_info, payload)
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_random_cancel_is_rejected_before_http(
    transport: SGLangSimulatorTransport, request_info: RequestInfo
) -> None:
    request_info.cancel_after_ns = 1
    post = AsyncMock()
    transport.aiohttp_client.post_request = post
    with pytest.raises(
        NotImplementedError, match="conversation 'conversation-1' turn 1"
    ):
        await transport.send_request(request_info, {})
    post.assert_not_awaited()


@pytest.mark.parametrize(
    "transport_type",
    [param(TransportType.HTTP, id="http"), param(TransportType.SGLANG_SIMULATOR, id="simulator")],
)  # fmt: skip
def test_inference_client_passes_identity_only_to_simulator(
    model_endpoint: ModelEndpointInfo, transport_type: TransportType
) -> None:
    model_endpoint.transport = transport_type
    constructor = MagicMock()
    endpoint = MagicMock()
    with patch(
        "aiperf.workers.inference_client.plugins.get_class",
        side_effect=[endpoint, constructor],
    ):
        InferenceClient(
            model_endpoint=model_endpoint,
            service_id="worker-1",
            benchmark_id="benchmark-1",
        )
    expected = {"model_endpoint": model_endpoint}
    if transport_type == TransportType.SGLANG_SIMULATOR:
        expected["benchmark_id"] = "benchmark-1"
    constructor.assert_called_once_with(**expected)


def test_virtual_record_never_subtracts_physical_credit_drop_time(
    model_endpoint: ModelEndpointInfo,
    request_info: RequestInfo,
) -> None:
    model_endpoint.transport = TransportType.SGLANG_SIMULATOR
    client = InferenceClient(model_endpoint, "worker", benchmark_id="benchmark-1")
    record = RequestRecord(error=ErrorDetails(message="payload failed"))
    request_info.turn_index = 0
    request_info.drop_perf_ns = 999999
    result = client._finalize_request_record(record=record, request_info=request_info)
    assert result.server_timing is not None
    assert result.credit_drop_latency is None
    assert result.start_perf_ns == 0 and result.end_perf_ns is None


@pytest.mark.asyncio
async def test_round_payloads_arrive_before_any_sse_response(
    transport: SGLangSimulatorTransport,
    request_info: RequestInfo,
    unused_tcp_port: int,
) -> None:
    received: set[str] = set()
    round_arrived = asyncio.Event()
    count = 4

    async def handler(request: web.Request) -> web.Response:
        body = orjson.loads(await request.read())
        received.add(body["rid"])
        if len(received) == count:
            round_arrived.set()
        await round_arrived.wait()
        return web.Response(
            text=response_body(body["rid"]),
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app, shutdown_timeout=1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    request_info.model_endpoint.endpoint.base_urls = [
        f"http://127.0.0.1:{unused_tcp_port}"
    ]
    try:
        records = await asyncio.wait_for(
            asyncio.gather(
                *(
                    transport.send_request(
                        request_info.model_copy(update={"credit_num": index}),
                        payload(),
                    )
                    for index in range(count)
                )
            ),
            timeout=5,
        )
    finally:
        round_arrived.set()
        await runner.cleanup()

    assert received == {request_id("benchmark-1", 2, index) for index in range(count)}
    assert all(record.error is None and record.status == 200 for record in records)
    assert all(record.responses[-1].get_text() == "[DONE]" for record in records)
