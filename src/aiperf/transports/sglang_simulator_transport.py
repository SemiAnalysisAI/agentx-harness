# SPDX-License-Identifier: Apache-2.0
"""HTTP transport for explicitly coordinated SGLang Simulator runs."""

from collections.abc import Mapping
from typing import Any

import orjson

from aiperf.common.enums import ConnectionReuseStrategy, RequestContentType
from aiperf.common.models import ModelEndpointInfo, RequestInfo, RequestRecord
from aiperf.plugin.enums import TransportType
from aiperf.simulation.config import validate_simulator_endpoint
from aiperf.simulation.identity import request_id
from aiperf.simulation.record_timing import apply_simulator_timing
from aiperf.transports.aiohttp_transport import AioHttpTransport
from aiperf.transports.base_transports import FirstTokenCallback, TransportMetadata


class SGLangSimulatorTransport(AioHttpTransport):
    """Reuse HTTP/SSE handling with coordinator-owned request identity."""

    def __init__(
        self,
        model_endpoint: ModelEndpointInfo,
        *,
        benchmark_id: str | None = None,
        tcp_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not benchmark_id:
            raise ValueError("sglang_simulator requires a benchmark_id")
        if (
            model_endpoint.endpoint.connection_reuse_strategy
            != ConnectionReuseStrategy.POOLED
        ):
            raise ValueError("sglang_simulator requires pooled connections")
        if model_endpoint.endpoint.request_content_type not in (
            None,
            RequestContentType.APPLICATION_JSON,
        ):
            raise ValueError("sglang_simulator requires JSON requests")
        validate_simulator_endpoint(model_endpoint.endpoint)
        self.benchmark_id = benchmark_id
        # Every payload in a Ready round must arrive before any response can finish.
        super().__init__(
            model_endpoint=model_endpoint,
            tcp_kwargs={**(tcp_kwargs or {}), "limit": 0, "limit_per_host": 0},
            **kwargs,
        )

    @classmethod
    def metadata(cls) -> TransportMetadata:
        return TransportMetadata(
            transport_type=TransportType.SGLANG_SIMULATOR,
            url_schemes=[],
        )

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any] | bytes,
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        location = f"conversation {request_info.conversation_id!r} turn {request_info.turn_index}"
        if request_info.cancel_after_ns is not None:
            raise NotImplementedError(
                f"{location}: sglang_simulator does not support random cancellation"
            )
        body = orjson.loads(payload) if isinstance(payload, bytes) else payload.copy()
        if not isinstance(body, dict):
            raise ValueError(f"{location}: simulator payload must be a JSON object")
        if "rid" in body:
            raise ValueError(f"{location}: simulator owns the request rid")
        if (
            body.get("stream") is not True
            or type(body.get("n", 1)) is not int
            or body.get("n", 1) != 1
        ):
            raise NotImplementedError(
                f"{location}: simulator requires streaming with n=1"
            )
        if (
            not isinstance(body.get("stream_options"), dict)
            or body["stream_options"].get("include_usage") is not True
        ):
            raise NotImplementedError(
                f"{location}: simulator requires stream_options.include_usage"
            )
        body["rid"] = request_id(
            self.benchmark_id, request_info.phase_index, request_info.credit_num
        )
        request_info.payload_bytes = orjson.dumps(body)
        record = await super().send_request(
            request_info,
            request_info.payload_bytes,
            first_token_callback=first_token_callback,
        )
        return apply_simulator_timing(record, body["rid"])
