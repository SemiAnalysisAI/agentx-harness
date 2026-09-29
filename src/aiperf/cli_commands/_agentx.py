# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile-only model discovery and scoped runtime defaults for AgentX."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import aiohttp
import orjson

from aiperf.config.flags.agentx import AGENTX_SCENARIO

if TYPE_CHECKING:
    from aiperf.config import AIPerfConfig
    from aiperf.config.flags import CLIConfig


def models_url(url: str) -> str:
    """Preserve proxy prefixes; accept root, /v1, and chat-completions URLs."""
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("AgentX model discovery requires an HTTP(S) --url.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "AgentX model discovery requires a URL without credentials, query, "
            "or fragment; use --api-key/--header, or specify --model explicitly."
        )
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path.removesuffix("/chat/completions")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path + "/models", "", ""))


async def discover_model(endpoint: dict[str, Any]) -> str:
    """Select only an unambiguous model; never guess or follow auth redirects."""
    urls = endpoint.get("urls", ["http://localhost:8000"])
    if len(urls) != 1:
        raise ValueError(
            "AgentX auto-discovery needs one --url; pass --model explicitly."
        )
    url = models_url(urls[0])
    headers = dict(endpoint.get("headers") or {})
    if endpoint.get("api_key"):
        headers = {
            key: val for key, val in headers.items() if key.lower() != "authorization"
        }
        headers["Authorization"] = f"Bearer {endpoint['api_key']}"
    try:
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10), headers=headers
            ) as session,
            session.get(url, allow_redirects=False) as response,
        ):
            if response.status != 200:
                raise ValueError(
                    f"AgentX model discovery returned HTTP {response.status}; "
                    "check --url/authentication or pass --model explicitly."
                )
            body = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                body.extend(chunk)
                if len(body) > 1024 * 1024:
                    raise ValueError(
                        "AgentX /v1/models response exceeds 1 MiB; pass --model explicitly."
                    )
            payload = orjson.loads(body)
    except (aiohttp.ClientError, TimeoutError, orjson.JSONDecodeError):
        # Transport exceptions can include credentials or response bodies.
        raise ValueError(
            "AgentX could not read /v1/models; check --url/authentication "
            "or pass --model explicitly."
        ) from None
    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        not isinstance(data, list)
        or len(data) != 1
        or not isinstance(data[0], dict)
        or not isinstance(data[0].get("id"), str)
        or not data[0]["id"].strip()
    ):
        raise ValueError(
            "AgentX needs exactly one model in /v1/models; pass --model explicitly."
        )
    return data[0]["id"]


def resolve_profile_config(cli: CLIConfig) -> AIPerfConfig:
    """Discover an omitted model only for the opt-in profile preset."""
    from aiperf.config.flags._converter_endpoint import build_endpoint
    from aiperf.config.flags.resolver import (
        _normalize_loaded_benchmark_shorthands,
        deep_merge,
        resolve_config,
    )
    from aiperf.config.loader import load_config_dict

    data = load_config_dict(cli.config_file) if cli.config_file else None
    if data is not None:
        _normalize_loaded_benchmark_shorthands(data)
    benchmark = data.get("benchmark", {}) if data is not None else {}
    scenario = (
        cli.scenario
        if "scenario" in cli.model_fields_set
        else benchmark.get("scenario")
    )
    model_supplied = "model_names" in cli.model_fields_set or "models" in benchmark
    if scenario == AGENTX_SCENARIO and not model_supplied:
        endpoint = dict(benchmark.get("endpoint", {}))
        cli_endpoint = build_endpoint(cli)
        if data is not None and "urls" not in cli.model_fields_set:
            cli_endpoint.pop("urls", None)
        endpoint = deep_merge(endpoint, cli_endpoint)
        model = asyncio.run(discover_model(endpoint))
        cli = cli.model_copy(update={"model_names": [model]})
    config = resolve_config(cli, config_dict=data)
    if (
        scenario == AGENTX_SCENARIO
        and "dir" not in config.benchmark.artifacts.model_fields_set
    ):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        config.benchmark.artifacts.dir = (
            Path("artifacts") / f"agentx-{stamp}-{uuid4().hex[:8]}"
        )
    return config


@contextmanager
def agentx_environment(scenario: str | None) -> Iterator[None]:
    """Apply runtime defaults to this process and spawned services, then restore."""
    if scenario != AGENTX_SCENARIO:
        yield
        return
    from aiperf.common.environment import Environment

    defaults = {
        "DATASET": {"CONFIGURATION_TIMEOUT": 1800},
        "SERVICE": {"PROFILE_CONFIGURE_TIMEOUT": 1800},
        "UI": {"REALTIME_METRICS_ENABLED": True},
        "HTTP": {"TCP_USER_TIMEOUT": 900000},
    }
    previous: dict[str, Any] = {}
    inserted: list[str] = []
    try:
        for group, values in defaults.items():
            settings = getattr(Environment, group)
            updates = {}
            for key, value in values.items():
                env_key = f"AIPERF_{group}_{key}"
                if env_key not in os.environ and key not in settings.model_fields_set:
                    os.environ[env_key] = str(value)
                    inserted.append(env_key)
                    updates[key] = value
            previous[group] = settings
            setattr(
                Environment,
                group,
                type(settings)(**{**settings.model_dump(), **updates}),
            )
        type(Environment).model_validate(Environment.model_dump())
        yield
    finally:
        for group, settings in previous.items():
            setattr(Environment, group, settings)
        for key in inserted:
            os.environ.pop(key, None)
