# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fire-and-forget inference-result push logs a failure instead of losing it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from aiperf.common.exceptions import CommunicationError
from aiperf.workers.worker import Worker


async def test_failed_push_is_logged() -> None:
    owner = SimpleNamespace(
        inference_results_push_client=SimpleNamespace(
            push=AsyncMock(side_effect=CommunicationError("encode failed"))
        ),
        exception=MagicMock(),
    )

    await Worker._push_inference_result(owner, MagicMock())

    owner.exception.assert_called_once()
    assert "encode failed" in owner.exception.call_args.args[0]


async def test_successful_push_logs_nothing() -> None:
    msg = MagicMock()
    owner = SimpleNamespace(
        inference_results_push_client=SimpleNamespace(push=AsyncMock()),
        exception=MagicMock(),
    )

    await Worker._push_inference_result(owner, msg)

    owner.inference_results_push_client.push.assert_awaited_once_with(msg)
    owner.exception.assert_not_called()
