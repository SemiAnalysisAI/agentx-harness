# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AssistantResponseMode(CaseInsensitiveStrEnum):
    """Source of assistant responses in Mooncake message conversations."""

    RECORDED = "recorded"
    """Replay each row's complete authored history."""

    LIVE = "live"
    """Accumulate message deltas and responses generated during the run."""
