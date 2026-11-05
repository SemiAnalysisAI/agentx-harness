# SPDX-License-Identifier: Apache-2.0
"""Optional generation boundaries supplied by a coordinated simulator."""

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class ServerTiming(AIPerfBaseModel):
    """Generation times in the request's performance-clock domain.

    An empty object means server timing was required but unavailable. Consumers
    must not fall back to client response timestamps in that case.
    """

    first_token_perf_ns: int | None = Field(
        default=None,
        ge=0,
        description="First committed generation token in the request performance-clock domain.",
    )
    last_token_perf_ns: int | None = Field(
        default=None,
        ge=0,
        description="Last committed generation token in the request performance-clock domain.",
    )
