# SPDX-License-Identifier: Apache-2.0
"""Stable request identity shared by the timing manager and HTTP workers."""

import hashlib

VIRTUAL_TIMESTAMP_ORIGIN_NS = 1_700_000_000_000_000_000
"""Shared synthetic epoch for published timestamps; elapsed time starts at zero."""


def request_id(benchmark_id: str, phase_index: int | None, credit_num: int) -> str:
    """Encode a credit without phase collisions or request-ID prefix aliases."""
    if not benchmark_id:
        raise ValueError("simulation requires a benchmark ID")
    if phase_index is not None and not 0 <= phase_index < 0xFFFFFFFF:
        raise ValueError("simulation phase index must fit in 32 bits")
    if not 0 <= credit_num < 1 << 64:
        raise ValueError("simulation credit number must fit in 64 bits")
    run = hashlib.sha256(benchmark_id.encode()).hexdigest()[:32]
    # Agentic replay's synthesized warmup has no explicit phase index.
    phase = 0xFFFFFFFF if phase_index is None else phase_index
    return f"{run}.{phase:08x}.{credit_num:016x}"
