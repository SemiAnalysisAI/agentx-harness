# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixture builders for ``weka_replay`` tests.

Mirrors the ``write_lines``-style helper used by ``test_dag_jsonl.py``: build
minimal-but-valid dicts for each level of the schema, then write them as a
``.jsonl`` file. :func:`write_v5_with_index` additionally computes real byte
offsets for a well-formed v5 trailing index (no hand-waved numbers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from aiperf.dataset.loader._weka_replay_ir import (
    SessionDiagnostics,
    SessionIR,
    normalize_session,
)
from aiperf.dataset.loader.weka_replay_models import ReplaySession


def build_header(schema_version: str = "replay-v3", **overrides: Any) -> dict:
    header = {
        "_schema": schema_version,
        "name": "test-capture",
        "generated_at": "2024-01-01T00:00:00Z",
        "source": "test",
        "summary": {},
    }
    header.update(overrides)
    return header


def build_request(
    request_id: int = 1,
    ts: str = "2024-01-01T00:00:01Z",
    model: str = "claude-3-opus",
    stream: bool = False,
    input_tokens: int = 100,
    prefill_tokens: int = 100,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    output_tokens: int = 50,
    **overrides: Any,
) -> dict:
    request = {
        "request_id": request_id,
        "ts": ts,
        "model": model,
        "stream": stream,
        "input_tokens": input_tokens,
        "prefill_tokens": prefill_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "output_tokens": output_tokens,
    }
    request.update(overrides)
    return request


def build_instance(
    instance_id: str = "sess1:sig:fp",
    role: str = "main",
    requests: list[dict] | None = None,
    **overrides: Any,
) -> dict:
    instance = {
        "instance_id": instance_id,
        "role": role,
        "requests": requests if requests is not None else [build_request()],
    }
    instance.update(overrides)
    return instance


def build_session(
    session_id: str = "sess1",
    start_ts: str = "2024-01-01T00:00:00Z",
    instances: list[dict] | None = None,
    **overrides: Any,
) -> dict:
    session = {
        "session_id": session_id,
        "start_ts": start_ts,
        "instances": instances if instances is not None else [build_instance()],
    }
    session.update(overrides)
    return session


def normalize_session_dict(
    session_dict: dict, *, schema_version: str = "replay-v3", lineno: int = 2
) -> tuple[SessionIR, SessionDiagnostics]:
    """Run a raw session dict through real pydantic validation and the real
    ``normalize_session`` IR step, returning ``(SessionIR, SessionDiagnostics)``.

    Lets tree-planning tests exercise the real ``normalize_session ->
    plan_session_tree`` pipeline (so a regression in the IR layer's link
    classification is also caught by tree tests) without hand-rolling raw
    ``ReplaySession`` JSON and a byte-level file for every case.
    """
    raw = ReplaySession.model_validate(session_dict)
    diag = SessionDiagnostics()
    ir = normalize_session(raw, schema_version=schema_version, lineno=lineno, diag=diag)
    return ir, diag


def write_lines(
    tmp_path: Path, lines: list[dict], filename: str = "replay.jsonl"
) -> Path:
    """Write a header + session dicts verbatim, one JSON object per line.

    No trailing index: the v3/v4 shape, and also usable for a deliberately
    malformed v5 file (e.g. a hand-inserted trailer that isn't last).
    """
    path = tmp_path / filename
    path.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))
    return path


class FakeTokenizer:
    """Minimal tokenizer stand-in: ``decode`` just stringifies token ids.

    Avoids loading a real HuggingFace/tiktoken tokenizer in fast unit tests
    that only need *some* deterministic text back from a token-id sequence.
    """

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"tok{t}" for t in token_ids)


class FakePromptGenerator:
    """Minimal stand-in for ``PromptGenerator``, exposing exactly the surface
    ``WekaReplayLoader``'s ``_PromptGeneratorCorpusAdapter`` touches
    (``tokenized_corpus`` and ``tokenizer.decode``), so loader-level unit
    tests don't need to build a real tokenizer/corpus.
    """

    def __init__(self, corpus_size: int = 10_000) -> None:
        self.tokenized_corpus: list[int] = list(range(corpus_size))
        self.tokenizer = FakeTokenizer()


def write_v5_with_index(
    tmp_path: Path,
    header: dict,
    sessions: list[dict],
    filename: str = "replay.jsonl",
) -> Path:
    """Write a well-formed v5 file: header, sessions, then a real trailing
    ``replay-index-v1`` line whose offsets/lengths are computed from the
    actual encoded bytes (never hand-waved)."""
    path = tmp_path / filename
    chunks: list[bytes] = [orjson.dumps(header)]
    index_sessions = []
    offset = len(chunks[0]) + 1  # +1: the newline terminating the header line.
    for sess in sessions:
        encoded = orjson.dumps(sess)
        request_count = sum(
            len(inst["requests"]) for inst in sess.get("instances") or []
        )
        index_sessions.append(
            {
                "session_id": sess["session_id"],
                "offset": offset,
                "length": len(encoded),
                "start_ts": sess["start_ts"],
                "end_ts": sess["start_ts"],
                "requests": request_count,
            }
        )
        chunks.append(encoded)
        offset += len(encoded) + 1  # +1: the newline terminating this session line.
    index_line = {"_schema": "replay-index-v1", "sessions": index_sessions}
    chunks.append(orjson.dumps(index_line))
    path.write_bytes(b"\n".join(chunks))
    return path
