# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-level file access for ``weka_replay`` captures.

Replay files have positional meaning (header, then sessions, then an optional
v5 trailing index) that a homogeneous line iterator doesn't express, and real
session lines reach ~54 MB — so this module uses ``mmap`` directly rather than
``BaseFileLoader._iter_record_dicts``. Lines are classified by byte prefix
before any JSON parse: line 1 is always the header; a line starting
``{"_schema":"replay-index-v1"`` is the v5 trailer (must be last, exactly one,
no session lines after it); anything else is a session line.

No module-level mutable state: every public function takes the file path (and
any already-parsed header/span) as arguments and opens its own ``mmap`` for
the duration of the call.
"""

from __future__ import annotations

import dataclasses
import logging
import mmap
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

import orjson
from pydantic import BaseModel, ValidationError

from aiperf.dataset.loader.weka_replay_models import (
    ReplayHeader,
    ReplayIndex,
    ReplaySession,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

logger = logging.getLogger(__name__)

_BOM = b"\xef\xbb\xbf"
_INDEX_PREFIX = b'{"_schema":"replay-index-v1"'
# Go's JSON encoder always emits session_id before start_ts, with no spaces.
_FAST_SESSION_RE = re.compile(
    rb'^\{"session_id":"((?:[^"\\]|\\.)*)","start_ts":"(?:[^"\\]|\\.)*"'
)
_MIN_TRAILER_BACKWARD_SCAN = (
    32 << 20
)  # 32 MiB: starting window for locating the v5 index's own start. Doubled
# (see _find_trailer_newline) rather than given up on, so a capture with far
# more sessions than this comfortably covers still finds its index -- at the
# cost of a few extra bounded re-scans -- instead of silently falling back to
# a full scan.


class WekaReplayLoadError(ValueError):
    """Raised whenever a ``weka_replay`` file cannot be parsed or is structurally invalid.

    Messages are shaped ``"weka_replay '<path>' line N: <message>"`` (or a
    more specific location when more context — session/instance/request — is
    available at the point of the error).
    """


@dataclasses.dataclass(frozen=True, slots=True)
class SessionSpan:
    """Byte location of one session line within a replay file."""

    session_id: str
    lineno: int
    offset: int
    length: int


def reject_inline_records(inline_records: object) -> None:
    """Raise if the caller tried to construct this loader with inline records.

    ``weka_replay`` only supports a single on-disk ``--input-file``; this
    format's positional structure (header, sessions, optional v5 trailer)
    doesn't have an inline-record equivalent.
    """
    if inline_records is not None:
        raise WekaReplayLoadError(
            "weka_replay does not support --inline-records in v1; pass a single "
            "--input-file path/to/replay.jsonl instead"
        )


def peek_header_schema(path: Path, max_bytes: int = 1 << 20) -> str | None:
    """Cheaply read just the first line's ``_schema`` value, for ``can_load`` probes.

    Never raises: any failure (missing file, directory, empty file, invalid
    JSON, missing/non-string ``_schema``) returns ``None`` so ``can_load`` can
    simply return ``False``.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(max_bytes)
    except OSError:
        return None
    line = _first_line(chunk)
    if not line:
        return None
    try:
        obj = orjson.loads(line)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    schema = obj.get("_schema")
    return schema if isinstance(schema, str) else None


def read_header(path: Path) -> ReplayHeader:
    """Parse and validate line 1 of the file as a :class:`ReplayHeader`."""
    _reject_directory(path)
    with _open_mmap(path) as mm:
        line, _next_pos = _read_line_at(mm, 0)
        return _parse_model(line, ReplayHeader, path, lineno=1, loc="header")


def scan_session_spans(
    path: Path, header: ReplayHeader
) -> tuple[list[SessionSpan], ReplayIndex | None]:
    """Build the list of :class:`SessionSpan` for every session in the file.

    For a ``replay-v5`` file with a well-formed trailing index, spans are
    built directly from it (O(1) relative to file size, aside from a bounded
    backward read to locate the trailer and a spot-check of the first span).
    Otherwise (v3/v4, or a corrupt/missing/ambiguous v5 index) falls back to
    one streaming forward pass over the whole file.
    """
    _reject_directory(path)
    with _open_mmap(path) as mm:
        _header_line, body_start = _read_line_at(mm, 0)
        if header.schema_version == "replay-v5":
            located = _try_locate_trailing_index(mm, body_start)
            if located is not None:
                offset, length = located
                index = _try_parse_index(mm, offset, length, path)
                if index is not None:
                    spans = _spans_from_index(index, path)
                    if spans and _spot_check_first_span(mm, spans[0]):
                        return spans, index
                    logger.warning(
                        "weka_replay '%s': replay-index-v1 trailer failed the "
                        "first-span spot-check; falling back to a full scan",
                        path,
                    )
        spans = _scan_all_session_lines(path, mm, body_start)
        return spans, None


def read_session(path: Path, span: SessionSpan) -> ReplaySession:
    """Read and validate exactly the byte range of one session, by span."""
    _reject_directory(path)
    with _open_mmap(path) as mm:
        raw = mm[span.offset : span.offset + span.length]
        return _parse_model(
            raw,
            ReplaySession,
            path,
            lineno=span.lineno,
            loc=f"session '{span.session_id}'",
        )


# --- internals --------------------------------------------------------------


def _reject_directory(path: Path) -> None:
    if path.is_dir():
        raise WekaReplayLoadError(
            f"weka_replay '{path}': expected a single .jsonl file, got a "
            "directory; weka_replay does not support directory input in v1"
        )


@contextmanager
def _open_mmap(path: Path) -> Iterator[mmap.mmap]:
    try:
        f = open(path, "rb")  # noqa: SIM115 - closed in finally below
    except OSError as exc:
        raise WekaReplayLoadError(
            f"weka_replay '{path}': cannot open file: {exc}"
        ) from exc
    try:
        size = os.fstat(f.fileno()).st_size
        if size == 0:
            raise WekaReplayLoadError(f"weka_replay '{path}': file is empty")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            yield mm
    finally:
        f.close()


def _first_line(chunk: bytes) -> bytes | None:
    if chunk.startswith(_BOM):
        chunk = chunk[len(_BOM) :]
    idx = chunk.find(b"\n")
    line = chunk if idx == -1 else chunk[:idx]
    if line.endswith(b"\r"):
        line = line[:-1]
    return line or None


def _read_line_at(mm: mmap.mmap, pos: int) -> tuple[bytes, int]:
    """Return (line bytes without terminator, offset of the next line).

    BOM is stripped only when ``pos == 0`` (start of file).
    """
    start = pos
    if start == 0 and mm[:3] == _BOM:
        start = 3
    nl = mm.find(b"\n", start)
    end = len(mm) if nl == -1 else nl
    line = mm[start:end]
    if line.endswith(b"\r"):
        line = line[:-1]
        end -= 1
    next_pos = len(mm) if nl == -1 else nl + 1
    return bytes(line), next_pos


def _iter_lines(
    mm: mmap.mmap, start_pos: int, start_lineno: int
) -> Iterator[tuple[int, bytes, int, int]]:
    """Yield (lineno, line bytes without terminator, offset, length) to EOF. Blank lines are skipped."""
    pos = start_pos
    lineno = start_lineno
    size = len(mm)
    while pos < size:
        nl = mm.find(b"\n", pos)
        end = size if nl == -1 else nl
        line = mm[pos:end]
        if line.endswith(b"\r"):
            line = line[:-1]
            end -= 1
        length = end - pos
        if length > 0:
            yield lineno, bytes(line), pos, length
        lineno += 1
        pos = size if nl == -1 else nl + 1


def _format_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return f"invalid data: {exc}"
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", "validation error")
    return f"{loc}: {msg}" if loc else msg


def _parse_model(
    raw: bytes, model: type[ModelT], path: Path, *, lineno: int, loc: str
) -> ModelT:
    try:
        obj = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise WekaReplayLoadError(
            f"weka_replay '{path}' line {lineno}: {loc}: invalid JSON: {exc}"
        ) from exc
    try:
        return model.model_validate(obj)
    except ValidationError as exc:
        raise WekaReplayLoadError(
            f"weka_replay '{path}' line {lineno}: {loc}: {_format_validation_error(exc)}"
        ) from exc


def _extract_session_id(line: bytes, path: Path, lineno: int) -> str:
    match = _FAST_SESSION_RE.match(line[:200])
    if match is not None:
        try:
            decoded = orjson.loads(b'"' + match.group(1) + b'"')
            if isinstance(decoded, str):
                return decoded
        except orjson.JSONDecodeError:
            pass
    try:
        obj = orjson.loads(line)
    except orjson.JSONDecodeError as exc:
        raise WekaReplayLoadError(
            f"weka_replay '{path}' line {lineno}: invalid JSON: {exc}"
        ) from exc
    sid = obj.get("session_id") if isinstance(obj, dict) else None
    if not isinstance(sid, str):
        raise WekaReplayLoadError(
            f"weka_replay '{path}' line {lineno}: missing or non-string 'session_id'"
        )
    return sid


def _scan_all_session_lines(
    path: Path, mm: mmap.mmap, body_start: int
) -> list[SessionSpan]:
    spans: list[SessionSpan] = []
    seen: dict[str, int] = {}
    index_seen_at: int | None = None
    for lineno, line, offset, length in _iter_lines(mm, body_start, 2):
        if line.startswith(_INDEX_PREFIX):
            if index_seen_at is not None:
                raise WekaReplayLoadError(
                    f"weka_replay '{path}' line {lineno}: duplicate replay-index-v1 "
                    f"trailer (first seen at line {index_seen_at}); exactly one "
                    "trailer is allowed"
                )
            index_seen_at = lineno
            continue
        if index_seen_at is not None:
            raise WekaReplayLoadError(
                f"weka_replay '{path}' line {lineno}: session line found after the "
                f"replay-index-v1 trailer (line {index_seen_at}); the trailer must "
                "be the last line in the file"
            )
        sid = _extract_session_id(line, path, lineno)
        if sid in seen:
            raise WekaReplayLoadError(
                f"weka_replay '{path}' line {lineno}: duplicate session_id '{sid}' "
                f"(first seen at line {seen[sid]})"
            )
        seen[sid] = lineno
        spans.append(
            SessionSpan(session_id=sid, lineno=lineno, offset=offset, length=length)
        )
    return spans


def _find_trailer_newline(mm: mmap.mmap, body_start: int, end: int) -> int | None:
    """Backward-search ``mm[body_start:end]`` for its last newline, widening
    the search window geometrically (starting at
    :data:`_MIN_TRAILER_BACKWARD_SCAN`) instead of giving up after one
    fixed-size attempt.

    Returns the newline's position, or ``None`` if ``mm[body_start:end]``
    contains no newline at all (meaning that whole span is one single line —
    not "the line is bigger than our first window", which instead triggers
    another, wider attempt).
    """
    window = _MIN_TRAILER_BACKWARD_SCAN
    while True:
        search_from = max(body_start, end - window)
        nl = mm.rfind(b"\n", search_from, end)
        if nl != -1:
            return nl
        if search_from <= body_start:
            return None
        window *= 2


def _try_locate_trailing_index(
    mm: mmap.mmap, body_start: int
) -> tuple[int, int] | None:
    """Return (offset, length) of the trailing index line's JSON bytes, or None.

    Only inspects the last line (and, as a guard against an ambiguous
    double-trailer, the line immediately before it) — never an unconditional
    full scan, though an index far larger than :data:`_MIN_TRAILER_BACKWARD_SCAN`
    still costs a few extra bounded re-scans (see :func:`_find_trailer_newline`).
    """
    size = len(mm)
    end = size
    while end > body_start and mm[end - 1] in (0x0A, 0x0D):
        end -= 1
    if end <= body_start:
        return None
    nl = _find_trailer_newline(mm, body_start, end)
    line_start = body_start if nl is None else nl + 1
    line = mm[line_start:end]
    if not line.startswith(_INDEX_PREFIX):
        return None
    # Guard against a second trailer-shaped line immediately preceding this one
    # (an ambiguous two-trailer file): bail out and let the full scan raise a
    # precise, line-numbered error instead of silently using just the last one.
    if line_start > body_start:
        prev_nl = _find_trailer_newline(mm, body_start, line_start - 1)
        prev_start = body_start if prev_nl is None else prev_nl + 1
        prev_line = mm[prev_start : line_start - 1]
        if prev_line.rstrip(b"\r").startswith(_INDEX_PREFIX):
            return None
    return line_start, end - line_start


def _try_parse_index(
    mm: mmap.mmap, offset: int, length: int, path: Path
) -> ReplayIndex | None:
    raw = mm[offset : offset + length]
    try:
        obj = orjson.loads(raw)
        return ReplayIndex.model_validate(obj)
    except (orjson.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "weka_replay '%s': malformed replay-index-v1 trailer (%s); falling "
            "back to a full scan",
            path,
            exc,
        )
        return None


def _spans_from_index(index: ReplayIndex, path: Path) -> list[SessionSpan]:
    spans: list[SessionSpan] = []
    seen: dict[str, int] = {}
    for i, sess in enumerate(index.sessions):
        if sess.session_id in seen:
            raise WekaReplayLoadError(
                f"weka_replay '{path}': replay-index-v1 trailer lists duplicate "
                f"session_id '{sess.session_id}' (entries {seen[sess.session_id]} "
                f"and {i})"
            )
        seen[sess.session_id] = i
        spans.append(
            SessionSpan(
                session_id=sess.session_id,
                lineno=i + 2,  # line 1 is the header; sessions follow in order.
                offset=sess.offset,
                length=sess.length,
            )
        )
    return spans


def _spot_check_first_span(mm: mmap.mmap, span: SessionSpan) -> bool:
    expected_prefix = b'{"session_id":' + orjson.dumps(span.session_id)
    read_len = min(max(64, len(expected_prefix)), span.length)
    probe = mm[span.offset : span.offset + read_len]
    return probe.startswith(expected_prefix)
