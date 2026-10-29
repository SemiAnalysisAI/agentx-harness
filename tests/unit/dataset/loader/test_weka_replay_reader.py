# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.dataset.loader import _weka_replay_reader as reader
from aiperf.dataset.loader._weka_replay_reader import (
    SessionSpan,
    WekaReplayLoadError,
    peek_header_schema,
    read_header,
    read_session,
    reject_inline_records,
    scan_session_spans,
)
from tests.unit.dataset.loader._weka_replay_builders import (
    build_header,
    build_instance,
    build_request,
    build_session,
    write_lines,
    write_v5_with_index,
)


class TestPeekHeaderSchema:
    def test_valid_header(self, tmp_path):
        path = write_lines(tmp_path, [build_header(schema_version="replay-v4")])
        assert peek_header_schema(path) == "replay-v4"

    def test_missing_file_returns_none(self, tmp_path):
        assert peek_header_schema(tmp_path / "nope.jsonl") is None

    def test_directory_returns_none(self, tmp_path):
        assert peek_header_schema(tmp_path) is None

    def test_not_json_returns_none(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        path.write_bytes(b"not json at all\n")
        assert peek_header_schema(path) is None

    def test_json_without_schema_returns_none(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        path.write_bytes(orjson.dumps({"foo": "bar"}))
        assert peek_header_schema(path) is None

    def test_non_dict_json_returns_none(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        path.write_bytes(orjson.dumps([1, 2, 3]))
        assert peek_header_schema(path) is None

    def test_non_string_schema_returns_none(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        path.write_bytes(orjson.dumps({"_schema": 5}))
        assert peek_header_schema(path) is None

    def test_handles_bom_and_crlf(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        line = orjson.dumps(build_header(schema_version="replay-v3"))
        path.write_bytes(b"\xef\xbb\xbf" + line + b"\r\n" + b"more stuff\r\n")
        assert peek_header_schema(path) == "replay-v3"


class TestRejectInlineRecords:
    def test_none_is_ok(self):
        reject_inline_records(None)

    def test_non_none_raises(self):
        with pytest.raises(WekaReplayLoadError, match="--inline-records"):
            reject_inline_records([{"a": 1}])


class TestReadHeader:
    def test_reads_valid_header(self, tmp_path):
        path = write_lines(tmp_path, [build_header(schema_version="replay-v5")])
        header = read_header(path)
        assert header.schema_version == "replay-v5"

    def test_directory_raises(self, tmp_path):
        with pytest.raises(WekaReplayLoadError, match="directory"):
            read_header(tmp_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(WekaReplayLoadError, match="cannot open"):
            read_header(tmp_path / "nope.jsonl")

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        path.write_bytes(b"")
        with pytest.raises(WekaReplayLoadError, match="empty"):
            read_header(path)

    def test_invalid_json_raises_with_line_number(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        path.write_bytes(b"{not valid json")
        with pytest.raises(WekaReplayLoadError, match=r"line 1"):
            read_header(path)

    def test_schema_validation_error_raises(self, tmp_path):
        raw = build_header()
        del raw["name"]
        path = write_lines(tmp_path, [raw])
        with pytest.raises(WekaReplayLoadError, match=r"line 1"):
            read_header(path)

    def test_bom_handled(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        line = orjson.dumps(build_header(schema_version="replay-v3"))
        path.write_bytes(b"\xef\xbb\xbf" + line)
        header = read_header(path)
        assert header.schema_version == "replay-v3"

    def test_crlf_handled(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        header_line = orjson.dumps(build_header(schema_version="replay-v3"))
        session_line = orjson.dumps(build_session())
        path.write_bytes(header_line + b"\r\n" + session_line + b"\r\n")
        header = read_header(path)
        assert header.schema_version == "replay-v3"


class TestScanSessionSpansV3:
    def test_single_session(self, tmp_path):
        header_dict = build_header(schema_version="replay-v3")
        session_dict = build_session(session_id="sess1")
        path = write_lines(tmp_path, [header_dict, session_dict])
        header = read_header(path)
        spans, index = scan_session_spans(path, header)
        assert index is None
        assert len(spans) == 1
        assert spans[0].session_id == "sess1"
        assert spans[0].lineno == 2
        session = read_session(path, spans[0])
        assert session.session_id == "sess1"

    def test_multiple_sessions(self, tmp_path):
        header_dict = build_header(schema_version="replay-v3")
        sessions = [build_session(session_id=f"sess{i}") for i in range(5)]
        path = write_lines(tmp_path, [header_dict, *sessions])
        header = read_header(path)
        spans, index = scan_session_spans(path, header)
        assert index is None
        assert [s.session_id for s in spans] == [f"sess{i}" for i in range(5)]
        assert [s.lineno for s in spans] == [2, 3, 4, 5, 6]
        for span in spans:
            session = read_session(path, span)
            assert session.session_id == span.session_id

    def test_duplicate_session_id_raises_with_both_line_numbers(self, tmp_path):
        header_dict = build_header(schema_version="replay-v3")
        s1 = build_session(session_id="dup")
        s2 = build_session(session_id="other")
        s3 = build_session(session_id="dup")
        path = write_lines(tmp_path, [header_dict, s1, s2, s3])
        header = read_header(path)
        with pytest.raises(
            WekaReplayLoadError, match=r"line 4.*duplicate session_id 'dup'.*line 2"
        ):
            scan_session_spans(path, header)

    def test_session_after_index_raises(self, tmp_path):
        header_dict = build_header(schema_version="replay-v3")
        s1 = build_session(session_id="sess1")
        index_line = {"_schema": "replay-index-v1", "sessions": []}
        s2 = build_session(session_id="sess2")
        path = write_lines(tmp_path, [header_dict, s1, index_line, s2])
        header = read_header(path)
        with pytest.raises(WekaReplayLoadError, match="session line found after"):
            scan_session_spans(path, header)

    def test_two_index_lines_raises(self, tmp_path):
        header_dict = build_header(schema_version="replay-v3")
        s1 = build_session(session_id="sess1")
        index_line = {"_schema": "replay-index-v1", "sessions": []}
        path = write_lines(tmp_path, [header_dict, s1, index_line, index_line])
        header = read_header(path)
        with pytest.raises(
            WekaReplayLoadError, match="duplicate replay-index-v1 trailer"
        ):
            scan_session_spans(path, header)

    def test_index_not_last_raises(self, tmp_path):
        """An index line present but not the final line is a 'session after
        index' violation regardless of header schema version."""
        header_dict = build_header(schema_version="replay-v3")
        index_line = {"_schema": "replay-index-v1", "sessions": []}
        s1 = build_session(session_id="sess1")
        path = write_lines(tmp_path, [header_dict, index_line, s1])
        header = read_header(path)
        with pytest.raises(WekaReplayLoadError, match="session line found after"):
            scan_session_spans(path, header)


class TestScanSessionSpansV5FastPath:
    def test_uses_index_without_full_scan(self, tmp_path, monkeypatch):
        header_dict = build_header(schema_version="replay-v5")
        sessions = [build_session(session_id=f"sess{i}") for i in range(4)]
        path = write_v5_with_index(tmp_path, header_dict, sessions)
        header = read_header(path)

        calls = {"n": 0}
        original = reader._scan_all_session_lines

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(reader, "_scan_all_session_lines", _spy)

        spans, index = scan_session_spans(path, header)

        assert calls["n"] == 0, "fast path must not fall back to a full scan"
        assert index is not None
        assert [s.session_id for s in spans] == [f"sess{i}" for i in range(4)]
        for span in spans:
            session = read_session(path, span)
            assert session.session_id == span.session_id

    def test_spans_match_index_offsets_exactly(self, tmp_path):
        header_dict = build_header(schema_version="replay-v5")
        sessions = [build_session(session_id=f"sess{i}") for i in range(3)]
        path = write_v5_with_index(tmp_path, header_dict, sessions)
        header = read_header(path)
        spans, index = scan_session_spans(path, header)
        assert index is not None
        for span, idx_sess in zip(spans, index.sessions, strict=True):
            assert span.offset == idx_sess.offset
            assert span.length == idx_sess.length

    def test_corrupted_session_line_within_index_range_only_caught_via_read(
        self, tmp_path
    ):
        """Proves the fast path truly reads by offset: the index's recorded
        length/offset for sess1 is made to point at garbage bytes that would
        only ever be noticed when something actually reads that byte range
        (i.e. read_session), never during span construction itself."""
        header_dict = build_header(schema_version="replay-v5")
        sessions = [
            build_session(session_id="sess0"),
            build_session(session_id="sess1"),
        ]
        path = write_v5_with_index(tmp_path, header_dict, sessions)
        header = read_header(path)

        spans, index = scan_session_spans(path, header)
        assert index is not None
        # span construction itself succeeds purely from index metadata.
        assert len(spans) == 2

        # Corrupt the on-disk bytes for sess1 in place (same length, invalid JSON).
        sess1_span = next(s for s in spans if s.session_id == "sess1")
        with open(path, "r+b") as f:
            f.seek(sess1_span.offset)
            garbage = b"x" * sess1_span.length
            f.write(garbage)

        with pytest.raises(WekaReplayLoadError):
            read_session(path, sess1_span)


class TestScanSessionSpansV5Fallback:
    def test_missing_index_falls_back_to_scan(self, tmp_path, monkeypatch):
        header_dict = build_header(schema_version="replay-v5")
        sessions = [build_session(session_id=f"sess{i}") for i in range(3)]
        path = write_lines(tmp_path, [header_dict, *sessions])  # no trailing index
        header = read_header(path)

        calls = {"n": 0}
        original = reader._scan_all_session_lines

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(reader, "_scan_all_session_lines", _spy)

        spans, index = scan_session_spans(path, header)
        assert calls["n"] == 1
        assert index is None
        assert [s.session_id for s in spans] == [f"sess{i}" for i in range(3)]

    def test_corrupted_index_json_falls_back_and_produces_correct_spans(self, tmp_path):
        header_dict = build_header(schema_version="replay-v5")
        sessions = [build_session(session_id=f"sess{i}") for i in range(3)]
        path = write_v5_with_index(tmp_path, header_dict, sessions)
        # Corrupt the trailing index line (same total file length considerations
        # don't matter here - we just need the last line to be invalid JSON that
        # still starts with the index prefix so the fast path attempts and fails).
        content = path.read_bytes()
        lines = content.split(b"\n")
        assert lines[-1].startswith(b'{"_schema":"replay-index-v1"')
        lines[-1] = b'{"_schema":"replay-index-v1", NOT VALID JSON'
        path.write_bytes(b"\n".join(lines))

        header = read_header(path)
        spans, index = scan_session_spans(path, header)
        assert index is None
        assert [s.session_id for s in spans] == [f"sess{i}" for i in range(3)]

    def test_index_spot_check_mismatch_falls_back(self, tmp_path):
        header_dict = build_header(schema_version="replay-v5")
        sessions = [build_session(session_id=f"sess{i}") for i in range(3)]
        path = write_v5_with_index(tmp_path, header_dict, sessions)
        content = path.read_bytes()
        lines = content.split(b"\n")
        index_obj = orjson.loads(lines[-1])
        # Point the first session's offset somewhere else in the file so the
        # spot-check (bytes at that offset must start with {"session_id":"sess0")
        # fails, forcing a fallback to the full scan.
        index_obj["sessions"][0]["offset"] = index_obj["sessions"][1]["offset"]
        lines[-1] = orjson.dumps(index_obj)
        path.write_bytes(b"\n".join(lines))

        header = read_header(path)
        spans, index = scan_session_spans(path, header)
        assert index is None
        assert [s.session_id for s in spans] == [f"sess{i}" for i in range(3)]


class TestLargeTrailerWindowScaling:
    """The backward scan for the index's start line widens geometrically
    (see _find_trailer_newline) instead of giving up after one fixed-size
    attempt, so an index far bigger than the starting window is still found
    rather than silently triggering a full scan. A real multi-MB index is
    impractical to construct in a unit test, so these shrink the starting
    window instead (same effect: index size > window size on the first try).
    """

    def test_index_far_larger_than_starting_window_still_uses_fast_path(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(reader, "_MIN_TRAILER_BACKWARD_SCAN", 4)
        header_dict = build_header(schema_version="replay-v5")
        sessions = [build_session(session_id=f"sess{i}") for i in range(20)]
        path = write_v5_with_index(tmp_path, header_dict, sessions)
        header = read_header(path)

        calls = {"n": 0}
        original = reader._scan_all_session_lines

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(reader, "_scan_all_session_lines", _spy)

        spans, index = scan_session_spans(path, header)

        assert calls["n"] == 0, "fast path must not fall back to a full scan"
        assert index is not None
        assert [s.session_id for s in spans] == [f"sess{i}" for i in range(20)]

    def test_shrunk_window_still_detects_ambiguous_double_trailer(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(reader, "_MIN_TRAILER_BACKWARD_SCAN", 4)
        header_dict = build_header(schema_version="replay-v5")
        index_line = {"_schema": "replay-index-v1", "sessions": []}
        s1 = build_session(session_id="sess1")
        path = write_lines(tmp_path, [header_dict, index_line, index_line, s1])
        header = read_header(path)
        with pytest.raises(
            WekaReplayLoadError, match="duplicate replay-index-v1 trailer"
        ):
            scan_session_spans(path, header)


class TestReadSession:
    def test_directory_raises(self, tmp_path):
        span = SessionSpan(session_id="x", lineno=2, offset=0, length=1)
        with pytest.raises(WekaReplayLoadError, match="directory"):
            read_session(tmp_path, span)

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "replay.jsonl"
        path.write_bytes(b"not json")
        span = SessionSpan(session_id="x", lineno=1, offset=0, length=len(b"not json"))
        with pytest.raises(WekaReplayLoadError, match="invalid JSON"):
            read_session(path, span)

    def test_validation_error_raises(self, tmp_path):
        header_dict = build_header(schema_version="replay-v3")
        bad_session = build_session()
        del bad_session["session_id"]
        path = write_lines(tmp_path, [header_dict, bad_session])
        # Build the span manually since scan_session_spans itself requires
        # session_id to be present to build a SessionSpan in the first place.
        header_line = orjson.dumps(header_dict)
        offset = len(header_line) + 1
        encoded = orjson.dumps(bad_session)
        span = SessionSpan(
            session_id="<unknown>", lineno=2, offset=offset, length=len(encoded)
        )
        with pytest.raises(WekaReplayLoadError, match="session_id"):
            read_session(path, span)


class TestLargeLine:
    def test_large_session_line_parses(self, tmp_path):
        """Stand-in (~5 MB) for the ~54 MB real-world session line case."""
        big_request = build_request()
        num_blocks = 80_000
        big_request["system_blocks"] = [
            {"type": "text", "hash": f"sha256:{i:016x}", "bytes": 1000, "tokens": 100}
            for i in range(num_blocks)
        ]
        instance = build_instance(requests=[big_request])
        session_dict = build_session(session_id="big", instances=[instance])
        header_dict = build_header(schema_version="replay-v3")
        path = write_lines(tmp_path, [header_dict, session_dict])
        assert path.stat().st_size > 4_000_000

        header = read_header(path)
        spans, _index = scan_session_spans(path, header)
        assert len(spans) == 1
        session = read_session(path, spans[0])
        assert session.session_id == "big"
        assert len(session.instances[0].requests[0].system_blocks) == num_blocks
