# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end unit tests for ``WekaReplayLoader`` against a real, hand-built
2-level fan-out fixture (root -> sync child -> async grandchild).

Uses :class:`FakePromptGenerator` (a lightweight ``TokenCorpusSource``-
compatible stand-in, not a real tokenizer) as the injected ``prompt_generator``
to keep this fast, matching how the composer would inject one for an
``is_trace: true`` loader but without paying for real tokenization.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader import weka_replay
from aiperf.dataset.loader._weka_replay_content import TurnContent
from aiperf.dataset.loader._weka_replay_ir import SessionDiagnostics, normalize_session
from aiperf.dataset.loader._weka_replay_timing import compute_branch_start_ms
from aiperf.dataset.loader.weka_replay import WekaReplayLoader
from aiperf.dataset.loader.weka_replay_models import ReplaySession
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli
from tests.unit.dataset.loader._weka_replay_builders import (
    FakePromptGenerator,
    build_header,
    build_instance,
    build_request,
    build_session,
    write_lines,
)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)
_EMPTY_TURN_CONTENT = TurnContent(raw_messages=[], raw_tools=None, reset_context=False)


def _ts(offset_ms: float) -> str:
    dt = _BASE + timedelta(milliseconds=offset_ms)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _msg(hash_suffix: str, role: str = "user", tokens: int = 20) -> dict:
    return {
        "hash": f"sha256:{hash_suffix.ljust(16, '0')}",
        "bytes": 100,
        "tokens": tokens,
        "role": role,
    }


def _build_session_dict() -> dict:
    main = build_instance(
        instance_id="root:main",
        role="main",
        requests=[
            build_request(
                request_id=1,
                ts=_ts(0),
                total_ms=500,
                messages=[_msg("m1")],
            ),
            build_request(
                request_id=2,
                ts=_ts(2000),
                total_ms=500,
                messages=[_msg("m1"), _msg("m2", role="assistant")],
            ),
        ],
    )
    # Synchronous SPAWN child: ends (900ms) well before main's turn 1 (2000ms).
    sync_child = build_instance(
        instance_id="root:sync",
        role="sub-agent",
        fan_out_group="1",
        parent_spawn_request_id=1,
        fan_out_size=1,
        fan_out_position=0,
        requests=[
            build_request(
                request_id=11,
                ts=_ts(600),
                total_ms=300,
                messages=[_msg("c1")],
            )
        ],
    )
    # Grandchild spawned off the sync child's only (= last) turn -> always
    # async, regardless of timing (no s+1 exists on the parent to gate).
    grandchild = build_instance(
        instance_id="root:sync:grand",
        role="sub-agent",
        fan_out_group="11",
        parent_spawn_request_id=11,
        fan_out_size=1,
        fan_out_position=0,
        requests=[
            build_request(
                request_id=111,
                ts=_ts(650),
                total_ms=50,
                messages=[_msg("g1")],
            )
        ],
    )
    return build_session(session_id="sessA", instances=[main, sync_child, grandchild])


def _build_loader(
    tmp_path: Path, *, schema_version: str = "replay-v3", fixed_schedule: bool = False
) -> WekaReplayLoader:
    header = build_header(schema_version)
    session = _build_session_dict()
    path = write_lines(tmp_path, [header, session])
    cli = CLIConfig(
        model_names=["model-a", "model-b"],
        input_file=str(path),
        custom_dataset_type=CustomDatasetType.WEKA_REPLAY,
        fixed_schedule=fixed_schedule,
    )
    run = make_run_from_cli(cli)
    return WekaReplayLoader(
        filename=path, run=run, prompt_generator=FakePromptGenerator()
    )


def _simple_instance(instance_id: str, role: str, request_id: int, **overrides) -> dict:
    return build_instance(
        instance_id=instance_id,
        role=role,
        requests=[
            build_request(
                request_id=request_id, ts=_ts(0), total_ms=100, messages=[_msg("x")]
            )
        ],
        **overrides,
    )


def _build_loader_from_sessions(
    tmp_path: Path,
    sessions: list[dict],
    *,
    schema_version: str = "replay-v3",
    **cli_overrides,
) -> WekaReplayLoader:
    header = build_header(schema_version)
    path = write_lines(tmp_path, [header, *sessions])
    cli = CLIConfig(
        model_names=["model-a"],
        input_file=str(path),
        custom_dataset_type=CustomDatasetType.WEKA_REPLAY,
        **cli_overrides,
    )
    run = make_run_from_cli(cli)
    return WekaReplayLoader(
        filename=path, run=run, prompt_generator=FakePromptGenerator()
    )


class TestFixedScheduleGuard:
    def test_fixed_schedule_raises_at_construction(self, tmp_path: Path):
        with pytest.raises(NotImplementedError, match="FIXED_SCHEDULE"):
            _build_loader(tmp_path, fixed_schedule=True)

    def test_plain_concurrency_does_not_raise(self, tmp_path: Path):
        _build_loader(tmp_path, fixed_schedule=False)  # must not raise


class TestLoadAndConvert:
    @pytest.fixture
    def conversations(self, tmp_path: Path):
        loader = _build_loader(tmp_path)
        data = loader.load_dataset()
        assert set(data) == {"sessions"}
        assert len(data["sessions"]) == 1
        return loader.convert_to_conversations(data)

    def test_conversation_count_and_ids(self, conversations):
        by_id = {c.session_id: c for c in conversations}
        assert set(by_id) == {"root:main", "root:sync", "root:sync:grand"}

    def test_depths_and_root_flag(self, conversations):
        by_id = {c.session_id: c for c in conversations}
        root = by_id["root:main"]
        sync_conv = by_id["root:sync"]
        grand_conv = by_id["root:sync:grand"]

        assert root.is_root is True
        assert root.agent_depth == 0
        assert root.parent_conversation_id is None

        assert sync_conv.is_root is False
        assert sync_conv.agent_depth == 1
        assert sync_conv.parent_conversation_id == "root:main"

        assert grand_conv.is_root is False
        assert grand_conv.agent_depth == 2
        assert grand_conv.parent_conversation_id == "root:sync"

        # Only the true primary root is_root=True.
        assert sum(1 for c in conversations if c.is_root) == 1

    def test_sync_branch_join_turn_is_declaring_plus_one(self, conversations):
        by_id = {c.session_id: c for c in conversations}
        root = by_id["root:main"]

        assert len(root.branches) == 1
        branch = root.branches[0]
        assert branch.mode == ConversationBranchMode.SPAWN
        assert branch.child_conversation_ids == ["root:sync"]
        assert branch.is_background is False

        declaring_turn = 0
        join_turn = declaring_turn + 1
        assert root.turns[declaring_turn].branch_ids == [branch.branch_id]
        join_prereqs = root.turns[join_turn].prerequisites
        assert any(
            p.kind == PrerequisiteKind.SPAWN_JOIN and p.branch_id == branch.branch_id
            for p in join_prereqs
        )
        # No other turn carries this branch's prerequisite.
        for idx, turn in enumerate(root.turns):
            if idx != join_turn:
                assert branch.branch_id not in [p.branch_id for p in turn.prerequisites]

    def test_grandchild_spawn_is_background_with_no_join(self, conversations):
        by_id = {c.session_id: c for c in conversations}
        sync_conv = by_id["root:sync"]

        assert len(sync_conv.branches) == 1
        branch = sync_conv.branches[0]
        assert branch.is_background is True
        assert branch.child_conversation_ids == ["root:sync:grand"]
        assert sync_conv.turns[0].branch_ids == [branch.branch_id]
        assert all(not t.prerequisites for t in sync_conv.turns)

    def test_turn_zero_raw_messages_non_empty_and_not_a_delta(self, conversations):
        by_id = {c.session_id: c for c in conversations}
        root = by_id["root:main"]
        assert root.turns[0].raw_messages
        assert root.turns[0].reset_context is False

    def test_theoretical_cache_fields_populated(self, conversations):
        populated = [
            t
            for conv in conversations
            for t in conv.turns
            if t.theoretical_prefix_cache_hit_blocks is not None
            and t.theoretical_prefix_cache_total_blocks is not None
        ]
        assert populated, "expected at least one turn with theoretical cache fields set"

    def test_model_names_mapped_to_configured_models(self, conversations):
        by_id = {c.session_id: c for c in conversations}
        for conv in by_id.values():
            for turn in conv.turns:
                assert turn.model in {"model-a", "model-b"}

    def test_max_tokens_is_at_least_one(self, conversations):
        for conv in conversations:
            for turn in conv.turns:
                assert turn.max_tokens is not None
                assert turn.max_tokens >= 1


class TestThinkMsCrossCheck:
    """``_build_turn``'s "prev" cross-check and ``_check_spawn_think_ms``'s
    "spawn" cross-check are validation-only (see _weka_replay_timing.
    think_ms_disagrees): they never change a Turn's actual delay/timestamp,
    only count agreement/disagreement against the capture's own recorded
    think_ms for the same reference.
    """

    def _normalized_session(self, instances: list[dict]):
        session = build_session(session_id="sessA", instances=instances)
        raw = ReplaySession.model_validate(session)
        diag = SessionDiagnostics()
        return normalize_session(raw, schema_version="replay-v3", lineno=2, diag=diag)

    def test_prev_agreement_not_counted_as_disagreement(self, tmp_path: Path):
        loader = _build_loader(tmp_path)
        instance = build_instance(
            requests=[
                build_request(request_id=1, ts=_ts(0), total_ms=500),
                build_request(
                    request_id=2,
                    ts=_ts(2000),
                    total_ms=500,
                    # raw_delay_ms = 2000 - (0 + 500) = 1500, matches exactly.
                    think_ms=1500,
                    think_ref="prev",
                ),
            ]
        )
        session_ir = self._normalized_session([instance])
        requests = session_ir.instances[0].requests
        diag = SessionDiagnostics()
        loader._build_turn(
            request=requests[1],
            prev_request=requests[0],
            turn_content=_EMPTY_TURN_CONTENT,
            session_id="sessA",
            turn_idx=1,
            role=session_ir.instances[0].role,
            diag=diag,
        )
        assert diag.think_prev_compared_count == 1
        assert diag.think_prev_disagreement_count == 0

    def test_prev_disagreement_is_counted(self, tmp_path: Path):
        loader = _build_loader(tmp_path)
        instance = build_instance(
            requests=[
                build_request(request_id=1, ts=_ts(0), total_ms=500),
                build_request(
                    request_id=2,
                    ts=_ts(2000),
                    total_ms=500,
                    # raw_delay_ms = 1500, but recorded think_ms says 10000.
                    think_ms=10_000,
                    think_ref="prev",
                ),
            ]
        )
        session_ir = self._normalized_session([instance])
        requests = session_ir.instances[0].requests
        diag = SessionDiagnostics()
        loader._build_turn(
            request=requests[1],
            prev_request=requests[0],
            turn_content=_EMPTY_TURN_CONTENT,
            session_id="sessA",
            turn_idx=1,
            role=session_ir.instances[0].role,
            diag=diag,
        )
        assert diag.think_prev_compared_count == 1
        assert diag.think_prev_disagreement_count == 1

    def test_spawn_agreement_not_counted_as_disagreement(self, tmp_path: Path):
        loader = _build_loader(tmp_path)
        parent = build_instance(
            instance_id="root:main",
            role="main",
            requests=[build_request(request_id=1, ts=_ts(0), total_ms=500)],
        )
        child = build_instance(
            instance_id="root:sync",
            role="sub-agent",
            fan_out_group="1",
            parent_spawn_request_id=1,
            requests=[
                build_request(
                    request_id=11,
                    ts=_ts(600),
                    total_ms=100,
                    # branch starts at parent_end (parent ts + 500ms); this
                    # child starts 100ms after that, matching think_ms exactly.
                    think_ms=100,
                    think_ref="spawn",
                )
            ],
        )
        session_ir = self._normalized_session([parent, child])
        instances_by_id = {inst.instance_id: inst for inst in session_ir.instances}
        parent_request = instances_by_id["root:main"].requests[0]
        child_ts_ms = instances_by_id["root:sync"].requests[0].ts_ms
        branch_start_ms = compute_branch_start_ms(
            parent_request.ts_ms, parent_request.api_time_ms, [child_ts_ms]
        )
        diag = SessionDiagnostics()
        loader._check_spawn_think_ms(
            branch_start_ms=branch_start_ms,
            child_cids=("root:sync",),
            instances_by_id=instances_by_id,
            diag=diag,
        )
        assert diag.think_spawn_compared_count == 1
        assert diag.think_spawn_disagreement_count == 0

    def test_spawn_disagreement_is_counted(self, tmp_path: Path):
        loader = _build_loader(tmp_path)
        parent = build_instance(
            instance_id="root:main",
            role="main",
            requests=[build_request(request_id=1, ts=_ts(0), total_ms=500)],
        )
        child = build_instance(
            instance_id="root:sync",
            role="sub-agent",
            fan_out_group="1",
            parent_spawn_request_id=1,
            requests=[
                build_request(
                    request_id=11,
                    ts=_ts(600),
                    total_ms=100,
                    # computed gap is 100ms, but recorded think_ms says 50000.
                    think_ms=50_000,
                    think_ref="spawn",
                )
            ],
        )
        session_ir = self._normalized_session([parent, child])
        instances_by_id = {inst.instance_id: inst for inst in session_ir.instances}
        parent_request = instances_by_id["root:main"].requests[0]
        child_ts_ms = instances_by_id["root:sync"].requests[0].ts_ms
        branch_start_ms = compute_branch_start_ms(
            parent_request.ts_ms, parent_request.api_time_ms, [child_ts_ms]
        )
        diag = SessionDiagnostics()
        loader._check_spawn_think_ms(
            branch_start_ms=branch_start_ms,
            child_cids=("root:sync",),
            instances_by_id=instances_by_id,
            diag=diag,
        )
        assert diag.think_spawn_compared_count == 1
        assert diag.think_spawn_disagreement_count == 1

    def test_session_start_ref_never_compared(self, tmp_path: Path):
        """think_ref == "session_start" is deliberately never cross-checked
        (see weka_replay.py module docs): replaying "waited 3 hours before
        typing" as a literal startup pause would be wrong for a compressed
        benchmark run, so this reference is simply ignored."""
        loader = _build_loader(tmp_path)
        instance = build_instance(
            requests=[
                build_request(request_id=1, ts=_ts(0), total_ms=500),
                build_request(
                    request_id=2,
                    ts=_ts(2000),
                    total_ms=500,
                    think_ms=999_999,
                    think_ref="session_start",
                ),
            ]
        )
        session_ir = self._normalized_session([instance])
        requests = session_ir.instances[0].requests
        diag = SessionDiagnostics()
        loader._build_turn(
            request=requests[1],
            prev_request=requests[0],
            turn_content=_EMPTY_TURN_CONTENT,
            session_id="sessA",
            turn_idx=1,
            role=session_ir.instances[0].role,
            diag=diag,
        )
        assert diag.think_prev_compared_count == 0
        assert diag.think_prev_disagreement_count == 0


class TestOrphanBucketSkippedBeforeParsing:
    """Regression test for a real production bug found on a live benchmark
    pod: the shared orphan-bucket pseudo-session (``orphan:no-session-id``)
    crams together many unrelated users' header-less requests with no real
    structure, so it can violate invariants ``normalize_session`` enforces
    for genuine sessions (e.g. a ``request_id`` reused across "instances"
    that's also referenced as a ``parent_spawn_request_id`` -- an
    unresolvable ambiguity for a real session, but just noise in a bucket
    with no real session boundaries). Since this session is always dropped
    by default (``include_orphans=False``), it must never get a chance to
    crash the whole load by being parsed/normalized in the first place.
    """

    @staticmethod
    def _ambiguous_orphan_session() -> dict:
        # request_id 99 appears as a real request on TWO different
        # "instances" (duplicate) AND is also referenced as a
        # parent_spawn_request_id by a third -- exactly the shape that made
        # normalize_session raise on real production data.
        return build_session(
            session_id="orphan:no-session-id",
            instances=[
                _simple_instance("orphan:a", "ephemeral (no system)", 99),
                _simple_instance("orphan:b", "ephemeral (no system)", 99),
                _simple_instance(
                    "orphan:c",
                    "sub-agent",
                    100,
                    parent_spawn_request_id=99,
                    fan_out_group="99",
                ),
            ],
        )

    def test_ambiguous_orphan_session_crashes_normalize_session_directly(self):
        """Confirms the premise: parsing this exact session shape directly
        (bypassing the loader's pre-filter) really does raise, so skipping it
        is doing real work, not guarding against a hypothetical."""
        raw = ReplaySession.model_validate(self._ambiguous_orphan_session())
        with pytest.raises(Exception, match="ambiguous"):
            normalize_session(
                raw, schema_version="replay-v3", lineno=2, diag=SessionDiagnostics()
            )

    def test_orphan_bucket_excluded_by_default_does_not_crash(self, tmp_path: Path):
        good_session = _build_session_dict()  # valid "sessA"
        loader = _build_loader_from_sessions(
            tmp_path, [good_session, self._ambiguous_orphan_session()]
        )
        data = loader.load_dataset()
        assert len(data["sessions"]) == 2  # both spans scanned...
        conversations = loader.convert_to_conversations(data)  # ...but must not raise
        ids = {c.session_id for c in conversations}
        assert ids == {"root:main", "root:sync", "root:sync:grand"}

    def test_orphan_bucket_never_parsed_when_excluded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        good_session = _build_session_dict()
        loader = _build_loader_from_sessions(
            tmp_path, [good_session, self._ambiguous_orphan_session()]
        )
        data = loader.load_dataset()

        seen_session_ids: list[str] = []
        real_read_session = weka_replay.read_session

        def _spy_read_session(filename, span):
            seen_session_ids.append(span.session_id)
            return real_read_session(filename, span)

        monkeypatch.setattr(weka_replay, "read_session", _spy_read_session)
        loader.convert_to_conversations(data)
        assert seen_session_ids == ["sessA"]  # orphan span never read at all

    def test_orphan_bucket_included_when_opted_in_still_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Opting in to orphans re-exposes the real ambiguity -- this is
        intentional (orphans are documented as structurally messy by
        nature), not something this fix silently papers over."""
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.DATASET, "WEKA_REPLAY_INCLUDE_ORPHANS", True)
        good_session = _build_session_dict()
        loader = _build_loader_from_sessions(
            tmp_path, [good_session, self._ambiguous_orphan_session()]
        )
        data = loader.load_dataset()
        with pytest.raises(Exception, match="ambiguous"):
            loader.convert_to_conversations(data)


class TestNumDatasetEntriesCapsSpansBeforeParsing:
    """Regression test for a real production bug found on a live benchmark
    pod: the loader had no entries-capping at all, so ``--num-dataset-entries
    32`` against a 5,441-session / 9.5 GB real corpus silently processed the
    entire file instead of the requested 32 -- the exact O(corpus) blowup the
    loader's whole streaming design exists to avoid.
    """

    @staticmethod
    def _sessions(n: int) -> list[dict]:
        return [
            build_session(
                session_id=f"sess{i}",
                instances=[_simple_instance(f"sess{i}:main", "main", 1)],
            )
            for i in range(n)
        ]

    def test_caps_to_first_n_sessions(self, tmp_path: Path):
        loader = _build_loader_from_sessions(
            tmp_path, self._sessions(5), conversation_num_dataset_entries=2
        )
        data = loader.load_dataset()
        assert len(data["sessions"]) == 5  # load_dataset itself stays cheap/uncapped
        conversations = loader.convert_to_conversations(data)
        ids = {c.session_id for c in conversations}
        assert ids == {"sess0:main", "sess1:main"}

    def test_sessions_beyond_cap_never_parsed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        loader = _build_loader_from_sessions(
            tmp_path, self._sessions(5), conversation_num_dataset_entries=2
        )
        data = loader.load_dataset()

        seen_session_ids: list[str] = []
        real_read_session = weka_replay.read_session

        def _spy_read_session(filename, span):
            seen_session_ids.append(span.session_id)
            return real_read_session(filename, span)

        monkeypatch.setattr(weka_replay, "read_session", _spy_read_session)
        loader.convert_to_conversations(data)
        assert seen_session_ids == ["sess0", "sess1"]

    def test_no_cap_when_entries_not_set(self, tmp_path: Path):
        loader = _build_loader_from_sessions(tmp_path, self._sessions(5))
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert {c.session_id for c in conversations} == {
            f"sess{i}:main" for i in range(5)
        }

    def test_entries_larger_than_corpus_is_a_no_op(self, tmp_path: Path):
        loader = _build_loader_from_sessions(
            tmp_path, self._sessions(3), conversation_num_dataset_entries=10
        )
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 3
