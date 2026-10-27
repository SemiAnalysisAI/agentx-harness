# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``WekaReplayLoader``: native AIPerf loader for ``wekai`` router replay
captures (``replay-v3``/``v4``/``v5``).

Wires together every independently-tested helper module built for this
loader (``weka_replay_models``, ``_weka_replay_reader``, ``_weka_replay_ir``,
``_weka_replay_tree``, ``_weka_replay_timing``, ``_weka_replay_identity``,
``_weka_replay_synth``, ``_weka_replay_options``) into the
``CustomDatasetLoaderProtocol`` the composer/plugin registry expects. See
the approved design plan (``weka_replay`` loader plan) sections 4.5, 5.5, 7,
and 8 for the authoritative spec this implements.

Each session's parse -> normalize -> plan -> synthesize -> emit pipeline runs
one session at a time in :meth:`WekaReplayLoader.convert_to_conversations`;
no reference to a prior session's intermediate state (raw/IR/plan/content
builder) is kept once that session's ``Conversation``s are built, which
bounds this loader's own peak working set to O(one session) (see plan §6).
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    PrerequisiteKind,
)
from aiperf.common.models import Conversation, DatasetMetadata, Turn, TurnPrerequisite
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader._weka_replay_ir import (
    InstanceIR,
    ReplayRole,
    RequestIR,
    SamplingIR,
    SessionDiagnostics,
    classify_orphan_shape,
    normalize_session,
)
from aiperf.dataset.loader._weka_replay_options import (
    WekaReplayOptions,
    resolve_options,
)
from aiperf.dataset.loader._weka_replay_reader import (
    SessionSpan,
    WekaReplayLoadError,
    peek_header_schema,
    read_header,
    read_session,
    reject_inline_records,
    scan_session_spans,
)
from aiperf.dataset.loader._weka_replay_synth import (
    ReplayContentBuilderImpl,
    ReplaySynthConfig,
)
from aiperf.dataset.loader._weka_replay_timing import (
    compute_delay_ms,
    think_ms_disagrees,
)
from aiperf.dataset.loader._weka_replay_tree import TreeFilterOptions, plan_session_tree
from aiperf.dataset.loader.base_loader import BaseFileLoader, LoaderProbeData
from aiperf.plugin.enums import DatasetSamplingStrategy, PhaseType, TimingMode

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.dataset.loader._weka_replay_content import TurnContent
    from aiperf.dataset.loader.weka_replay_models import ReplayHeader

__all__ = ["WekaReplayLoader"]

# Matches any replay-vN header broadly (can_load's job: "could plausibly be a
# replay capture"); load_dataset below narrows to the schema versions this
# loader actually knows how to normalize. Deliberately does NOT match
# "replay-index-v1" (the v5 trailing-index line's own _schema value) since
# that string has no trailing digit run immediately after "replay-v".
_SCHEMA_RE = re.compile(r"^replay-v\d+$")

_SUPPORTED_SCHEMA_VERSIONS = frozenset({"replay-v3", "replay-v4", "replay-v5"})

# Cadence for the mid-load progress line in convert_to_conversations: wall
# time, not a session count, so it fires at a predictable rate regardless of
# how heavy individual sessions are (a few huge sessions vs. many small ones).
_PROGRESS_LOG_INTERVAL_S = 30.0


@dataclass(slots=True)
class _PhaseTimings:
    """Cumulative wall-time (seconds) spent in each major phase of
    :meth:`WekaReplayLoader.convert_to_conversations`'s per-session loop,
    across the whole load.

    Measured once per session (or once per instance/conversation for the
    phases that run at that granularity) with ``time.monotonic()`` -- never
    per block or per request -- so the timing overhead itself stays
    negligible next to the work being measured. Folded into the final
    corpus summary log line so a slow load can be attributed to a phase
    instead of guessed at.
    """

    read_s: float = 0.0
    """Time spent in ``read_session`` (reading + parsing one session's raw JSONL)."""

    normalize_s: float = 0.0
    """Time spent in ``normalize_session`` (building the session IR)."""

    plan_s: float = 0.0
    """Time spent in ``plan_session_tree`` (conversation/branch planning)."""

    synth_s: float = 0.0
    """Time spent in content synthesis: ``build_instance_turns`` plus the
    session-wide theoretical cache-hit ``finish()`` call."""

    turn_build_s: float = 0.0
    """Time spent in ``_build_turn`` across every request of every instance."""

    def total(self) -> float:
        return (
            self.read_s
            + self.normalize_s
            + self.plan_s
            + self.synth_s
            + self.turn_build_s
        )


def _phase_resolves_to_fixed_schedule(phase: Any) -> bool:
    """Mirror ``aiperf.timing.config._phase_timing_mode``'s FIXED_SCHEDULE
    detection without importing that module-private helper: an explicit
    ``timing_mode`` on the phase wins, otherwise fall back to the phase type.
    """
    explicit = getattr(phase, "timing_mode", None)
    if explicit is not None:
        return explicit == TimingMode.FIXED_SCHEDULE
    return getattr(phase, "type", None) == PhaseType.FIXED_SCHEDULE


class _PromptGeneratorCorpusAdapter:
    """Adapts a real ``PromptGenerator``'s pre-tokenized corpus to the
    ``TokenCorpusSource`` protocol ``_weka_replay_synth.py`` expects.

    Deliberately never touches ``PromptGenerator._cache``, ``.generate()``,
    or ``.set_trace_id()`` -- those are ``weka_trace``-specific and would
    conflict with this loader's own per-block hash identity scheme (plan
    §5.3).
    """

    def __init__(self, prompt_generator: PromptGenerator) -> None:
        self._pg = prompt_generator

    def token_at(self, index: int) -> int:
        return self._pg.tokenized_corpus[index]

    @property
    def size(self) -> int:
        return len(self._pg.tokenized_corpus)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._pg.tokenizer.decode(list(token_ids))


class _CorpusDiagnostics:
    """Accumulates every int counter on :class:`SessionDiagnostics` across
    every session processed by one loader run, for the single end-of-load
    summary log line.

    Generic over ``SessionDiagnostics``'s field list (rather than naming each
    counter here) so a future counter added there is picked up automatically.
    The ``*_examples``/``anomalous_links`` list fields are informational
    spot-check aids only and are not summed.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.sessions_seen = 0
        self.sessions_kept = 0

    def accumulate(self, diag: SessionDiagnostics) -> None:
        self.sessions_seen += 1
        for f in fields(diag):
            if f.name.endswith("_examples") or f.name == "anomalous_links":
                continue
            value = getattr(diag, f.name)
            if isinstance(value, int):
                self.counts[f.name] = self.counts.get(f.name, 0) + value


class WekaReplayLoader(BaseFileLoader):
    """Native loader for ``wekai`` router ``replay-v3``/``v4``/``v5`` captures.

    Preserves the recorded multi-level sub-agent tree (``fan_out_group``
    causal links) as SPAWN branches via the generic branch-orchestration
    mechanism ``dag_jsonl`` also uses, absolute request timestamps,
    server-reported token/cache counts, and real sampling parameters.

    Usage (composer-driven, the only supported path in v1)::

        aiperf profile --custom-dataset-type weka_replay --input-file replay.jsonl ...

    The composer injects ``prompt_generator``/``default_block_size`` because
    this loader is registered with ``is_trace: true`` in ``plugins.yaml``.
    """

    def __init__(
        self,
        *,
        filename: str | Path | None = None,
        inline_records: Any | None = None,
        run: BenchmarkRun | None = None,
        prompt_generator: PromptGenerator | None = None,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        reject_inline_records(inline_records)
        super().__init__(filename=filename, run=run, **kwargs)
        # Inlined rather than importing _weka_replay_reader._reject_directory
        # (module-private, underscore-prefixed): this check needs to fire at
        # construction time, before any file is opened, so the composer gets
        # a clean error immediately rather than on the first read_header call.
        if self.filename is not None and self.filename.is_dir():
            raise WekaReplayLoadError(
                f"weka_replay '{self.filename}': expected a single .jsonl file, "
                "got a directory; weka_replay does not support directory input "
                "in v1"
            )
        if prompt_generator is None:
            raise WekaReplayLoadError(
                "weka_replay requires a prompt_generator for content synthesis "
                "(the 'is_trace: true' plugin metadata should guarantee the "
                "composer injects one); got None. If constructing this loader "
                "directly (tests, offline tooling), pass a TokenCorpusSource-"
                "compatible prompt_generator."
            )
        self._prompt_generator = prompt_generator
        self._default_block_size = default_block_size
        self._options: WekaReplayOptions = resolve_options(self.run)
        self._corpus_adapter = _PromptGeneratorCorpusAdapter(prompt_generator)
        self._content_builder_factory = ReplayContentBuilderImpl(
            self._corpus_adapter, cfg=ReplaySynthConfig()
        )

        dataset = self.run.cfg.get_default_dataset()
        self._ignore_trace_delays: bool = getattr(dataset, "ignore_trace_delays", False)
        self._delay_cap_tracker = DelayCapTracker(
            cap_seconds=getattr(dataset, "inter_turn_delay_cap_seconds", None)
        )
        self._configured_model_names = list(self.run.cfg.get_model_names())
        self._model_map: dict[str, str] = {}
        self._warned_unknown_roles: set[str] = set()
        self._warned_v3_timing = False
        self._header: ReplayHeader | None = None

        # Plan §4.5 / §8: FIXED_SCHEDULE has no is_root filter and would
        # double-dispatch every SPAWN child. ``run.cfg.phases`` is already
        # inspectable at construction time (weka_trace.py reads the same
        # attribute in its own __init__ for the fixed-schedule window), so
        # the guard fires here rather than being deferred to
        # convert_to_conversations.
        for phase in self.run.cfg.phases:
            if _phase_resolves_to_fixed_schedule(phase):
                raise NotImplementedError(
                    "weka_replay: FIXED_SCHEDULE timing is not supported - "
                    "FixedScheduleStrategy has no is_root filter and would "
                    "double-dispatch every SPAWN child; use a YAML phase "
                    "config with timing_mode: agentic_replay, or plain "
                    "concurrency mode"
                )

    # --- plugin-facing API -----------------------------------------------

    @classmethod
    def can_load(
        cls,
        data: LoaderProbeData | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Broad match on a replay header's ``_schema`` field (``^replay-v\\d+$``).

        ``data``, when given, is the raw dict probed from the first JSONL
        line (before any pydantic alias resolution -- so the key is the
        on-disk ``_schema``, not a resolved ``schema_version``). When
        ``data`` is ``None`` and ``filename`` names a regular file,
        :func:`peek_header_schema` reads just its first line. Never raises:
        any failure collapses to ``False``.
        """
        try:
            if isinstance(data, dict):
                schema = data.get("_schema")
                return isinstance(schema, str) and _SCHEMA_RE.match(schema) is not None
            if data is None and filename is not None:
                path = Path(filename) if isinstance(filename, str) else filename
                if not path.is_file():
                    return False
                schema = peek_header_schema(path)
                return isinstance(schema, str) and _SCHEMA_RE.match(schema) is not None
            return False
        except Exception:
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode:
        return ConversationContextMode.DELTAS_WITH_RESPONSES

    def load_dataset(self) -> dict[str, list[SessionSpan]]:
        """Read the header, reject an unsupported schema, and build session spans.

        Returns ``{"sessions": [SessionSpan, ...]}`` -- cheap byte-location
        references, not parsed session content (plan §6's streaming
        contract: the real parse/plan/emit work happens in
        :meth:`convert_to_conversations`, one session at a time).

        Note on ``CustomDatasetT``: :data:`aiperf.dataset.loader.models.CustomDatasetT`
        is only bound to a union of pydantic models under ``TYPE_CHECKING``;
        at runtime the bound collapses to plain ``AIPerfBaseModel`` and
        nothing enforces it on an actual ``TypeVar``, so returning plain
        ``SessionSpan`` dataclasses here (rather than wrapping each one in a
        throwaway pydantic model) is runtime-safe and matches the existing
        loader's own ``SessionSpan`` dataclass contract.
        """
        schema = peek_header_schema(self.filename)
        if schema is None or schema not in _SUPPORTED_SCHEMA_VERSIONS:
            raise WekaReplayLoadError(
                f"weka_replay '{self.filename}': unsupported or missing schema "
                f"version {schema!r}; weka_replay only supports "
                f"{sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
            )
        header = read_header(self.filename)
        self._header = header
        spans, _index = scan_session_spans(self.filename, header)
        return {"sessions": spans}

    def convert_to_conversations(
        self, custom_data: dict[str, list[SessionSpan]]
    ) -> list[Conversation]:
        if self._header is None:
            # Defensive: the plugin contract always calls load_dataset()
            # first, but a direct/offline caller might not.
            self.load_dataset()
        assert self._header is not None
        header = self._header

        spans = self._cap_spans(custom_data["sessions"])
        tree_options = self._tree_filter_options()
        corpus_diag = _CorpusDiagnostics()
        depth_histogram: Counter[int] = Counter()
        theoretical_hit_tokens = 0
        recorded_cache_read_tokens = 0
        all_conversations: list[Conversation] = []
        # Churn-heuristic validation (plan 5.1: "pending one real-corpus
        # validation run" for the cache_control-fragment-size assumption) --
        # never wired into any loader-visible logging until now.
        corpus_minted_blocks = 0
        corpus_reused_blocks = 0
        corpus_flip_hist: Counter[int] = Counter()
        corpus_tight_loose_disagreement = 0
        corpus_resolution_counts: Counter[str] = Counter()

        phase_timings = _PhaseTimings()
        total_spans = len(spans)
        load_start = time.monotonic()
        last_progress_log = load_start

        for i, span in enumerate(spans, 1):
            try:
                # Skip orphan-shaped sessions BEFORE parsing/normalizing them,
                # not just at plan_session_tree's filtering step: the shared
                # orphan bucket crams together many unrelated users' header-less
                # requests with no real structure, so it can violate invariants
                # normalize_session enforces for genuine sessions (e.g. a
                # request_id reused across "instances" that's also referenced as
                # a parent_spawn_request_id, which is a hard, unrecoverable
                # ambiguity for a real session but just noise here) -- a session
                # we're always going to drop should never get a chance to crash
                # the whole load first. session_id is already known from the
                # scan phase, so this needs no parse at all.
                if (
                    classify_orphan_shape(span.session_id) != "none"
                    and not self._options.include_orphans
                ):
                    skip_diag = SessionDiagnostics()
                    skip_diag.orphan_session_dropped += 1
                    corpus_diag.accumulate(skip_diag)
                    continue

                t0 = time.monotonic()
                raw = read_session(self.filename, span)
                phase_timings.read_s += time.monotonic() - t0

                diag = SessionDiagnostics()

                t0 = time.monotonic()
                session_ir = normalize_session(
                    raw,
                    schema_version=header.schema_version,
                    lineno=span.lineno,
                    diag=diag,
                )
                phase_timings.normalize_s += time.monotonic() - t0

                t0 = time.monotonic()
                plan = plan_session_tree(session_ir, options=tree_options, diag=diag)
                phase_timings.plan_s += time.monotonic() - t0
                if plan is None:
                    corpus_diag.accumulate(diag)
                    continue

                content_builder = self._content_builder_factory.for_session(session_ir)
                session_conversations: list[Conversation] = []
                # (Turn, instance_id, request_id) for every turn built this
                # session, consumed after content_builder.finish() resolves the
                # session-wide theoretical cache-hit metric.
                cache_targets: list[tuple[Turn, str, int]] = []
                # For the think_ms cross-check on SPAWN branches below: a
                # branch's child_cids are instance_ids (plan §... PlannedConversation
                # docstring: "cid == instance_id verbatim").
                instances_by_id = {
                    inst.instance_id: inst for inst in session_ir.instances
                }

                for conv in plan.conversations:
                    t0 = time.monotonic()
                    turn_contents = content_builder.build_instance_turns(conv.instance)
                    phase_timings.synth_s += time.monotonic() - t0
                    if len(turn_contents) != len(conv.instance.requests):
                        raise RuntimeError(
                            f"weka_replay: internal error: build_instance_turns "
                            f"returned {len(turn_contents)} turn(s) for instance "
                            f"'{conv.cid}' with {len(conv.instance.requests)} "
                            "request(s); these must be 1:1"
                        )

                    turns: list[Turn] = []
                    prev_request: RequestIR | None = None
                    t0 = time.monotonic()
                    for turn_idx, (request, turn_content) in enumerate(
                        zip(conv.instance.requests, turn_contents, strict=True)
                    ):
                        turn = self._build_turn(
                            request=request,
                            prev_request=prev_request,
                            turn_content=turn_content,
                            session_id=session_ir.session_id,
                            turn_idx=turn_idx,
                            role=conv.instance.role,
                            diag=diag,
                        )
                        turns.append(turn)
                        cache_targets.append(
                            (turn, conv.instance.instance_id, request.request_id)
                        )
                        recorded_cache_read_tokens += request.cache_read_tokens
                        prev_request = request
                    phase_timings.turn_build_s += time.monotonic() - t0

                    conv_obj = Conversation(
                        session_id=conv.cid,
                        context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
                        turns=turns,
                        is_root=conv.is_root,
                        agent_depth=conv.agent_depth,
                        parent_conversation_id=conv.parent_cid,
                        replay_scope_id=conv.cid,
                        branches=[],
                    )
                    for branch in conv.branches:
                        conv_obj.branches.append(
                            ConversationBranchInfo(
                                branch_id=branch.branch_id,
                                child_conversation_ids=list(branch.child_cids),
                                mode=ConversationBranchMode.SPAWN,
                                is_background=branch.is_background,
                                start_timestamp_ms=branch.start_timestamp_ms,
                            )
                        )
                        conv_obj.turns[branch.declaring_turn].branch_ids.append(
                            branch.branch_id
                        )
                        if branch.join_turn is not None:
                            conv_obj.turns[branch.join_turn].prerequisites.append(
                                TurnPrerequisite(
                                    kind=PrerequisiteKind.SPAWN_JOIN,
                                    branch_id=branch.branch_id,
                                )
                            )
                        self._check_spawn_think_ms(
                            branch_start_ms=branch.start_timestamp_ms,
                            child_cids=branch.child_cids,
                            instances_by_id=instances_by_id,
                            diag=diag,
                        )

                    session_conversations.append(conv_obj)
                    depth_histogram[conv.agent_depth] += 1

                t0 = time.monotonic()
                cache_map = content_builder.finish()
                phase_timings.synth_s += time.monotonic() - t0
                for turn, instance_id, request_id in cache_targets:
                    hit_total = cache_map.get((instance_id, request_id))
                    if hit_total is not None:
                        hit_blocks, total_blocks = hit_total
                        turn.theoretical_prefix_cache_hit_blocks = hit_blocks
                        turn.theoretical_prefix_cache_total_blocks = total_blocks
                        theoretical_hit_tokens += hit_blocks * 64

                synth_diag = content_builder.diagnostics
                corpus_minted_blocks += synth_diag.minted_blocks
                corpus_reused_blocks += synth_diag.reused_blocks
                churn_stats = content_builder.identity_stats
                corpus_resolution_counts["slot_same_hash"] += (
                    churn_stats.slot_same_hash_count
                )
                corpus_resolution_counts["exact_hash"] += churn_stats.exact_hash_count
                corpus_resolution_counts["seed_hash"] += churn_stats.seed_hash_count
                corpus_resolution_counts["churn"] += churn_stats.churn_count
                corpus_resolution_counts["minted"] += churn_stats.minted_count
                for delta, count in churn_stats.flip_byte_delta_hist.items():
                    corpus_flip_hist[delta] += count
                corpus_tight_loose_disagreement += (
                    churn_stats.tight_loose_disagreement_count
                )

                validate_for_orchestrator_v1(
                    DatasetMetadata(
                        conversations=[c.metadata() for c in session_conversations],
                        sampling_strategy=self.get_preferred_sampling_strategy(),
                    )
                )

                corpus_diag.sessions_kept += 1
                corpus_diag.accumulate(diag)
                all_conversations.extend(session_conversations)
                # Discard this session's intermediate state before the next
                # iteration (plan §6): session_ir/raw/plan/content_builder/
                # cache_targets all go out of scope here, holding nothing beyond
                # the emitted Conversation objects.
            finally:
                last_progress_log = self._maybe_log_progress(
                    processed=i,
                    total=total_spans,
                    load_start=load_start,
                    last_log=last_progress_log,
                )

        self._delay_cap_tracker.log_summary(logger_name=__name__)
        self._log_corpus_summary(
            corpus_diag=corpus_diag,
            depth_histogram=depth_histogram,
            conversations_built=len(all_conversations),
            theoretical_hit_tokens=theoretical_hit_tokens,
            recorded_cache_read_tokens=recorded_cache_read_tokens,
            corpus_minted_blocks=corpus_minted_blocks,
            corpus_reused_blocks=corpus_reused_blocks,
            corpus_resolution_counts=corpus_resolution_counts,
            corpus_flip_hist=corpus_flip_hist,
            corpus_tight_loose_disagreement=corpus_tight_loose_disagreement,
            phase_timings=phase_timings,
            total_elapsed_s=time.monotonic() - load_start,
        )
        return all_conversations

    # --- internals ---------------------------------------------------------

    def _cap_spans(self, spans: list[SessionSpan]) -> list[SessionSpan]:
        """Apply ``--num-dataset-entries`` at the span level, before any
        session is parsed.

        Unlike ``weka_trace``, which must fully parse every trace before it
        can even know enough to filter-then-cap (peak context length isn't
        knowable until parsed), a replay-v3/v5 ``SessionSpan`` already carries
        everything needed to cap by count alone: this keeps the first N spans
        in file order and never touches the rest, so an entries-capped run
        costs O(N), not O(corpus) -- the same no-work-on-discarded-sessions
        property the rest of this loader is built around (see module
        docstring). This can keep slightly fewer than N final conversations
        if some of the first N spans are later dropped by orphan/ephemeral/
        role/max-context-length filtering in ``plan_session_tree`` -- that's
        an accepted, documented trade-off for not having to parse ahead to
        know which spans will survive.
        """
        dataset = self.run.cfg.get_default_dataset()
        entries = getattr(dataset, "entries", None)
        entries_explicit = (
            dataset is not None
            and "entries" in dataset.model_fields_set
            and entries is not None
        )
        if not entries_explicit:
            return spans
        if len(spans) <= entries:
            return spans
        self.info(
            f"weka_replay: --num-dataset-entries={entries} < {len(spans)} "
            f"sessions in the file; capping to the first {entries} in file "
            "order before parsing any of them."
        )
        return spans[:entries]

    def _tree_filter_options(self) -> TreeFilterOptions:
        include_roles: frozenset[ReplayRole] | None = None
        if self._options.include_roles is not None:
            roles: set[ReplayRole] = set()
            for raw_role in self._options.include_roles:
                try:
                    roles.add(ReplayRole(raw_role))
                except ValueError:
                    if raw_role not in self._warned_unknown_roles:
                        self._warned_unknown_roles.add(raw_role)
                        self.warning(
                            f"weka_replay: unrecognized role {raw_role!r} in "
                            "AIPERF_DATASET_WEKA_REPLAY_INCLUDE_ROLES; ignoring "
                            "(it can never match an instance's role)"
                        )
            include_roles = frozenset(roles)

        return TreeFilterOptions(
            session_mode=self._options.session_mode,
            include_orphans=self._options.include_orphans,
            include_ephemeral=self._options.include_ephemeral,
            include_roles=include_roles,
            synthesis_max_isl=self._options.synthesis_max_isl,
            max_context_length=self._options.max_context_length,
            synthesis_max_osl=self._options.synthesis_max_osl,
        )

    def _map_model(self, model: str) -> str:
        """Map a recorded model name to ``run.cfg.get_model_names()`` by
        first-appearance order across the WHOLE load (not per-session), so a
        real Claude model name never reaches a vLLM/sglang endpoint. Built
        lazily/incrementally as new model strings are encountered; wraps
        (modulo) when distinct trace models exceed configured models.
        """
        configured = self._configured_model_names
        if not configured:
            return model
        mapped = self._model_map.get(model)
        if mapped is None:
            mapped = configured[len(self._model_map) % len(configured)]
            self._model_map[model] = mapped
        return mapped

    def _capped_max_tokens(self, output_tokens: int) -> int:
        """``max(1, output_tokens)``, capped by ``--synthesis-max-osl`` if
        set. Mirrors ``weka_trace.py``'s own ``_cap_output`` order (cap
        first, then upgrade a recorded 0 to 1) for consistency."""
        capped = output_tokens
        max_osl = self._options.synthesis_max_osl
        if max_osl is not None and capped > max_osl:
            capped = max_osl
        return capped if capped >= 1 else 1

    def _build_extra_body(self, sampling: SamplingIR) -> dict[str, Any] | None:
        passthrough = self._options.sampling_passthrough
        extra_body: dict[str, Any] = {}
        if "temperature" in passthrough and sampling.temperature is not None:
            extra_body["temperature"] = sampling.temperature
        if "top_p" in passthrough and sampling.top_p is not None:
            extra_body["top_p"] = sampling.top_p
        if "top_k" in passthrough and sampling.top_k is not None:
            extra_body["top_k"] = sampling.top_k
        if "thinking" in passthrough and sampling.thinking is not None:
            extra_body["thinking"] = sampling.thinking
        return extra_body or None

    def _build_turn(
        self,
        *,
        request: RequestIR,
        prev_request: RequestIR | None,
        turn_content: TurnContent,
        session_id: str,
        turn_idx: int,
        role: ReplayRole,
        diag: SessionDiagnostics,
    ) -> Turn:
        ignore_delays = self._ignore_trace_delays

        delay: float | None = None
        if turn_idx > 0 and prev_request is not None and not ignore_delays:
            if prev_request.api_time_ms is None and not self._warned_v3_timing:
                self._warned_v3_timing = True
                self.warning(
                    "weka_replay: inter-turn delay(s) computed from input with "
                    "no recorded request duration (replay-v3, or a replay-v4+ "
                    "request with total_ms<=0) include the previous request's "
                    "recorded server-side processing time, not just idle time. "
                    "Regenerate the capture as replay-v5 for exact inter-turn "
                    "timing."
                )
            raw_delay_ms = compute_delay_ms(
                prev_request.ts_ms, prev_request.api_time_ms, request.ts_ms
            )
            delay = self._delay_cap_tracker.clamp(raw_delay_ms)
            if request.think_ms is not None and request.think_ref == "prev":
                diag.think_prev_compared_count += 1
                if think_ms_disagrees(raw_delay_ms, request.think_ms):
                    diag.think_prev_disagreement_count += 1

        return Turn(
            model=self._map_model(request.model),
            timestamp=None if ignore_delays else request.ts_ms,
            delay=None if ignore_delays else delay,
            api_time_ms=None if ignore_delays else request.api_time_ms,
            max_tokens=self._capped_max_tokens(request.output_tokens),
            raw_messages=turn_content.raw_messages,
            raw_tools=turn_content.raw_tools,
            extra_body=self._build_extra_body(request.sampling),
            reset_context=turn_content.reset_context,
            source_trace_id=session_id,
            source_inner_idx=turn_idx,
            source_kind=f"weka_replay:{role}",
        )

    def _check_spawn_think_ms(
        self,
        *,
        branch_start_ms: float | None,
        child_cids: tuple[str, ...],
        instances_by_id: dict[str, InstanceIR],
        diag: SessionDiagnostics,
    ) -> None:
        """Cross-check this branch's computed start against each child's own
        recorded ``think_ms`` where ``think_ref == "spawn"`` -- validation
        only, mirrors the "prev" cross-check in :meth:`_build_turn`. A
        non-causal branch (co-session anchoring) never has a child whose
        first request carries ``think_ref == "spawn"``, so this is a no-op
        for those by construction.
        """
        if branch_start_ms is None:
            return
        for child_cid in child_cids:
            child = instances_by_id.get(child_cid)
            if child is None or not child.requests:
                continue
            first_request = child.requests[0]
            if first_request.think_ms is None or first_request.think_ref != "spawn":
                continue
            computed_ms = first_request.ts_ms - branch_start_ms
            diag.think_spawn_compared_count += 1
            if think_ms_disagrees(computed_ms, first_request.think_ms):
                diag.think_spawn_disagreement_count += 1

    def _maybe_log_progress(
        self,
        *,
        processed: int,
        total: int,
        load_start: float,
        last_log: float,
    ) -> float:
        """Log a mid-load progress line at most once every
        :data:`_PROGRESS_LOG_INTERVAL_S` of wall time, and return the
        (possibly updated) timestamp of the last log so the caller can carry
        it into the next iteration.

        Called from a ``finally`` block around every span of the main
        per-session loop (skipped, dropped, or fully converted alike) so
        progress always advances regardless of which exit path that
        iteration took.
        """
        now = time.monotonic()
        if now - last_log < _PROGRESS_LOG_INTERVAL_S and processed < total:
            return last_log

        elapsed = now - load_start
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = total - processed
        eta_s = remaining / rate if rate > 0 else None
        eta_str = f"{eta_s:.0f}s" if eta_s is not None else "unknown"
        self.info(
            lambda: (
                f"weka_replay: progress {processed}/{total} session(s) "
                f"({100.0 * processed / total:.1f}%) in {elapsed:.1f}s "
                f"({rate:.2f} sessions/s); ETA {eta_str}"
            )
        )
        return now

    def _log_corpus_summary(
        self,
        *,
        corpus_diag: _CorpusDiagnostics,
        depth_histogram: Counter[int],
        conversations_built: int,
        theoretical_hit_tokens: int,
        recorded_cache_read_tokens: int,
        corpus_minted_blocks: int,
        corpus_reused_blocks: int,
        corpus_resolution_counts: Counter[str],
        corpus_flip_hist: Counter[int],
        corpus_tight_loose_disagreement: int,
        phase_timings: _PhaseTimings,
        total_elapsed_s: float,
    ) -> None:
        self.info(
            f"weka_replay: loaded {corpus_diag.sessions_kept}/"
            f"{corpus_diag.sessions_seen} session(s) into {conversations_built} "
            f"conversation(s); agent_depth histogram: "
            f"{dict(sorted(depth_histogram.items()))}; diagnostics: "
            f"{corpus_diag.counts}; theoretical cache-hit tokens="
            f"{theoretical_hit_tokens:,} vs recorded cache_read_tokens="
            f"{recorded_cache_read_tokens:,}"
        )
        total_blocks = corpus_minted_blocks + corpus_reused_blocks
        minted_pct = (
            100.0 * corpus_minted_blocks / total_blocks if total_blocks else 0.0
        )
        top_deltas = sorted(
            corpus_flip_hist.items(), key=lambda kv: kv[1], reverse=True
        )[:15]
        self.info(
            "weka_replay content-identity: resolution counts="
            f"{dict(corpus_resolution_counts)}; minted={corpus_minted_blocks:,} "
            f"({minted_pct:.1f}%) reused={corpus_reused_blocks:,} of {total_blocks:,} "
            "blocks total; tight/loose churn-gate disagreements="
            f"{corpus_tight_loose_disagreement:,}; top cache_control flip Δbytes "
            f"(value: occurrence_count, including non-matching flips): {top_deltas}"
        )

        def _pct(phase_s: float) -> float:
            return 100.0 * phase_s / total_elapsed_s if total_elapsed_s > 0 else 0.0

        self.info(
            lambda: (
                f"weka_replay phase timing (total wall time {total_elapsed_s:.1f}s): "
                f"read={phase_timings.read_s:.1f}s ({_pct(phase_timings.read_s):.1f}%), "
                f"normalize={phase_timings.normalize_s:.1f}s "
                f"({_pct(phase_timings.normalize_s):.1f}%), "
                f"tree={phase_timings.plan_s:.1f}s ({_pct(phase_timings.plan_s):.1f}%), "
                f"synth={phase_timings.synth_s:.1f}s ({_pct(phase_timings.synth_s):.1f}%), "
                f"turn_build={phase_timings.turn_build_s:.1f}s "
                f"({_pct(phase_timings.turn_build_s):.1f}%), "
                f"unaccounted={max(0.0, total_elapsed_s - phase_timings.total()):.1f}s "
                f"({_pct(max(0.0, total_elapsed_s - phase_timings.total())):.1f}%)"
            )
        )
