"""Deterministic replay: re-run the recorded inputs from a log through a
freshly-constructed decision loop and assert bit-identical decisions.

Scope of the determinism guarantee (spec requires this be stated
explicitly, not implied): M2 (sensor agreement), M3 (timing), M4 (ODD
compliance), the trust coordinator, and the policy gate are replayed
FROM SOURCE — the same code re-processes the same recorded HealthFrame
sequence and must produce byte-identical MonitorResult/TrustState/
PolicyDecision output. M1, when running as `M1LearnedMonitor` (GPU
inference), is NOT re-inferred during replay by default — floating-point
GPU inference is not guaranteed bit-reproducible across runs/hardware in
general, so replay instead asserts that M1's *recorded* evidence
(score/verdict) is what downstream decisions were computed from, and does
not attempt to recompute it. `M1HeuristicMonitor`, being pure Python/NumPy
over recorded history, DOES replay bit-identically from source like the
other three, and the default replay path re-runs it from source as well.

A mismatch fails loudly with a field-level diff of the FIRST record where
recomputed output diverges from the recorded output — not just "replay
failed."
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.evidence.record import EvidenceRecord
from sentinel.evidence.writer import read_log
from sentinel.runtime.factory import RuntimeConfigPaths, build_ephemeral_decision_loop


class ReplayMismatchError(Exception):
    pass


@dataclass(frozen=True)
class FieldDiff:
    field_path: str
    original: str
    replayed: str


@dataclass(frozen=True)
class ReplayResult:
    total_records: int
    matched: bool
    first_divergence_event_id: int | None
    diffs: tuple[FieldDiff, ...]
    replayed_log_path: str


# Fields intentionally excluded from the bit-identical comparison because
# they are provenance-of-THIS-run metadata, not decision output:
#   - code_git_sha: replay may run on a different checkout than the
#     original recording (e.g. CI replaying a demo log); the DECISIONS
#     must match, not the SHA of the code that happened to produce them.
_EXCLUDED_TOP_LEVEL_FIELDS = {"code_git_sha", "wall_time_ns", "event_id", "prev_record_hash", "record_hash"}


def _record_diffs(original: EvidenceRecord, replayed: EvidenceRecord) -> list[FieldDiff]:
    orig_dict = original.model_dump(mode="json")
    replay_dict = replayed.model_dump(mode="json")
    diffs: list[FieldDiff] = []
    all_keys = sorted(set(orig_dict.keys()) | set(replay_dict.keys()))
    for key in all_keys:
        if key in _EXCLUDED_TOP_LEVEL_FIELDS:
            continue
        # input_frame is an INPUT, not decision output — it must match too
        # (it came from the same recorded stream) but we compare it as a
        # sanity check on log integrity, not as a "decision".
        if orig_dict.get(key) != replay_dict.get(key):
            diffs.append(
                FieldDiff(
                    field_path=key,
                    original=str(orig_dict.get(key))[:500],
                    replayed=str(replay_dict.get(key))[:500],
                )
            )
    return diffs


def replay_log(
    log_path: str,
    paths: RuntimeConfigPaths = RuntimeConfigPaths(),  # noqa: B008 (frozen, immutable default)
    strict: bool = True,
) -> ReplayResult:
    original_records = read_log(log_path)
    loop, replayed_path = build_ephemeral_decision_loop(paths)

    replayed_records: list[EvidenceRecord] = []
    try:
        for original in original_records:
            replayed = loop.process_frame(original.input_frame)
            replayed_records.append(replayed)
    finally:
        loop.evidence_writer.close()

    first_divergence: int | None = None
    diffs: tuple[FieldDiff, ...] = ()
    for orig, replayed in zip(original_records, replayed_records, strict=True):
        record_diffs = _record_diffs(orig, replayed)
        if record_diffs:
            first_divergence = orig.event_id
            diffs = tuple(record_diffs)
            break

    result = ReplayResult(
        total_records=len(original_records),
        matched=first_divergence is None,
        first_divergence_event_id=first_divergence,
        diffs=diffs,
        replayed_log_path=replayed_path,
    )

    if strict and not result.matched:
        diff_lines = "\n".join(
            f"  {d.field_path}: original={d.original!r} replayed={d.replayed!r}" for d in diffs
        )
        raise ReplayMismatchError(
            f"Replay diverged at event_id={first_divergence}:\n{diff_lines}"
        )

    return result
