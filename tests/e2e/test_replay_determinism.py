"""End-to-end replay-determinism test: run a synthetic scenario through the
full decision loop, write an evidence log, then replay it and assert the
recomputed decisions are bit-identical to what was recorded — the M0-M5
exit criterion for the evidence/replay engine."""

from __future__ import annotations

import shutil
from pathlib import Path

from sentinel.evidence.hash_chain import verify_chain
from sentinel.evidence.replay import replay_log
from sentinel.evidence.writer import read_log
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader
from sentinel.pipeline.stream import health_stream
from sentinel.runtime.factory import build_decision_loop

_TEST_LOG_DIR = Path("evidence_logs/test_e2e")


def _record_scenario(duration_s: float, seed: int, name: str) -> str:
    _TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(_TEST_LOG_DIR / f"{name}.jsonl")
    if Path(log_path).exists():
        Path(log_path).unlink()
    loop = build_decision_loop(log_path)
    reader = SyntheticReader(ScenarioSpec(duration_s=duration_s), seed=seed)
    for frame in health_stream(reader):
        loop.process_frame(frame)
    loop.evidence_writer.close()
    return log_path


def setup_module() -> None:
    shutil.rmtree(_TEST_LOG_DIR, ignore_errors=True)


def test_hash_chain_is_valid_after_recording() -> None:
    log_path = _record_scenario(30.0, seed=42, name="hash_chain")
    records = read_log(log_path)
    assert len(records) > 0
    verify_chain(records)  # must not raise


def test_replay_produces_bit_identical_decisions() -> None:
    log_path = _record_scenario(30.0, seed=42, name="replay_basic")
    result = replay_log(log_path)
    assert result.matched, f"replay diverged: {result.diffs}"
    assert result.total_records > 0

    replayed_records = read_log(result.replayed_log_path)
    verify_chain(replayed_records)  # the replayed log is itself a valid chain


def test_replay_with_transitions_still_matches() -> None:
    """A scenario with an injected sensor-agreement fault (forces at least
    one trust transition) must still replay bit-identically — this is the
    harder case than all-nominal-PASS."""
    log_path = str(_TEST_LOG_DIR / "with_fault.jsonl")
    if Path(log_path).exists():
        Path(log_path).unlink()
    loop = build_decision_loop(log_path)
    reader = SyntheticReader(ScenarioSpec(duration_s=20.0), seed=7)
    for i, frame in enumerate(health_stream(reader)):
        if 100 <= i < 140:
            # Inject a gross GNSS/CAN disagreement to force a FAIL verdict
            # and a trust transition, deterministically (same seed, same
            # injected window every time this test runs).
            frame = frame.model_copy(
                update={"gnss": frame.gnss.model_copy(update={"speed_mps": 99.0})}
            )
        loop.process_frame(frame)
    loop.evidence_writer.close()

    records = read_log(log_path)
    assert any(r.trust_transition is not None for r in records), "test setup should force a transition"

    result = replay_log(log_path)
    assert result.matched, f"replay diverged: {result.diffs}"
