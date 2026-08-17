"""M0 exit criterion: one synthetic log flows end to end through a stub
monitor, deterministically."""

from __future__ import annotations

from sentinel.monitors.stub import StubMonitor
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader
from sentinel.pipeline.stream import health_stream
from sentinel.schemas.monitor_result import Verdict


def test_synthetic_stream_end_to_end() -> None:
    reader = SyntheticReader(ScenarioSpec(duration_s=2.0), seed=42)
    monitor = StubMonitor()
    frames = list(health_stream(reader))
    assert len(frames) == 40  # 2s @ 20Hz
    non_gap = [f for f in frames if not f.is_gap]
    assert len(non_gap) > 0
    for frame in non_gap:
        result = monitor.update(frame)
        assert result.verdict == Verdict.PASS
        assert 0.0 <= result.score <= 1.0


def test_synthetic_stream_is_deterministic() -> None:
    spec = ScenarioSpec(duration_s=1.0)
    r1 = list(health_stream(SyntheticReader(spec, seed=7)))
    r2 = list(health_stream(SyntheticReader(spec, seed=7)))
    assert [f.model_dump_json() for f in r1] == [f.model_dump_json() for f in r2]


def test_synthetic_stream_different_seed_differs() -> None:
    spec = ScenarioSpec(duration_s=1.0)
    r1 = list(health_stream(SyntheticReader(spec, seed=1)))
    r2 = list(health_stream(SyntheticReader(spec, seed=2)))
    assert [f.model_dump_json() for f in r1] != [f.model_dump_json() for f in r2]


def test_gap_frames_marked_not_interpolated() -> None:
    spec = ScenarioSpec(duration_s=5.0, drop_probability=0.3)
    frames = list(health_stream(SyntheticReader(spec, seed=3)))
    gaps = [f for f in frames if f.is_gap]
    assert len(gaps) > 0
    for g in gaps:
        assert g.gap_reason == "dropped"
        assert g.perception is None  # no fabricated values
