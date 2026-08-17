from __future__ import annotations

from sentinel.config import load_yaml_with_hash
from sentinel.monitors.m1_perception import HeuristicThresholds, M1HeuristicMonitor
from sentinel.monitors.m2_sensor_agreement import (
    M2SensorAgreementMonitor,
    SensorAgreementThresholds,
)
from sentinel.monitors.m3_timing import M3TimingMonitor, TimingThresholds
from sentinel.monitors.m4_odd_compliance import M4OddComplianceMonitor, OddThresholds
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader
from sentinel.pipeline.stream import health_stream
from sentinel.policy.evidence_adapter import SceneTag, VLMEvidence
from sentinel.schemas.health_frame import RoadClass
from sentinel.schemas.monitor_result import Verdict


def _load_thresholds() -> dict:
    parsed, _ = load_yaml_with_hash("config/monitor_thresholds.yaml")
    return parsed


def test_m2_nominal_data_mostly_passes() -> None:
    thresholds = SensorAgreementThresholds(**_load_thresholds()["m2_sensor_agreement"])
    mon = M2SensorAgreementMonitor(thresholds)
    reader = SyntheticReader(ScenarioSpec(duration_s=60.0), seed=99)
    verdicts = []
    for frame in health_stream(reader):
        if frame.is_gap:
            continue
        verdicts.append(mon.update(frame).verdict)
    pass_rate = verdicts.count(Verdict.PASS) / len(verdicts)
    assert pass_rate > 0.9, f"expected >90% PASS on nominal data, got {pass_rate:.2%}"


def test_m2_flags_gross_sensor_disagreement() -> None:
    thresholds = SensorAgreementThresholds(**_load_thresholds()["m2_sensor_agreement"])
    mon = M2SensorAgreementMonitor(thresholds)
    reader = SyntheticReader(ScenarioSpec(duration_s=5.0), seed=1)
    frames = [f for f in health_stream(reader) if not f.is_gap]
    # Inject a gross CAN/GNSS disagreement on one frame.
    bad_frame = frames[10].model_copy(
        update={"gnss": frames[10].gnss.model_copy(update={"speed_mps": 40.0})}
    )
    for f in frames[:10]:
        mon.update(f)
    result = mon.update(bad_frame)
    assert result.verdict in (Verdict.WARN, Verdict.FAIL)


def test_m3_flags_deadline_miss() -> None:
    thresholds = TimingThresholds(**_load_thresholds()["m3_timing"])
    mon = M3TimingMonitor(thresholds)
    reader = SyntheticReader(ScenarioSpec(duration_s=2.0), seed=1)
    frames = [f for f in health_stream(reader) if not f.is_gap]
    slow_frame = frames[5].model_copy(
        update={"compute": frames[5].compute.model_copy(update={"e2e_latency_ms": 500.0})}
    )
    for f in frames[:5]:
        mon.update(f)
    result = mon.update(slow_frame)
    assert result.verdict == Verdict.FAIL
    assert "e2e_latency_ms" in result.evidence["reasons"]


def test_m3_flags_consecutive_drops() -> None:
    thresholds = TimingThresholds(**_load_thresholds()["m3_timing"])
    mon = M3TimingMonitor(thresholds)
    reader = SyntheticReader(ScenarioSpec(duration_s=5.0, drop_probability=0.0), seed=1)
    frames = list(health_stream(reader))[:20]
    last = None
    for f in frames[:5]:
        last = mon.update(f)
    gap = frames[5].model_copy(update={"is_gap": True})
    for _ in range(6):
        last = mon.update(gap)
    assert last.verdict == Verdict.FAIL


def test_m3_never_allocates_surprises_on_pass() -> None:
    thresholds = TimingThresholds(**_load_thresholds()["m3_timing"])
    mon = M3TimingMonitor(thresholds)
    reader = SyntheticReader(ScenarioSpec(duration_s=2.0), seed=1)
    for f in health_stream(reader):
        if f.is_gap:
            continue
        r = mon.update(f)
        assert r.verdict == Verdict.PASS
        assert r.score == 1.0


def test_m4_speed_violation_on_residential() -> None:
    thresholds = OddThresholds(**_load_thresholds()["m4_odd"])
    mon = M4OddComplianceMonitor(thresholds)
    reader = SyntheticReader(
        ScenarioSpec(duration_s=1.0, nominal_speed_mps=25.0, road_class=RoadClass.RESIDENTIAL),
        seed=1,
    )
    frames = [f for f in health_stream(reader) if not f.is_gap]
    results = [mon.update(f) for f in frames]
    assert any(r.verdict == Verdict.FAIL for r in results)


def test_m4_gnss_fix_invalid_fails() -> None:
    thresholds = OddThresholds(**_load_thresholds()["m4_odd"])
    mon = M4OddComplianceMonitor(thresholds)
    reader = SyntheticReader(ScenarioSpec(duration_s=1.0), seed=1)
    frame = next(f for f in health_stream(reader) if not f.is_gap)
    bad = frame.model_copy(update={"gnss": frame.gnss.model_copy(update={"fix_valid": False})})
    result = mon.update(bad)
    assert result.verdict == Verdict.FAIL
    assert result.evidence["localization_degraded"] is True


def test_m4_vlm_evidence_only_adds_never_removes_warning() -> None:
    """The construction-zone worked example: VLM evidence can flag an
    ADDITIONAL condition, but a clean frame with no VLM evidence must not
    become worse than it would be without any VLM ever being wired in."""
    thresholds = OddThresholds(**_load_thresholds()["m4_odd"])
    mon_no_vlm = M4OddComplianceMonitor(thresholds)
    mon_with_vlm = M4OddComplianceMonitor(thresholds)

    reader = SyntheticReader(ScenarioSpec(duration_s=1.0), seed=1)
    frame = next(f for f in health_stream(reader) if not f.is_gap)

    baseline = mon_no_vlm.update(frame)

    mon_with_vlm.submit_vlm_evidence(
        VLMEvidence(tag=SceneTag.CLEAR, confidence=0.95, model_version="test-vlm", frame_id=frame.frame_id)
    )
    with_clear_tag = mon_with_vlm.update(frame)

    assert with_clear_tag.verdict == baseline.verdict
    assert with_clear_tag.score == baseline.score


def test_m1_heuristic_warms_up_then_scores() -> None:
    mon = M1HeuristicMonitor(HeuristicThresholds(min_window_for_scoring=5))
    reader = SyntheticReader(ScenarioSpec(duration_s=2.0), seed=1)
    frames = [f for f in health_stream(reader) if not f.is_gap]
    results = [mon.update(f) for f in frames[:5]]
    assert all(r.confidence < 0.5 for r in results)  # warming up
    later = mon.update(frames[5])
    assert later.confidence >= 0.9


def test_monitor_reports_stale_after_silence() -> None:
    thresholds = TimingThresholds(**_load_thresholds()["m3_timing"])
    mon = M3TimingMonitor(thresholds)
    reader = SyntheticReader(ScenarioSpec(duration_s=1.0), seed=1)
    frame = next(f for f in health_stream(reader) if not f.is_gap)
    mon.update(frame)
    for _ in range(mon.staleness_budget_ticks + 1):
        mon._current_tick += 1  # simulate ticks passing without an update() call
    stale = mon.health_check(frame)
    assert stale is not None
    assert stale.verdict == Verdict.STALE


def test_monitor_never_crashes_on_missing_data() -> None:
    thresholds = SensorAgreementThresholds(**_load_thresholds()["m2_sensor_agreement"])
    mon = M2SensorAgreementMonitor(thresholds)
    reader = SyntheticReader(ScenarioSpec(duration_s=1.0), seed=1)
    frame = next(f for f in health_stream(reader) if not f.is_gap)
    empty_frame = frame.model_copy(update={"can": None, "gnss": None, "ego_pose": None})
    result = mon.update(empty_frame)  # must not raise
    assert result.verdict in (Verdict.WARN, Verdict.FAIL)
