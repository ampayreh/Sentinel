from __future__ import annotations

from sentinel.config import load_yaml_with_hash
from sentinel.schemas.monitor_result import MonitorResult, Verdict
from sentinel.trust.config_loader import load_trust_config
from sentinel.trust.coordinator import TrustCoordinator
from sentinel.trust.states import TrustState


def _coordinator() -> TrustCoordinator:
    parsed, _ = load_yaml_with_hash("config/trust_weights.yaml")
    return TrustCoordinator(load_trust_config(parsed))


def _result(monitor_id: str, score: float, verdict: Verdict) -> MonitorResult:
    return MonitorResult(
        monitor_id=monitor_id,
        monitor_version="test",
        frame_id=0,
        monotonic_ns=0,
        score=score,
        verdict=verdict,
        confidence=1.0,
        evidence={},
        latency_us=0,
    )


ALL_PASS = {
    "M1_PERCEPTION": _result("M1_PERCEPTION", 1.0, Verdict.PASS),
    "M2_SENSOR_AGREEMENT": _result("M2_SENSOR_AGREEMENT", 1.0, Verdict.PASS),
    "M3_TIMING": _result("M3_TIMING", 1.0, Verdict.PASS),
    "M4_ODD_COMPLIANCE": _result("M4_ODD_COMPLIANCE", 1.0, Verdict.PASS),
}


def test_all_pass_stays_full_autonomy() -> None:
    coord = _coordinator()
    for _ in range(50):
        result = coord.step(ALL_PASS)
    assert result.trust_state == TrustState.FULL_AUTONOMY


def test_single_fail_does_not_immediately_escalate_below_k() -> None:
    coord = _coordinator()
    bad = dict(ALL_PASS, M2_SENSOR_AGREEMENT=_result("M2_SENSOR_AGREEMENT", 0.0, Verdict.FAIL))
    result = coord.step(bad)  # k-of-n requires >1 consecutive; escalate_k=2
    assert result.trust_state == TrustState.FULL_AUTONOMY


def test_repeated_fail_escalates_fast() -> None:
    coord = _coordinator()
    bad = dict(ALL_PASS, M2_SENSOR_AGREEMENT=_result("M2_SENSOR_AGREEMENT", 0.0, Verdict.FAIL))
    result = None
    for _ in range(3):
        result = coord.step(bad)
    assert result.trust_state != TrustState.FULL_AUTONOMY
    assert result.transitioned or result.trust_state != TrustState.FULL_AUTONOMY


def test_deescalation_requires_dwell_and_clean_streak() -> None:
    coord = _coordinator()
    bad = dict(ALL_PASS, M2_SENSOR_AGREEMENT=_result("M2_SENSOR_AGREEMENT", 0.0, Verdict.FAIL))
    for _ in range(3):
        coord.step(bad)
    escalated_state = coord.step(bad).trust_state
    assert escalated_state != TrustState.FULL_AUTONOMY

    # Immediately go clean — should NOT snap back to FULL_AUTONOMY yet
    # (min_dwell_ticks not satisfied).
    result = coord.step(ALL_PASS)
    assert result.trust_state == escalated_state

    # Feed enough clean samples to satisfy dwell + clean_n and confirm it
    # eventually recovers.
    final = None
    for _ in range(200):
        final = coord.step(ALL_PASS)
    assert final.trust_state == TrustState.FULL_AUTONOMY


def test_correlated_failure_across_independent_monitors_flagged() -> None:
    coord = _coordinator()
    both_fail = dict(
        ALL_PASS,
        M2_SENSOR_AGREEMENT=_result("M2_SENSOR_AGREEMENT", 0.0, Verdict.FAIL),
        M3_TIMING=_result("M3_TIMING", 0.0, Verdict.FAIL),
    )
    result = coord.step(both_fail)
    assert result.correlated_failure_detected is True


def test_stale_monitor_treated_as_risk() -> None:
    coord = _coordinator()
    stale = dict(ALL_PASS, M1_PERCEPTION=_result("M1_PERCEPTION", 1.0, Verdict.STALE))
    result = coord.step(stale)
    assert result.per_monitor_risk["M1_PERCEPTION"] == 1.0


def test_missing_monitor_excluded_not_zero_filled() -> None:
    coord = _coordinator()
    partial = {k: v for k, v in ALL_PASS.items() if k != "M3_TIMING"}
    result = coord.step(partial)
    assert "M3_TIMING" not in result.per_monitor_risk
    # Remaining monitors still fuse to a valid, unpenalized risk since
    # M3's absence is excluded from the weighted average, not zero-filled.
    assert result.fused_risk == 0.0
