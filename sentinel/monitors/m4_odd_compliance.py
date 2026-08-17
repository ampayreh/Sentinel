"""M4 ODD (Operating Design Domain) compliance monitor — CPU, rule-based.

Checks the operating envelope: speed vs road-class limit, localization
quality (GNSS satellite count + HDOP), geofence, model-version validity.
Consumes VLM scene tags strictly as *advisory evidence* (see
sentinel.policy.evidence_adapter) — a VLM tag can only ever add a FAIL/WARN
condition on top of what the deterministic rules already found, never
remove one. This monitor is itself one of the two hard rule-based gates
(the other being sentinel.policy.gate) referenced in the spec's worked
example: "construction zone + degraded localization => overtake denied
(POL-014)" is decided here at the evidence level and enforced downstream
by the policy gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.monitors.base import Monitor
from sentinel.policy.evidence_adapter import SceneTag, VLMEvidence, is_actionable
from sentinel.schemas.health_frame import HealthFrame, RoadClass
from sentinel.schemas.monitor_result import MonitorResult, Verdict


@dataclass(frozen=True)
class OddThresholds:
    min_gnss_satellites: int
    max_hdop_valid: float
    max_speed_residential_mps: float
    max_speed_arterial_mps: float = 22.3  # 50 mph
    max_speed_highway_mps: float = 33.5
    construction_zone_overtake_prohibited: bool = True


_ROAD_SPEED_LIMIT_KEY = {
    RoadClass.RESIDENTIAL: "max_speed_residential_mps",
    RoadClass.ARTERIAL: "max_speed_arterial_mps",
    RoadClass.HIGHWAY: "max_speed_highway_mps",
}


class M4OddComplianceMonitor(Monitor):
    monitor_id = "M4_ODD_COMPLIANCE"
    monitor_version = "0.1.0"

    def __init__(self, thresholds: OddThresholds) -> None:
        super().__init__()
        self.t = thresholds
        self._latest_vlm_evidence: VLMEvidence | None = None

    def submit_vlm_evidence(self, evidence: VLMEvidence) -> None:
        """Called by the VLM context adapter (sentinel/demo or eval harness)
        to hand this monitor the latest advisory scene tag. Never called
        from inside `_compute` — evidence flows in, it is never pulled."""
        self._latest_vlm_evidence = evidence

    def _compute(self, frame: HealthFrame) -> MonitorResult:
        t = self.t
        reasons: list[str] = []
        verdict = Verdict.PASS

        localization_degraded = False
        if frame.gnss:
            if (
                frame.gnss.num_satellites is not None
                and frame.gnss.num_satellites < t.min_gnss_satellites
            ) or (frame.gnss.hdop is not None and frame.gnss.hdop > t.max_hdop_valid):
                localization_degraded = True
                reasons.append("localization_degraded")
                verdict = _max_verdict(verdict, Verdict.WARN)
            if frame.gnss.fix_valid is False:
                localization_degraded = True
                verdict = _max_verdict(verdict, Verdict.FAIL)
                reasons.append("gnss_fix_invalid")

        speed = frame.can.speed_mps if frame.can else None
        limit_key = _ROAD_SPEED_LIMIT_KEY.get(frame.road_class)
        limit = getattr(t, limit_key) if limit_key else None
        speed_violation = speed is not None and limit is not None and speed > limit
        if speed_violation:
            verdict = _max_verdict(verdict, Verdict.FAIL)
            reasons.append(f"speed_over_limit:{speed:.1f}>{limit:.1f}")

        if frame.geofence_ok is False:
            verdict = _max_verdict(verdict, Verdict.FAIL)
            reasons.append("outside_geofence")

        # --- VLM advisory evidence: can only ADD to what's already found ---
        vlm = self._latest_vlm_evidence
        construction_zone_flagged = (
            is_actionable(vlm)
            and vlm is not None
            and vlm.tag == SceneTag.CONSTRUCTION_ZONE
        )
        if construction_zone_flagged and t.construction_zone_overtake_prohibited:
            # POL-014 worked example: this monitor only *records* the
            # advisory fact; the DENY itself is enforced in the policy
            # gate (sentinel/policy/gate.py), which is the sole
            # authorization authority per the hard architectural rule.
            reasons.append("advisory:construction_zone_overtake_restricted")
            if localization_degraded:
                verdict = _max_verdict(verdict, Verdict.WARN)
                reasons.append("advisory:construction_zone+degraded_localization")

        score = 1.0 if verdict == Verdict.PASS else (0.5 if verdict == Verdict.WARN else 0.0)

        return MonitorResult(
            monitor_id=self.monitor_id,
            monitor_version=self.monitor_version,
            frame_id=frame.frame_id,
            monotonic_ns=frame.monotonic_ns,
            score=score,
            verdict=verdict,
            confidence=1.0,
            evidence={
                "localization_degraded": localization_degraded,
                "speed_violation": speed_violation,
                "construction_zone_flagged": construction_zone_flagged,
                "reasons": ";".join(reasons) if reasons else "",
            },
            latency_us=0,
        )


def _verdict_rank(v: Verdict) -> int:
    return {Verdict.PASS: 0, Verdict.WARN: 1, Verdict.FAIL: 2, Verdict.STALE: 3}[v]


def _max_verdict(a: Verdict, b: Verdict) -> Verdict:
    return a if _verdict_rank(a) >= _verdict_rank(b) else b
