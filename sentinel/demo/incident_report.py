"""Human-readable incident report, matching the exact shape from the
spec's worked example."""

from __future__ import annotations

import datetime as dt

from sentinel.evidence.record import EvidenceRecord
from sentinel.policy.schema import ActionClass
from sentinel.schemas.monitor_result import Verdict

_ACTION_LABELS: dict[ActionClass, str] = {
    ActionClass.REDUCE_SPEED: "Reduce speed",
    ActionClass.MAINTAIN_LANE: "Maintain lane",
    ActionClass.LANE_CHANGE: "Change lane",
    ActionClass.OVERTAKE: "Execute overtaking maneuver",
    ActionClass.UNPROTECTED_TURN: "Unprotected turn",
    ActionClass.INCREASE_AUTONOMY_AUTHORITY: "Increase autonomy authority",
}

_MONITOR_LABELS = {
    "M1_PERCEPTION": "M1 Perception",
    "M2_SENSOR_AGREEMENT": "M2 Sensor Agreement",
    "M3_TIMING": "M3 Timing",
    "M4_ODD_COMPLIANCE": "M4 ODD Compliance",
}


def format_incident_report(record: EvidenceRecord) -> str:
    ts = dt.datetime.fromtimestamp(record.wall_time_ns / 1e9, tz=dt.UTC).strftime("%H:%M:%S.%f")[:-3]

    m2 = next((m for m in record.monitor_results if m.monitor_id == "M2_SENSOR_AGREEMENT"), None)
    m3 = next((m for m in record.monitor_results if m.monitor_id == "M3_TIMING"), None)
    m1 = next((m for m in record.monitor_results if m.monitor_id == "M1_PERCEPTION"), None)
    m4 = next((m for m in record.monitor_results if m.monitor_id == "M4_ODD_COMPLIANCE"), None)

    perception_conf = f"{m1.score:.2f}" if m1 else "n/a"
    sensor_agreement = f"{m2.score:.2f}" if m2 else "n/a"
    jitter_ms = m3.evidence.get("jitter_ewma_ms") if m3 else None
    jitter_str = f"{jitter_ms:.0f} ms" if isinstance(jitter_ms, int | float) else "n/a"
    model_version = record.model_versions.get("M1_PERCEPTION", "n/a")
    odd_valid = "VALID" if (m4 and m4.verdict == Verdict.PASS) else "INVALID"

    lines = []
    lines.append(f"EVENT ID: {record.event_id:07d}          T = {ts}")
    lines.append(f"Perception confidence: {perception_conf}   Sensor agreement: {sensor_agreement}")
    lines.append(f"Timing jitter: {jitter_str}          Model version: {model_version}")
    lines.append(f"Operating envelope: {odd_valid}     Policy state: {record.trust_state}")
    lines.append("MONITOR RESULTS")
    for monitor_id in ["M1_PERCEPTION", "M2_SENSOR_AGREEMENT", "M3_TIMING", "M4_ODD_COMPLIANCE"]:
        result = next((m for m in record.monitor_results if m.monitor_id == monitor_id), None)
        label = _MONITOR_LABELS[monitor_id]
        verdict = result.verdict.value if result else "STALE"
        lines.append(f"  {label:<20}{verdict}")
    lines.append(f"TRUST DECISION: {record.trust_state}")
    lines.append("AUTHORIZED ACTIONS")

    allowed = {d.action_class: d.decision == "ALLOW" for d in record.policy_decisions}
    ordered = [ActionClass.REDUCE_SPEED, ActionClass.MAINTAIN_LANE, ActionClass.OVERTAKE, ActionClass.INCREASE_AUTONOMY_AUTHORITY]
    col1 = ordered[:2]
    col2 = ordered[2:]
    for a, b in zip(col1, col2, strict=True):
        a_mark = "✓" if allowed.get(a.value, False) else "✗"
        b_mark = "✓" if allowed.get(b.value, False) else "✗"
        lines.append(f"  {a_mark} {_ACTION_LABELS[a]:<18} {b_mark} {_ACTION_LABELS[b]}")

    if record.trust_transition is not None:
        lines.append("")
        lines.append(f"TRANSITION: {record.trust_transition.from_state} -> {record.trust_transition.to_state}")
        lines.append(f"  Fired rules: {', '.join(record.trust_transition.fired_rules)}")
        lines.append(f"  Reason: {record.trust_transition.reason}")

    denies = [d for d in record.policy_decisions if d.decision == "DENY" and d.rule_id != "POL-BASE-FULL_AUTONOMY"]
    if denies:
        lines.append("")
        lines.append("CITED RULES:")
        for d in denies:
            lines.append(f"  [{d.rule_id}] {d.action_class}: {d.justification}")

    return "\n".join(lines)
