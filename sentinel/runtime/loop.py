"""The actual Monitor -> Decide -> Constrain -> Act -> Record loop.

This module is the decision path's orchestrator. It imports monitors,
trust, policy, and evidence — but only the heuristic M1 by default, so
importing `sentinel.runtime.loop` does not require torch. Pass a
different M1 implementation (e.g. the learned monitor from
sentinel.monitors.m1_learned_model) via `DecisionLoop(m1_monitor=...)` if
you want it; the loop itself is agnostic.

Every action class in `sentinel.policy.schema.ActionClass` is evaluated
against the policy gate every tick (not just the one the autonomy stack
happens to be requesting), because the evidence record and the demo's
"authorized actions" table both need the full authority picture at that
instant, not a single yes/no.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from sentinel.evidence.record import EvidenceRecord, PolicyDecisionRecord, TrustTransitionRecord
from sentinel.evidence.writer import EvidenceWriter
from sentinel.monitors.base import Monitor
from sentinel.policy.context import derive_context_flags
from sentinel.policy.gate import PolicyTable, authorize
from sentinel.policy.schema import ActionClass
from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult
from sentinel.trust.coordinator import TrustCoordinator, TrustStepResult


def get_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except Exception:  # noqa: BLE001 - git absence must not crash the loop
        return "unknown"


@dataclass
class DecisionLoopConfig:
    config_hash: str
    code_git_sha: str


class DecisionLoop:
    def __init__(
        self,
        monitors: dict[str, Monitor],
        trust_coordinator: TrustCoordinator,
        policy_table: PolicyTable,
        evidence_writer: EvidenceWriter,
        config: DecisionLoopConfig,
    ) -> None:
        self.monitors = monitors
        self.trust_coordinator = trust_coordinator
        self.policy_table = policy_table
        self.evidence_writer = evidence_writer
        self.config = config
        self._last_trust_state: str | None = None

    def process_frame(self, frame: HealthFrame) -> EvidenceRecord:
        t_start = time.monotonic_ns()

        # --- Monitor ---
        results: dict[str, MonitorResult] = {}
        for monitor_id in sorted(self.monitors.keys()):  # sorted: deterministic evaluation order
            results[monitor_id] = self.monitors[monitor_id].update(frame)

        # --- Decide (trust coordinator) ---
        trust_step: TrustStepResult = self.trust_coordinator.step(results)

        transition_record: TrustTransitionRecord | None = None
        if trust_step.transitioned:
            transition_record = TrustTransitionRecord(
                from_state=self._last_trust_state or trust_step.trust_state.value,
                to_state=trust_step.trust_state.value,
                fired_rules=[trust_step.rule_id],
                reason=trust_step.reason,
                contributing_monitors=list(trust_step.contributing_monitors),
            )
        self._last_trust_state = trust_step.trust_state.value

        # --- Constrain (policy gate), evaluated for every action class ---
        context_flags = derive_context_flags(results.get("M4_ODD_COMPLIANCE"), frame)
        policy_decisions = [
            PolicyDecisionRecord(
                action_class=decision.action_class,
                decision=decision.decision,
                rule_id=decision.rule_id,
                justification=decision.justification,
            )
            for action_class in sorted(ActionClass, key=lambda a: a.value)
            for decision in [
                authorize(trust_step.trust_state, action_class, context_flags, self.policy_table)
            ]
        ]

        input_digest = _digest(frame)
        model_versions = {mid: r.monitor_version for mid, r in sorted(results.items())}

        record = EvidenceRecord(
            event_id=0,  # filled by EvidenceWriter
            monotonic_ns=frame.monotonic_ns,
            wall_time_ns=frame.wall_time_ns,
            input_frame=frame,
            input_digest=input_digest,
            monitor_results=[results[mid] for mid in sorted(results.keys())],
            trust_state=trust_step.trust_state.value,
            trust_transition=transition_record,
            policy_decisions=policy_decisions,
            model_versions=model_versions,
            config_hash=self.config.config_hash,
            code_git_sha=self.config.code_git_sha,
            prev_record_hash="",  # filled by EvidenceWriter
            record_hash="",  # filled by EvidenceWriter
        )
        final_record = self.evidence_writer.build_and_append(record)

        _decision_latency_us = (time.monotonic_ns() - t_start) // 1000
        return final_record


def _digest(frame: HealthFrame) -> str:
    import hashlib

    return hashlib.sha256(frame.model_dump_json().encode("utf-8")).hexdigest()
