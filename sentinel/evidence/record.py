"""EvidenceRecord: one append-only, hash-chained entry in the tamper-evident
log. This is what `sentinel replay` and `sentinel verify` operate on.

Every field that participates in the hash must be present and in a fixed
order — see `hash_chain.canonical_bytes`. Do not add a field here without
also handling it there, or verify() will silently stop covering it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sentinel.schemas.health_frame import HealthFrame
from sentinel.schemas.monitor_result import MonitorResult


class TrustTransitionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_state: str
    to_state: str
    fired_rules: list[str]  # e.g. ["ESCALATE_K_OF_N", "META_CORRELATED_FAILURE"]
    reason: str  # human-readable, cites monitors + values
    contributing_monitors: list[str]


class PolicyDecisionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_class: str
    decision: str  # "ALLOW" | "DENY"
    rule_id: str
    justification: str


class EvidenceRecord(BaseModel):
    """One decision-loop tick, fully self-contained for replay."""

    model_config = ConfigDict(frozen=True)

    event_id: int
    monotonic_ns: int
    wall_time_ns: int

    input_frame: HealthFrame
    input_digest: str  # sha256 of input_frame's canonical JSON

    monitor_results: list[MonitorResult]

    trust_state: str
    trust_transition: TrustTransitionRecord | None  # None if state unchanged this tick

    policy_decisions: list[PolicyDecisionRecord]

    model_versions: dict[str, str]  # monitor_id -> version, for provenance at a glance
    config_hash: str
    code_git_sha: str

    prev_record_hash: str  # "0" * 64 for the first record in a log
    record_hash: str  # sha256(canonical_bytes(this record minus record_hash) + prev_record_hash)
