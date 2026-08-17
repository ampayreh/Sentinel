"""Builds the default (heuristic M1) decision-loop components from the
on-disk config files. Used by both `sentinel run` and `sentinel replay` so
a live run and its replay are built from the exact same construction
path.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from sentinel.config import load_yaml_with_hash
from sentinel.evidence.writer import EvidenceWriter
from sentinel.monitors.base import Monitor
from sentinel.monitors.m1_perception import HeuristicThresholds, M1HeuristicMonitor
from sentinel.monitors.m2_sensor_agreement import (
    M2SensorAgreementMonitor,
    SensorAgreementThresholds,
)
from sentinel.monitors.m3_timing import M3TimingMonitor, TimingThresholds
from sentinel.monitors.m4_odd_compliance import M4OddComplianceMonitor, OddThresholds
from sentinel.policy.gate import PolicyTable, build_policy_table
from sentinel.runtime.loop import DecisionLoop, DecisionLoopConfig, get_git_sha
from sentinel.trust.config_loader import load_trust_config
from sentinel.trust.coordinator import TrustCoordinator


@dataclass(frozen=True)
class RuntimeConfigPaths:
    monitor_thresholds: str = "config/monitor_thresholds.yaml"
    trust_weights: str = "config/trust_weights.yaml"
    policy_table: str = "config/policy_table.yaml"


def build_monitors(monitor_thresholds_parsed: dict) -> dict[str, Monitor]:
    mt = monitor_thresholds_parsed
    monitors: dict[str, Monitor] = {
        "M1_PERCEPTION": M1HeuristicMonitor(HeuristicThresholds()),
        "M2_SENSOR_AGREEMENT": M2SensorAgreementMonitor(
            SensorAgreementThresholds(**mt["m2_sensor_agreement"])
        ),
        "M3_TIMING": M3TimingMonitor(TimingThresholds(**mt["m3_timing"])),
        "M4_ODD_COMPLIANCE": M4OddComplianceMonitor(
            OddThresholds(**mt["m4_odd"])
        ),
    }
    return monitors


def build_policy(policy_table_parsed: dict) -> PolicyTable:
    return build_policy_table(policy_table_parsed)


@dataclass
class RuntimeBundle:
    monitors: dict[str, Monitor]
    trust_coordinator: TrustCoordinator
    policy_table: PolicyTable
    loop_config: DecisionLoopConfig
    combined_config_hash: str


def build_runtime(paths: RuntimeConfigPaths = RuntimeConfigPaths()) -> RuntimeBundle:  # noqa: B008 (frozen, immutable default)
    mt_parsed, mt_hash = load_yaml_with_hash(paths.monitor_thresholds)
    tw_parsed, tw_hash = load_yaml_with_hash(paths.trust_weights)
    pt_parsed, pt_hash = load_yaml_with_hash(paths.policy_table)

    monitors = build_monitors(mt_parsed)
    trust_coordinator = TrustCoordinator(load_trust_config(tw_parsed))
    policy_table = build_policy(pt_parsed)

    import hashlib

    combined_hash = hashlib.sha256((mt_hash + tw_hash + pt_hash).encode()).hexdigest()

    loop_config = DecisionLoopConfig(config_hash=combined_hash, code_git_sha=get_git_sha())
    return RuntimeBundle(
        monitors=monitors,
        trust_coordinator=trust_coordinator,
        policy_table=policy_table,
        loop_config=loop_config,
        combined_config_hash=combined_hash,
    )


def build_decision_loop(
    evidence_log_path: str | Path,
    paths: RuntimeConfigPaths = RuntimeConfigPaths(),  # noqa: B008 (frozen, immutable default)
    flush_every_n_records: int = 1,
) -> DecisionLoop:
    bundle = build_runtime(paths)
    writer = EvidenceWriter(evidence_log_path, flush_every_n_records=flush_every_n_records)
    return DecisionLoop(
        monitors=bundle.monitors,
        trust_coordinator=bundle.trust_coordinator,
        policy_table=bundle.policy_table,
        evidence_writer=writer,
        config=bundle.loop_config,
    )


def build_ephemeral_decision_loop(
    paths: RuntimeConfigPaths = RuntimeConfigPaths(),  # noqa: B008 (frozen, immutable default)
) -> tuple[DecisionLoop, str]:
    """Build a decision loop writing to a fresh temp file — used by replay,
    which needs its own output log to diff against the original."""
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - lifetime managed by caller
        prefix="sentinel_replay_", suffix=".jsonl", delete=False
    )
    tmp.close()
    loop = build_decision_loop(tmp.name, paths)
    return loop, tmp.name
