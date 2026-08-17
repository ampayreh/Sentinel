"""Monotonicity fuzzing test suite: proves that advisory LLM/VLM evidence can NEVER expand authority."""

from __future__ import annotations
import itertools
import pytest

from sentinel.config import load_yaml_with_hash
from sentinel.monitors.m4_odd_compliance import M4OddComplianceMonitor, OddThresholds
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader
from sentinel.pipeline.stream import health_stream
from sentinel.policy.context import derive_context_flags
from sentinel.policy.evidence_adapter import SceneTag, VLMEvidence
from sentinel.policy.gate import authorize, build_policy_table
from sentinel.policy.schema import ActionClass
from sentinel.trust.states import TrustState

_thresholds = OddThresholds(**load_yaml_with_hash("config/monitor_thresholds.yaml")[0]["m4_odd"])
_policy_table = build_policy_table(load_yaml_with_hash("config/policy_table.yaml")[0])
ALL_TAGS = list(SceneTag)

_reader = SyntheticReader(ScenarioSpec(duration_s=2.0), seed=1)
_SAMPLE_FRAME = next(health_stream(_reader))

@pytest.mark.parametrize("state", list(TrustState))
def test_advisory_never_grants_authority_fuzz(state):
    """Fuzz over all tag pairs across all trust states."""
    mon_base = M4OddComplianceMonitor(_thresholds)
    base_m4 = mon_base.update(_SAMPLE_FRAME)
    base_flags = derive_context_flags(base_m4, _SAMPLE_FRAME)
    
    baseline_allowed = {
        action: (authorize(state, action, base_flags, _policy_table).decision == "ALLOW")
        for action in ActionClass
    }

    subsets = [()] + [(t,) for t in ALL_TAGS] + list(itertools.combinations(ALL_TAGS, 2))

    for tag_tuple in subsets:
        mon = M4OddComplianceMonitor(_thresholds)
        for tag in tag_tuple:
            mon.submit_vlm_evidence(VLMEvidence(tag=tag, confidence=0.9, model_version="fuzz", frame_id=1))
        m4_res = mon.update(_SAMPLE_FRAME)
        flags = derive_context_flags(m4_res, _SAMPLE_FRAME)

        for action in ActionClass:
            decision = authorize(state, action, flags, _policy_table).decision
            if decision == "ALLOW":
                assert baseline_allowed[action], (
                    f"MONOTONICITY VIOLATED: tag {tag_tuple} granted {action} in {state}"
                )
