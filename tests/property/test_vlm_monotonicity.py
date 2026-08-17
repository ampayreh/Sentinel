"""Property-based proof of the spec's hard architectural rule: no VLM/LLM
output can ever move Sentinel to a MORE permissive trust/authority state.

Structure of the proof: for a fixed HealthFrame, compute the full set of
ALLOWed action classes with NO VLM evidence submitted to M4 at all (the
baseline — what the deterministic sensor-only path decides). Then, for
ANY VLMEvidence hypothesis can generate (any scene tag, any confidence in
[0,1], any frame_id), recompute the ALLOWed set after submitting that
evidence to M4. The property under test: the with-VLM ALLOWed set is
ALWAYS a subset of the baseline ALLOWed set — VLM evidence can only
remove authority, never grant it back or add to it.

This is checked through the real M4 monitor + real policy gate (not a
mock), so the proof covers the actual code path evidence flows through,
not just the gate's context-override plumbing (already covered by
test_policy_gate_properties.py's override-subset check).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

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

_reader = SyntheticReader(ScenarioSpec(duration_s=5.0), seed=123)
_FRAMES = [f for f in health_stream(_reader) if not f.is_gap][:20]

_vlm_evidence_strategy = st.builds(
    VLMEvidence,
    tag=st.sampled_from(list(SceneTag)),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    model_version=st.just("hypothesis-test-vlm"),
    frame_id=st.integers(min_value=0, max_value=10_000),
)


def _allowed_set(frame, evidence: VLMEvidence | None, trust_state: TrustState) -> set[ActionClass]:
    mon = M4OddComplianceMonitor(_thresholds)
    if evidence is not None:
        mon.submit_vlm_evidence(evidence)
    m4_result = mon.update(frame)
    context_flags = derive_context_flags(m4_result, frame)
    return {
        action
        for action in ActionClass
        if authorize(trust_state, action, context_flags, _policy_table).decision == "ALLOW"
    }


@given(evidence=_vlm_evidence_strategy, frame_idx=st.integers(min_value=0, max_value=len(_FRAMES) - 1))
def test_vlm_evidence_never_expands_authorized_actions(evidence: VLMEvidence, frame_idx: int) -> None:
    frame = _FRAMES[frame_idx]
    for trust_state in TrustState:
        baseline = _allowed_set(frame, None, trust_state)
        with_vlm = _allowed_set(frame, evidence, trust_state)
        assert with_vlm.issubset(baseline), (
            f"VLM evidence {evidence} EXPANDED authority at {trust_state}: "
            f"baseline={baseline} with_vlm={with_vlm}"
        )
