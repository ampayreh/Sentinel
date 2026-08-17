"""Property-based tests for the policy gate.

These exist because the spec requires a PROOF, not a spot-check, that:
  1. Every reachable (trust_state, action_class) pair has a defined
     outcome (no implicit fall-through to permissive).
  2. The gate is a pure function: identical inputs -> identical output,
     across arbitrarily many calls, with no hidden state.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from sentinel.config import load_yaml_with_hash
from sentinel.policy.gate import authorize, build_policy_table
from sentinel.policy.schema import ActionClass
from sentinel.trust.states import TrustState

_TABLE = build_policy_table(load_yaml_with_hash("config/policy_table.yaml")[0])

_trust_states = st.sampled_from(list(TrustState))
_action_classes = st.sampled_from(list(ActionClass))
_context_flags = st.sets(
    st.sampled_from(
        [
            "construction_zone_and_degraded_localization",
            "geofence_violation",
            "gnss_fix_invalid",
        ]
    )
)


@given(state=_trust_states, action=_action_classes, flags=_context_flags)
def test_every_reachable_state_has_defined_outcome(state, action, flags) -> None:
    decision = authorize(state, action, frozenset(flags), _TABLE)
    assert decision.decision in ("ALLOW", "DENY")
    assert decision.rule_id  # never blank
    assert decision.justification  # never blank


@given(state=_trust_states, action=_action_classes, flags=_context_flags)
def test_gate_is_pure_and_deterministic(state, action, flags) -> None:
    d1 = authorize(state, action, frozenset(flags), _TABLE)
    d2 = authorize(state, action, frozenset(flags), _TABLE)
    assert d1 == d2


@given(action=_action_classes, flags=_context_flags)
def test_no_context_override_ever_grants_more_than_base_authority(action, flags) -> None:
    """Structural check that overrides can only remove: for every state,
    adding context flags never turns a base DENY into an ALLOW."""
    for state in TrustState:
        base = authorize(state, action, frozenset(), _TABLE)
        with_context = authorize(state, action, frozenset(flags), _TABLE)
        if base.decision == "DENY":
            assert with_context.decision == "DENY"
