from __future__ import annotations

from sentinel.config import load_yaml_with_hash
from sentinel.policy.gate import authorize, build_policy_table
from sentinel.policy.schema import ActionClass
from sentinel.trust.states import TrustState


def _table():
    parsed, _ = load_yaml_with_hash("config/policy_table.yaml")
    return build_policy_table(parsed)


def test_full_autonomy_allows_overtake() -> None:
    table = _table()
    decision = authorize(TrustState.FULL_AUTONOMY, ActionClass.OVERTAKE, frozenset(), table)
    assert decision.decision == "ALLOW"


def test_handoff_denies_everything() -> None:
    table = _table()
    for action in ActionClass:
        decision = authorize(TrustState.HANDOFF, action, frozenset(), table)
        assert decision.decision == "DENY"


def test_degraded_denies_overtake_via_base_authority() -> None:
    table = _table()
    decision = authorize(TrustState.DEGRADED, ActionClass.OVERTAKE, frozenset(), table)
    assert decision.decision == "DENY"
    assert decision.rule_id.startswith("POL-BASE")


def test_worked_example_construction_zone_denies_overtake_even_at_full_autonomy() -> None:
    """The spec's exact worked example: construction zone + degraded
    localization must deny overtaking even if trust_state alone would
    otherwise allow it."""
    table = _table()
    decision = authorize(
        TrustState.FULL_AUTONOMY,
        ActionClass.OVERTAKE,
        frozenset({"construction_zone_and_degraded_localization"}),
        table,
    )
    assert decision.decision == "DENY"
    assert decision.rule_id == "POL-014"
    assert "construction" in decision.justification.lower()


def test_unknown_action_class_defaults_to_deny() -> None:
    table = _table()
    decision = authorize(TrustState.FULL_AUTONOMY, "levitate", frozenset(), table)
    assert decision.decision == "DENY"
    assert decision.rule_id == "POL-000-DEFAULT-DENY"


def test_every_decision_cites_a_rule_id_and_justification() -> None:
    table = _table()
    for state in TrustState:
        for action in ActionClass:
            decision = authorize(state, action, frozenset(), table)
            assert decision.rule_id
            assert decision.justification


def test_reachable_state_totality() -> None:
    """Every (TrustState, ActionClass) pair must resolve to a decision —
    no implicit fall-through to permissive, per spec."""
    table = _table()
    for state in TrustState:
        for action in ActionClass:
            decision = authorize(state, action, frozenset(), table)
            assert decision.decision in ("ALLOW", "DENY")
