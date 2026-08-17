"""Policy gate: the sole authorization authority in Sentinel.

Pure function, by construction:
  - No I/O (the table is loaded once, elsewhere, and passed in as a plain
    frozen dataclass).
  - No randomness.
  - No wall-clock reads.
  - No floats compared without explicit epsilon (this module contains no
    float comparisons at all — trust state and context flags are
    categorical by the time they reach the gate; all float thresholding
    already happened in the monitors/trust coordinator).

Default-deny: any action class not present in `ActionClass`, or a
trust_state not present in `base_authority` (impossible after schema
validation, but checked anyway — see `_all_states_present` in schema.py
plus the runtime assertion below), resolves to DENY with a rule ID that
says so. There is no code path that falls through to ALLOW.

Monotonicity: `context_overrides` can only ever REMOVE action classes
from what `base_authority` already grants for a trust state — the schema
gives overrides no ALLOW form. This is what makes "VLM/monitor evidence
can only tighten authority, never expand it" a structural property:
overrides are applied as a set-difference against the base grant, never
a union.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.policy.schema import ActionClass, PolicyTableFile
from sentinel.trust.states import TrustState

DEFAULT_DENY_RULE_ID = "POL-000-DEFAULT-DENY"
BASE_AUTHORITY_RULE_PREFIX = "POL-BASE"


@dataclass(frozen=True)
class ContextOverrideRule:
    rule_id: str
    description: str
    context_flags: frozenset[str]
    deny_action_classes: frozenset[ActionClass]


@dataclass(frozen=True)
class PolicyTable:
    """Runtime-ready, immutable form of policy_table.yaml. Build once via
    `build_policy_table`, then pass into every `authorize()` call — the
    gate itself never touches the YAML file."""

    base_authority: dict[TrustState, frozenset[ActionClass]]
    # Sorted by rule_id at build time so evaluation order (and therefore
    # which rule_id gets cited on a DENY, when multiple could apply) is
    # deterministic and stable across replay.
    context_overrides: tuple[ContextOverrideRule, ...]


def build_policy_table(parsed_yaml: dict) -> PolicyTable:
    f = PolicyTableFile.model_validate(parsed_yaml)
    base = {state: frozenset(actions) for state, actions in f.base_authority.items()}
    overrides = tuple(
        sorted(
            (
                ContextOverrideRule(
                    rule_id=o.rule_id,
                    description=o.description,
                    context_flags=frozenset(o.when_context_flags_any),
                    deny_action_classes=frozenset(o.deny_action_classes),
                )
                for o in f.context_overrides
            ),
            key=lambda r: r.rule_id,
        )
    )
    return PolicyTable(base_authority=base, context_overrides=overrides)


@dataclass(frozen=True)
class PolicyDecision:
    action_class: str
    decision: str  # "ALLOW" | "DENY"
    rule_id: str
    justification: str


def authorize(
    trust_state: TrustState,
    action_class: ActionClass | str,
    context_flags: frozenset[str],
    table: PolicyTable,
) -> PolicyDecision:
    """The single entry point. Every call returns a PolicyDecision — there
    is no exception path and no None return, so callers cannot forget to
    handle "unknown outcome"."""

    # Default-deny on unknown action classes (e.g. a string that doesn't
    # map to the closed ActionClass enum).
    if isinstance(action_class, str):
        try:
            action_class = ActionClass(action_class)
        except ValueError:
            return PolicyDecision(
                action_class=str(action_class),
                decision="DENY",
                rule_id=DEFAULT_DENY_RULE_ID,
                justification=f"'{action_class}' is not a recognized action class; default-deny.",
            )

    granted = table.base_authority.get(trust_state)
    if granted is None:
        # Should be unreachable after schema validation; kept as a hard
        # safety net so an internal bug fails DENY, never ALLOW.
        return PolicyDecision(
            action_class=action_class.value,
            decision="DENY",
            rule_id=DEFAULT_DENY_RULE_ID,
            justification=f"no base_authority entry for trust_state={trust_state}; default-deny.",
        )

    if action_class not in granted:
        return PolicyDecision(
            action_class=action_class.value,
            decision="DENY",
            rule_id=f"{BASE_AUTHORITY_RULE_PREFIX}-{trust_state.value}",
            justification=(
                f"trust_state={trust_state.value} does not grant {action_class.value}."
            ),
        )

    # Context overrides can only remove, never add, and are evaluated in
    # deterministic rule_id order; the first matching override wins.
    for rule in table.context_overrides:
        if action_class in rule.deny_action_classes and (rule.context_flags & context_flags):
            matched_flags = sorted(rule.context_flags & context_flags)
            return PolicyDecision(
                action_class=action_class.value,
                decision="DENY",
                rule_id=rule.rule_id,
                justification=f"{rule.description} (matched context: {', '.join(matched_flags)})",
            )

    return PolicyDecision(
        action_class=action_class.value,
        decision="ALLOW",
        rule_id=f"{BASE_AUTHORITY_RULE_PREFIX}-{trust_state.value}",
        justification=f"trust_state={trust_state.value} grants {action_class.value}; no override matched.",
    )
