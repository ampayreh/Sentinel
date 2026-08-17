"""Schema + validator for config/policy_table.yaml.

The known set of action classes and trust states is closed (not derived
from the YAML) — this is deliberate: a typo'd action class in the YAML
must fail validation loudly at load time, not silently create a new
permissive category.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, model_validator

from sentinel.trust.states import TrustState


class ActionClass(str, Enum):
    MAINTAIN_LANE = "maintain_lane"
    REDUCE_SPEED = "reduce_speed"
    LANE_CHANGE = "lane_change"
    OVERTAKE = "overtake"
    UNPROTECTED_TURN = "unprotected_turn"
    INCREASE_AUTONOMY_AUTHORITY = "increase_autonomy_authority"


class ContextOverride(BaseModel):
    rule_id: str
    description: str
    when_context_flags_any: list[str]
    deny_action_classes: list[ActionClass]


class PolicyTableFile(BaseModel):
    schema_version: int
    base_authority: dict[TrustState, list[ActionClass]]
    context_overrides: list[ContextOverride]

    @model_validator(mode="after")
    def _all_states_present(self) -> PolicyTableFile:
        missing = [s for s in TrustState if s not in self.base_authority]
        if missing:
            raise ValueError(
                f"base_authority missing entries for trust states: {missing} — "
                "every TrustState must be explicit (no implicit fall-through)"
            )
        return self

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> PolicyTableFile:
        ids = [o.rule_id for o in self.context_overrides]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate rule_id(s) in context_overrides: {dupes}")
        return self
