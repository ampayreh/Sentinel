"""VLM/LLM evidence adapter — the ONLY door a learned language/vision model
gets into Sentinel.

Hard architectural rule (spec): no LLM/VLM may ever be a safety authority.
A VLM never sets a trust state or authorizes an action; it only produces a
typed `VLMEvidence` object with a scene tag and confidence. That object is
consumed by M4 (ODD compliance) as *advisory* input to a deterministic
rule, and by the policy gate's monotonicity wrapper, which enforces —
structurally, not by convention — that evidence can only ever *tighten*
authority, never expand it.

Worked example from the spec:
    VLMEvidence(tag="construction_zone", confidence=0.86)
    -> M4 rule POL-014: construction_zone present AND localization degraded
       => overtake DENY.
A missing/low-confidence VLM tag can, at most, cause M4 to fall back to
"no advisory evidence" (neutral). It can never cause M4 to relax a rule
that would otherwise fire from sensor evidence alone.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator


class SceneTag(str, Enum):
    CONSTRUCTION_ZONE = "construction_zone"
    SCHOOL_ZONE = "school_zone"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    PEDESTRIAN_DENSE = "pedestrian_dense"
    UNKNOWN_OBSTACLE = "unknown_obstacle"
    CLEAR = "clear"


class VLMEvidence(BaseModel):
    """Typed, immutable output of a local VLM scene-tagging pass. This is
    the ENTIRE surface area through which a VLM can influence Sentinel."""

    model_config = ConfigDict(frozen=True)

    tag: SceneTag
    confidence: float
    model_version: str
    frame_id: int

    @field_validator("confidence")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {v}")
        return v


# Minimum confidence for a VLM tag to be treated as present at all by M4.
# Below this, evidence is neutral (equivalent to no tag) — never negative,
# never positive. This threshold itself lives in config in a full
# deployment; hardcoded here with a comment because it gates a structural
# safety property (monotonicity), not a tunable operating threshold.
MIN_ADVISORY_CONFIDENCE = 0.5


def is_actionable(evidence: VLMEvidence | None) -> bool:
    """An evidence object only counts as 'present' to downstream rules if
    it clears the minimum confidence bar. This is the sole gate a VLM
    output passes through before reaching M4 — there is no other path."""
    return evidence is not None and evidence.confidence >= MIN_ADVISORY_CONFIDENCE
