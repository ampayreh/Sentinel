"""Validated loader: config/trust_weights.yaml -> TrustCoordinatorConfig.

Kept separate from sentinel/config.py's generic YAML loader so the
pydantic schema for this specific file lives next to the dataclasses it
populates.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from sentinel.trust.coordinator import MetaConfig, TrustCoordinatorConfig
from sentinel.trust.fusion import FusionWeights
from sentinel.trust.hysteresis import HysteresisConfig
from sentinel.trust.states import RiskThresholds


class _RiskThresholdsSchema(BaseModel):
    full_autonomy_max_risk: float = Field(ge=0, le=1)
    degraded_max_risk: float = Field(ge=0, le=1)
    minimal_risk_max_risk: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _monotonic(self) -> _RiskThresholdsSchema:
        if not (self.full_autonomy_max_risk < self.degraded_max_risk < self.minimal_risk_max_risk):
            raise ValueError(
                "risk_thresholds must be strictly increasing: "
                "full_autonomy_max_risk < degraded_max_risk < minimal_risk_max_risk"
            )
        return self


class _HysteresisSchema(BaseModel):
    escalate_k: int = Field(ge=1)
    escalate_n: int = Field(ge=1)
    min_dwell_ticks: int = Field(ge=0)
    deescalate_clean_n: int = Field(ge=1)

    @model_validator(mode="after")
    def _k_le_n(self) -> _HysteresisSchema:
        if self.escalate_k > self.escalate_n:
            raise ValueError("escalate_k must be <= escalate_n")
        return self


class _MetaSchema(BaseModel):
    stale_monitor_risk: float = Field(ge=0, le=1)
    correlated_failure_min_monitors: int = Field(ge=2)
    correlated_failure_window_ticks: int = Field(ge=1)
    correlated_failure_risk_bonus: float = Field(ge=0, le=1)


class TrustWeightsFile(BaseModel):
    schema_version: int
    weights: dict[str, float]
    risk_thresholds: _RiskThresholdsSchema
    hysteresis: _HysteresisSchema
    meta: _MetaSchema

    @model_validator(mode="after")
    def _weights_positive(self) -> TrustWeightsFile:
        if not self.weights:
            raise ValueError("weights must not be empty")
        for mid, w in self.weights.items():
            if w <= 0:
                raise ValueError(f"weight for {mid} must be > 0, got {w}")
        return self


def load_trust_config(parsed: dict) -> TrustCoordinatorConfig:
    f = TrustWeightsFile.model_validate(parsed)
    return TrustCoordinatorConfig(
        weights=FusionWeights(weights=dict(f.weights)),
        risk_thresholds=RiskThresholds(
            full_autonomy_max_risk=f.risk_thresholds.full_autonomy_max_risk,
            degraded_max_risk=f.risk_thresholds.degraded_max_risk,
            minimal_risk_max_risk=f.risk_thresholds.minimal_risk_max_risk,
        ),
        hysteresis=HysteresisConfig(
            escalate_k=f.hysteresis.escalate_k,
            escalate_n=f.hysteresis.escalate_n,
            min_dwell_ticks=f.hysteresis.min_dwell_ticks,
            deescalate_clean_n=f.hysteresis.deescalate_clean_n,
        ),
        meta=MetaConfig(
            stale_monitor_risk=f.meta.stale_monitor_risk,
            correlated_failure_min_monitors=f.meta.correlated_failure_min_monitors,
            correlated_failure_window_ticks=f.meta.correlated_failure_window_ticks,
            correlated_failure_risk_bonus=f.meta.correlated_failure_risk_bonus,
        ),
    )
