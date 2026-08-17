"""Config loading: pydantic-settings for the top-level runtime config, plus
a generic versioned-YAML loader with schema validation for the three
config-as-data files (trust weights, policy table, monitor thresholds).

Rule enforced here: nothing in `sentinel/trust`, `sentinel/policy`, or
`sentinel/evidence` reads a config file directly off disk at decision time.
Config is loaded once, validated, and passed in as plain Python objects —
so replay can pin the exact config content (and its hash) into the
evidence record instead of re-reading a file that might have changed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeConfig(BaseModel):
    tick_hz: float = Field(gt=0)
    log_level: str = "INFO"
    json_logs: bool = False
    decision_latency_budget_ms: float = Field(gt=0)


class EvidenceConfig(BaseModel):
    log_dir: str
    format: str = "jsonl"
    flush_every_n_records: int = Field(ge=1)


class PathsConfig(BaseModel):
    trust_weights: str
    policy_table: str
    monitor_thresholds: str


class SentinelConfig(BaseSettings):
    """Loaded from config/sentinel.yaml, overridable via SENTINEL_* env vars."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_nested_delimiter="__")

    schema_version: int
    runtime: RuntimeConfig
    evidence: EvidenceConfig
    paths: PathsConfig


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_yaml_with_hash(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a YAML file, returning (parsed_dict, sha256_of_raw_bytes).

    The hash is computed over the raw file bytes, not the parsed structure,
    so it changes on any byte-level edit (comments, whitespace, ordering)
    and is safe to embed in evidence records as a config provenance stamp.
    """
    p = Path(path)
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    parsed = yaml.safe_load(raw.decode("utf-8"))
    return parsed, digest


def load_sentinel_config(path: str | Path = "config/sentinel.yaml") -> SentinelConfig:
    parsed, _digest = load_yaml_with_hash(path)
    return SentinelConfig.model_validate(parsed)
