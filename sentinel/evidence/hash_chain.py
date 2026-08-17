"""Tamper-evident SHA-256 hash chain over EvidenceRecords.

canonical_bytes() is the single source of truth for what gets hashed: it
serializes the record (excluding `record_hash` itself) as JSON with sorted
keys and no whitespace ambiguity, so the same record always hashes to the
same digest regardless of platform, dict insertion order, or serializer
version drift within this codebase. GENUINELY DO NOT change this
function's output format without a schema_version bump — doing so breaks
verify() for every previously written log.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sentinel.evidence.record import EvidenceRecord

GENESIS_HASH = "0" * 64


def canonical_bytes(record: EvidenceRecord) -> bytes:
    payload: dict[str, Any] = record.model_dump(mode="json", exclude={"record_hash"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_record_hash(record: EvidenceRecord) -> str:
    h = hashlib.sha256()
    h.update(canonical_bytes(record))
    h.update(record.prev_record_hash.encode("utf-8"))
    return h.hexdigest()


class HashChainError(Exception):
    """Raised by verify_chain when a record's hash does not match its
    recomputed value, or the chain linkage (prev_record_hash) is broken."""


def verify_chain(records: list[EvidenceRecord]) -> None:
    """Raises HashChainError on the first broken link, with the offending
    event_id in the message. Silent success means the entire log is
    provably untampered since it was written."""
    expected_prev = GENESIS_HASH
    for record in records:
        if record.prev_record_hash != expected_prev:
            raise HashChainError(
                f"event_id={record.event_id}: prev_record_hash mismatch "
                f"(expected {expected_prev}, got {record.prev_record_hash})"
            )
        recomputed = compute_record_hash(record)
        if recomputed != record.record_hash:
            raise HashChainError(
                f"event_id={record.event_id}: record_hash mismatch "
                f"(stored {record.record_hash}, recomputed {recomputed}) — "
                "record was modified after being written"
            )
        expected_prev = record.record_hash
