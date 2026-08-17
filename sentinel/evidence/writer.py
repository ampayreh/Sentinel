"""Append-only-WITHIN-A-RUN JSONL evidence writer with hash chaining.

One EvidenceWriter owns exactly one log file for the lifetime of a run,
and starts a FRESH hash chain (genesis) each time it is constructed —
"append-only" describes how records accumulate during a single run (never
rewritten or reordered), not that re-pointing a new run at an existing
path continues its chain. Opening in append mode while resetting the
in-memory chain state to genesis would silently corrupt an existing file
into two concatenated, mutually-inconsistent chains (verify_chain would
correctly reject it at the seam) — this was caught by
`sentinel demo` writing to a fixed default path across repeated
invocations. Each run therefore truncates its target path by default;
pass `resume=True` only if you specifically want to continue an existing
chain (not currently used anywhere in this codebase, but the flag exists
for that explicit, deliberate opt-in rather than making it the silent
default).
"""

from __future__ import annotations

from pathlib import Path

from sentinel.evidence.hash_chain import GENESIS_HASH, compute_record_hash
from sentinel.evidence.record import EvidenceRecord
from sentinel.logging import get_logger

logger = get_logger(__name__)


class EvidenceWriter:
    def __init__(self, path: str | Path, flush_every_n_records: int = 1, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every_n_records
        self._unflushed = 0
        self._last_hash = GENESIS_HASH
        self._next_event_id = 0
        if resume:
            raise NotImplementedError(
                "resume=True would need to read the existing file's last "
                "record to recover _last_hash/_next_event_id before "
                "appending — not implemented; every current caller wants a "
                "fresh chain per run."
            )
        self._fh = self.path.open("w", encoding="utf-8")

    def build_and_append(self, record_without_hash: EvidenceRecord) -> EvidenceRecord:
        """Takes a record with placeholder event_id/prev_record_hash/record_hash,
        fills in the real chain fields, writes it, and returns the final record."""
        record = record_without_hash.model_copy(
            update={
                "event_id": self._next_event_id,
                "prev_record_hash": self._last_hash,
                "record_hash": "",
            }
        )
        final_hash = compute_record_hash(record)
        record = record.model_copy(update={"record_hash": final_hash})

        self._fh.write(record.model_dump_json() + "\n")
        self._unflushed += 1
        if self._unflushed >= self._flush_every:
            self._fh.flush()
            self._unflushed = 0

        self._last_hash = final_hash
        self._next_event_id += 1
        return record

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> EvidenceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_log(path: str | Path) -> list[EvidenceRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(EvidenceRecord.model_validate_json(line))
    return records
