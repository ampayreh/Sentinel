"""Reader ABC: pluggable, streaming-first source adapters.

Every reader (synthetic, comma2k19, PhysicalAI, ...) implements
`__iter__` -> Iterator[HealthFrame] and must:
  - be a generator / lazy iterator, never materialize the whole source in
    memory (bounded-memory requirement)
  - emit frames with source-native timestamps only; alignment to the fixed
    tick happens downstream in `sentinel.pipeline.align`
  - be deterministic: same file/config -> byte-identical HealthFrame
    sequence (readers that involve randomness, e.g. the synthetic
    generator, must accept and use an explicit seed)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from sentinel.schemas.health_frame import HealthFrame


class Reader(ABC):
    source_name: str

    @abstractmethod
    def __iter__(self) -> Iterator[HealthFrame]:
        raise NotImplementedError
