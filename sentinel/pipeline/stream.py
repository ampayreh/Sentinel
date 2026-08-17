"""Unified streaming interface: the same code path serves offline replay
and (eventually) live ingest. A `Reader` + `AlignConfig` in, an aligned
`Iterator[HealthFrame]` out. No buffering beyond what `align_to_ticks`
needs, so memory stays bounded regardless of source length.
"""

from __future__ import annotations

from collections.abc import Iterator

from sentinel.pipeline.align import AlignConfig, align_to_ticks
from sentinel.pipeline.readers.base import Reader
from sentinel.schemas.health_frame import HealthFrame


def health_stream(reader: Reader, align_config: AlignConfig | None = None) -> Iterator[HealthFrame]:
    """Compose a reader with tick alignment into one streaming iterator."""
    yield from align_to_ticks(iter(reader), align_config)
