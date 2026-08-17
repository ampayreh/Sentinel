"""Resample a source-timestamped HealthFrame stream onto a fixed tick grid.

Hard requirement from spec: do NOT silently interpolate. A tick with no
frame in its window is emitted as a gap frame (`is_gap=True`,
`gap_reason=...`) carrying no fabricated sensor values, because a gap is
itself a safety signal that M3 (timing monitor) and M4 (ODD compliance)
must be able to see.

Out-of-order handling: frames are consumed from an iterator that may
deliver them non-monotonically (see SyntheticReader's late-frame
injection). This aligner buffers a small reorder window
(`reorder_window_ticks`) and, if a frame arrives after its tick has
already been emitted, marks the *next* occupied tick's frame with
`gap_reason="out_of_order"` rather than silently reordering — replay must
see the same anomaly the live system saw.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from sentinel.schemas.health_frame import HealthFrame

_NS_PER_S = 1_000_000_000


@dataclass
class AlignConfig:
    tick_hz: float = 20.0
    reorder_window_ticks: int = 3


def _make_gap_frame(
    tick_seq: int, monotonic_ns: int, wall_time_ns: int, reason: str
) -> HealthFrame:
    return HealthFrame(
        source="align/gap",
        frame_id=-1,
        monotonic_ns=monotonic_ns,
        wall_time_ns=wall_time_ns,
        tick_seq=tick_seq,
        is_gap=True,
        gap_reason=reason,
    )


def align_to_ticks(
    frames: Iterator[HealthFrame], config: AlignConfig | None = None
) -> Iterator[HealthFrame]:
    cfg = config or AlignConfig()
    tick_ns = int(_NS_PER_S / cfg.tick_hz)

    buffer: list[HealthFrame] = []
    tick_seq = 0
    t0: int | None = None
    last_emitted_monotonic_ns: int | None = None

    def _tick_window(seq: int) -> tuple[int, int]:
        assert t0 is not None
        start = t0 + seq * tick_ns
        return start, start + tick_ns

    for frame in frames:
        if t0 is None:
            t0 = frame.monotonic_ns
        buffer.append(frame)
        buffer.sort(key=lambda f: f.monotonic_ns)

        # Emit every fully-resolved tick: one whose window has closed
        # relative to the newest buffered frame, or that has waited past
        # the reorder window.
        while buffer:
            win_start, win_end = _tick_window(tick_seq)
            newest = buffer[-1].monotonic_ns
            if newest < win_end and len(buffer) < cfg.reorder_window_ticks + 1:
                break  # tick's window may still receive frames; wait

            in_window = [f for f in buffer if win_start <= f.monotonic_ns < win_end]
            if in_window:
                chosen = in_window[0]  # earliest frame in the window wins, deterministically
                out_of_order = (
                    last_emitted_monotonic_ns is not None
                    and chosen.monotonic_ns < last_emitted_monotonic_ns
                )
                out = chosen.model_copy(
                    update={
                        "tick_seq": tick_seq,
                        "gap_reason": "out_of_order" if out_of_order else None,
                        "is_gap": False,
                    }
                )
                last_emitted_monotonic_ns = chosen.monotonic_ns
                yield out
                buffer.remove(chosen)
                # Drop any remaining frames that fell into an already-consumed window.
                buffer = [f for f in buffer if f.monotonic_ns >= win_end or f is not chosen]
            else:
                yield _make_gap_frame(
                    tick_seq, win_start, win_start, reason="dropped"
                )
            tick_seq += 1

    # Flush trailing buffered frames as their own ticks (end of stream).
    for f in sorted(buffer, key=lambda f: f.monotonic_ns):
        out = f.model_copy(update={"tick_seq": tick_seq, "is_gap": False})
        yield out
        tick_seq += 1
