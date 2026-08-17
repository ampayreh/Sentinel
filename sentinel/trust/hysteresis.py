"""Asymmetric escalate-fast / de-escalate-slow hysteresis.

Escalation (moving to a less permissive state) fires on k-of-n consecutive
ticks whose target state is at least as severe as the candidate — small k,
small n, so a real problem is acted on within a few hundred milliseconds.

De-escalation (moving to a more permissive state) requires BOTH a minimum
dwell time in the current state AND a run of `deescalate_clean_n`
consecutive ticks whose target state is strictly better than current. This
is what stops a monitor sitting exactly on a threshold from flapping
authority back and forth every tick.

This class is pure state + pure transition function — no I/O, no
wall-clock. It is driven one tick at a time by the coordinator, which
supplies the already-computed instantaneous "target state" for this tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel.trust.states import TrustState, rank


@dataclass(frozen=True)
class HysteresisConfig:
    escalate_k: int
    escalate_n: int
    min_dwell_ticks: int
    deescalate_clean_n: int


@dataclass
class HysteresisState:
    current: TrustState = TrustState.FULL_AUTONOMY
    ticks_in_current: int = 0
    recent_targets: list[TrustState] = field(default_factory=list)  # last `escalate_n` targets
    clean_streak: int = 0  # consecutive ticks with target strictly better than current


def step(
    state: HysteresisState, target: TrustState, cfg: HysteresisConfig
) -> tuple[HysteresisState, bool, str]:
    """Advance hysteresis by one tick given this tick's instantaneous
    target state. Returns (new_state, transitioned, rule_id)."""

    recent = (state.recent_targets + [target])[-cfg.escalate_n :]

    # --- Escalation: fast ---
    worse_count = sum(1 for t in recent if rank(t) >= rank(target) and rank(t) > rank(state.current))
    if rank(target) > rank(state.current) and worse_count >= cfg.escalate_k:
        new_state = HysteresisState(
            current=target,
            ticks_in_current=0,
            recent_targets=recent,
            clean_streak=0,
        )
        return new_state, True, "ESCALATE_K_OF_N"

    # --- De-escalation: slow ---
    clean_streak = state.clean_streak + 1 if rank(target) < rank(state.current) else 0

    if (
        rank(target) < rank(state.current)
        and state.ticks_in_current >= cfg.min_dwell_ticks
        and clean_streak >= cfg.deescalate_clean_n
    ):
        # De-escalate exactly one level at a time, never jump straight to
        # FULL_AUTONOMY from HANDOFF in one tick, even if target says so —
        # authority is earned back incrementally.
        next_rank = rank(state.current) - 1
        new_current = next(s for s in TrustState if rank(s) == next_rank)
        new_state = HysteresisState(
            current=new_current,
            ticks_in_current=0,
            recent_targets=recent,
            clean_streak=0,
        )
        return new_state, True, "DEESCALATE_DWELL_AND_CLEAN_N"

    # --- No transition ---
    new_state = HysteresisState(
        current=state.current,
        ticks_in_current=state.ticks_in_current + 1,
        recent_targets=recent,
        clean_streak=clean_streak,
    )
    return new_state, False, "NO_TRANSITION"
