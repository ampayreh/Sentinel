"""Live terminal dashboard: monitor scores over time, current trust state
with transition history, the authorized-action table, and evidence
records — updating live as `sentinel run --scenario ... --dashboard` (or
`sentinel dashboard <log> --follow`) processes frames.

Built on `rich` (already a core dependency) rather than a separate web
stack, per the spec's "live terminal OR lightweight web dashboard" — a
terminal UI has zero extra runtime dependencies and works identically
over SSH on the GB10 unit itself, which a browser-based dashboard would
not without extra plumbing.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sentinel.evidence.record import EvidenceRecord

_SCORE_HISTORY_LEN = 40
_TRANSITION_HISTORY_LEN = 8

_TRUST_COLORS = {
    "FULL_AUTONOMY": "green",
    "DEGRADED": "yellow",
    "MINIMAL_RISK": "orange3",
    "HANDOFF": "red",
}


def _sparkline(values: list[float]) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return ""
    return "".join(blocks[max(0, min(8, int(v * 8)))] for v in values)


class DashboardState:
    def __init__(self) -> None:
        self.score_history: dict[str, deque[float]] = {
            mid: deque(maxlen=_SCORE_HISTORY_LEN)
            for mid in ["M1_PERCEPTION", "M2_SENSOR_AGREEMENT", "M3_TIMING", "M4_ODD_COMPLIANCE"]
        }
        self.transitions: deque[str] = deque(maxlen=_TRANSITION_HISTORY_LEN)
        self.latest: EvidenceRecord | None = None
        self.n_records = 0

    def update(self, record: EvidenceRecord) -> None:
        self.latest = record
        self.n_records += 1
        for result in record.monitor_results:
            if result.monitor_id in self.score_history:
                self.score_history[result.monitor_id].append(result.score)
        if record.trust_transition is not None:
            self.transitions.append(
                f"t={record.monotonic_ns / 1e9:6.1f}s  {record.trust_transition.from_state} -> "
                f"{record.trust_transition.to_state}  ({record.trust_transition.reason[:60]})"
            )

    def render(self) -> Group:
        if self.latest is None:
            return Group(Text("waiting for first frame..."))

        trust_state = self.latest.trust_state
        color = _TRUST_COLORS.get(trust_state, "white")

        header = Panel(
            Text(f"TRUST STATE: {trust_state}", style=f"bold {color}", justify="center"),
            subtitle=f"records={self.n_records}  event_id={self.latest.event_id}  t={self.latest.monotonic_ns / 1e9:.1f}s",
        )

        scores_table = Table(title="Monitor scores", show_lines=False)
        scores_table.add_column("Monitor")
        scores_table.add_column("Verdict")
        scores_table.add_column("Score")
        scores_table.add_column("Trend")
        for result in sorted(self.latest.monitor_results, key=lambda r: r.monitor_id):
            history = list(self.score_history.get(result.monitor_id, []))
            verdict_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "STALE": "magenta"}.get(
                result.verdict.value, "white"
            )
            scores_table.add_row(
                result.monitor_id,
                Text(result.verdict.value, style=verdict_color),
                f"{result.score:.2f}",
                _sparkline(history),
            )

        actions_table = Table(title="Authorized actions")
        actions_table.add_column("Action")
        actions_table.add_column("Status")
        actions_table.add_column("Rule")
        for decision in self.latest.policy_decisions:
            mark = Text("✓ ALLOW", style="green") if decision.decision == "ALLOW" else Text("✗ DENY", style="red")
            actions_table.add_row(decision.action_class, mark, decision.rule_id)

        transitions_panel = Panel(
            Text("\n".join(self.transitions) if self.transitions else "(no transitions yet)"),
            title="Transition history",
        )

        return Group(header, scores_table, actions_table, transitions_panel)


def run_dashboard(records: Iterator[EvidenceRecord], refresh_hz: float = 10.0) -> None:
    console = Console()
    state = DashboardState()
    min_interval = 1.0 / refresh_hz
    last_render = 0.0

    with Live(state.render(), console=console, refresh_per_second=refresh_hz) as live:
        for record in records:
            state.update(record)
            now = time.monotonic()
            if now - last_render >= min_interval:
                live.update(state.render())
                last_render = now
        live.update(state.render())
