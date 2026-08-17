"""Terminal User Interface (TUI) for Sentinel Runtime Assurance.

Uses Rich for high-density, low-latency display on NVIDIA GB10 / DGX Spark consoles.
"""
from __future__ import annotations
import time
from typing import Any
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress
from rich.live import Live

from sentinel.types import TrustState, ActionVerdict, MonitorStatus


def create_sentinel_layout() -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="trust_bar", size=5),
        Layout(name="monitors", ratio=1),
        Layout(name="policy", size=8),
    )
    layout["right"].split_column(
        Layout(name="telemetry", size=8),
        Layout(name="advisor", ratio=1),
    )
    return layout


def render_dashboard(state: dict[str, Any]) -> Layout:
    layout = create_sentinel_layout()
    trust = state.get("latest_trust")
    policy = state.get("latest_policy")
    monitors = state.get("latest_monitors", [])
    llm = state.get("latest_llm")
    snap_count = state.get("snap_count", 0)
    fps = state.get("fps", 0.0)

    # Header
    title = Text(" 🛡️  SENTINEL // RUNTIME ASSURANCE FOR AI-NATIVE AUTONOMY ", style="bold white on #1a1a2e")
    info = Text(f"GB10 Grace Blackwell | Frames: {snap_count} | {fps:.1f} FPS | {time.strftime('%H:%M:%S')}", style="cyan on #1a1a2e")
    header_table = Table.grid(expand=True)
    header_table.add_column(justify="left")
    header_table.add_column(justify="right")
    header_table.add_row(title, info)
    layout["header"].update(Panel(header_table, style="on #1a1a2e"))

    # Trust Bar Panel
    curr_state = trust.state if trust else TrustState.ALLOW
    state_style = {
        TrustState.ALLOW: "bold green on #0a290a",
        TrustState.DEGRADE: "bold yellow on #3d3000",
        TrustState.HANDOFF: "bold white on #4a0000",
    }.get(curr_state, "bold white")
    
    trust_text = Text(f"\n  CURRENT ASSURANCE STATE:  [{curr_state.value}]  ", style=state_style)
    layout["trust_bar"].update(Panel(trust_text, title="Trust Coordinator Status", border_style="bright_blue"))

    # Monitors Table
    mon_table = Table(expand=True, box=None)
    mon_table.add_column("Monitor", style="bold cyan")
    mon_table.add_column("Status", justify="center")
    mon_table.add_column("Confidence", justify="right")
    mon_table.add_column("Diagnostic Evidence")

    for m in monitors:
        s_text = {
            MonitorStatus.PASS: Text("PASS", style="bold green"),
            MonitorStatus.WARN: Text("WARN", style="bold yellow"),
            MonitorStatus.FAIL: Text("FAIL", style="bold red"),
        }.get(m.status, Text(m.status.value))
        
        conf_bar = f"{int(m.confidence * 100)}%"
        ev_summary = ", ".join(f"{k}: {v}" for k, v in list(m.evidence.items())[:3] if k != "raw")
        mon_table.add_row(m.name, s_text, conf_bar, ev_summary[:45])

    layout["monitors"].update(Panel(mon_table, title="Monitor Bank (Parallel Evaluators)", border_style="cyan"))

    # Policy Actions
    pol_table = Table(expand=True, box=None)
    pol_table.add_column("Action", style="bold white")
    pol_table.add_column("Verdict", justify="center")
    
    if policy:
        for action, verdict in policy.actions.items():
            v_style = "bold green" if verdict == ActionVerdict.PERMIT else "bold red"
            v_icon = "✔ PERMIT" if verdict == ActionVerdict.PERMIT else "✖ BLOCK"
            pol_table.add_row(action.upper(), Text(v_icon, style=v_style))
            
    layout["policy"].update(Panel(pol_table, title="Deterministic Policy Gate", border_style="magenta"))

    # Telemetry HUD
    tele_table = Table.grid(expand=True, padding=(0, 1))
    tele_table.add_column(style="bold white")
    tele_table.add_column(style="yellow")
    
    last_snap = state.get("latest_snapshot")
    if last_snap:
        spd = last_snap.can.get("speed", 0.0) * 3.6 if last_snap.can else 0.0
        steer = last_snap.can.get("steering", 0.0) if last_snap.can else 0.0
        lat = last_snap.gnss.get("lat", 0.0) if last_snap.gnss else 0.0
        lon = last_snap.gnss.get("lon", 0.0) if last_snap.gnss else 0.0
        tele_table.add_row("Vehicle Speed:", f"{spd:.1f} km/h")
        tele_table.add_row("Steering Angle:", f"{steer:+.2f}°")
        tele_table.add_row("GNSS Coordinate:", f"{lat:.4f}, {lon:.4f}")
    layout["telemetry"].update(Panel(tele_table, title="Vehicle Sensor HUD", border_style="green"))

    # LLM Advisor Panel
    desc = llm.scene_desc if llm else "Analyzing continuous scene context via Nemotron 3.5 Lightning..."
    tags = f"Tags: {', '.join(llm.context_tags)}" if (llm and llm.context_tags) else "Tags: nominal"
    adv_text = Text(f"{desc}\n\n", style="italic white")
    adv_text.append(Text(tags, style="bold cyan"))
    layout["advisor"].update(Panel(adv_text, title="LLM Cognitive Advisor (Nemotron 3.5)", border_style="bright_yellow"))

    # Footer
    footer_text = Text("Press Ctrl+C to stop Sentinel. Evidence log written to ./evidence/*.jsonl", style="dim white")
    layout["footer"].update(Panel(footer_text, border_style="dim"))

    return layout
