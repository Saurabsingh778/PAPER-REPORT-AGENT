"""Text layouts for every log file, kept in one place so the formats stay consistent."""
from __future__ import annotations

from typing import Any, Mapping

from .utils import describe, human_duration, truncate

LEVEL_WIDTH = 7


def main_line(ts: str, level: str, phase: str, text: str) -> str:
    """main.log:  2026-10-11 11:50:01 | INFO    | phase2_planner | message"""
    return f"{ts} | {level:<{LEVEL_WIDTH}} | {phase:<22} | {text}"


def phase_line(clock_ts: str, level: str, text: str) -> str:
    """phase.log:  [11:50:01.123] INFO    message"""
    return f"[{clock_ts}] {level:<{LEVEL_WIDTH}} {text}"


def console_line(clock_ts: str, stream: str, text: str) -> str:
    """Captured print()/stderr output, tagged so it is easy to grep apart from log lines."""
    tag = "ERR" if stream == "stderr" else "OUT"
    return f"[{clock_ts}] {tag:<{LEVEL_WIDTH}} | {text}"


def raw_block(title: str, body: str) -> str:
    bar = "-" * 12
    return f"{bar} BEGIN {title} {bar}\n{body.rstrip()}\n{bar} END {title} {bar}"


def format_args(args: Mapping[str, Any] | None, per_value: int = 80, total: int = 400) -> str:
    if not args:
        return ""
    parts = [f"{k}={describe(v, per_value)}" for k, v in args.items()]
    return truncate(", ".join(parts), total)


def call_summary_text(rec: Mapping[str, Any]) -> str:
    """One line per call for main.log."""
    status = "OK" if rec.get("status") == "ok" else "FAILED"
    text = f"{rec.get('name')}({rec.get('args_text', '')}) -> {status} in {human_duration(rec.get('duration_s'))}"
    if rec.get("result"):
        text += f" | {rec['result']}"
    if rec.get("error"):
        text += f" | ERROR {rec['error']}"
    if rec.get("raw_ref"):
        text += f" | raw: {rec['raw_ref']}"
    return text
