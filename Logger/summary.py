"""Render a run's manifest.json + main.jsonl as a readable summary.md.

Works from files on disk only, so the CLI can regenerate it for any past run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import human_duration, truncate
from .writers import read_json, read_jsonl


def _cell(value: Any, limit: int = 90) -> str:
    text = "" if value is None else str(value)
    return truncate(text.replace("|", "\\|").replace("\n", " "), limit)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def build_summary_md(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json", {}) or {}
    records = list(read_jsonl(run_dir / "main.jsonl"))
    env = manifest.get("environment", {})

    lines = [
        f"# Run `{manifest.get('run_id', run_dir.name)}`",
        "",
        f"- **Entry:** {manifest.get('entry')}",
        f"- **Status:** {manifest.get('status')}",
        f"- **Started:** {manifest.get('started')}",
        f"- **Duration:** {human_duration(manifest.get('duration_s'))}",
        f"- **Upstream run:** {manifest.get('upstream_run_id') or '-'}",
        f"- **Command:** `{' '.join(env.get('argv', []))}`",
        f"- **Git commit:** {env.get('git_commit') or '-'}  |  Python {env.get('python')}  |  {env.get('host')}",
        "",
        "## Phases",
        "",
    ]
    phases = manifest.get("phases", [])
    lines += _table(
        ["#", "phase", "status", "duration", "calls", "failed", "warn", "err", "folder"],
        [[p.get("index"), p.get("name"), p.get("status"), human_duration(p.get("duration_s")),
          p.get("calls"), p.get("calls_failed"), p.get("warnings"), p.get("errors"),
          p.get("dir")] for p in phases],
    )

    sessions = manifest.get("sessions", [])
    if sessions:
        lines += ["", "## Execution sessions", ""]
        rows = []
        for s in sessions:
            res = s.get("results") or {}
            docker = s.get("docker") or {}
            gpu = s.get("gpu") or {}
            rows.append([s.get("kind"), s.get("status"), human_duration(s.get("duration_s")),
                         docker.get("returncode"),
                         f"{res.get('steps_ok', '-')}/{len(res.get('steps', [])) if res else '-'}",
                         "on" if gpu.get("enabled") else "off", s.get("dir")])
        lines += _table(["kind", "status", "duration", "docker rc", "steps ok", "gpu", "folder"], rows)

    calls = [r for r in records if r.get("type") == "call"]
    lines += ["", f"## Calls ({len(calls)})", ""]
    if calls:
        lines += _table(
            ["id", "phase", "call", "status", "duration", "result / error"],
            [[c.get("call_id"), c.get("phase"), c.get("name"), c.get("status"),
              human_duration(c.get("duration_s")), c.get("error") or c.get("result")] for c in calls],
        )
    else:
        lines.append("_no instrumented calls_")

    problems = [r for r in records if r.get("level") in ("WARNING", "ERROR")]
    lines += ["", f"## Warnings and errors ({len(problems)})", ""]
    if problems:
        for r in problems:
            lines.append(f"- `{r.get('level')}` [{r.get('phase')}] {truncate(r.get('msg', ''), 220)}")
    else:
        lines.append("_none_")

    outputs = manifest.get("outputs", [])
    if outputs:
        lines += ["", "## Outputs", ""]
        lines += [f"- [{o.get('kind')}] {o.get('path')}" for o in outputs]

    lines += [
        "",
        "## Where to look",
        "",
        "- `main.log` / `main.jsonl` - this summary, one line per call",
        "- `phases/<NN>_<name>/phase.log` - raw output of each phase; `events.jsonl` has full call arguments/results",
        "- `phases/<NN>_<name>/artifacts/` - raw prompts, model responses, copied files",
        "- `local_session/` - the local Docker layer (container output, per-step results, raw command logs)",
        "",
    ]
    return "\n".join(lines)
