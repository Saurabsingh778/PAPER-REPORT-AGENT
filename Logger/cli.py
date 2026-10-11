"""Browse logs from the terminal.

    python -m Logger.cli list                      recent runs
    python -m Logger.cli show [RUN]                summary of a run (default: latest)
    python -m Logger.cli phase RUN NAME [--tail N] raw log of one phase (name = substring)
    python -m Logger.cli session RUN [--tail N]    local Docker session log
    python -m Logger.cli chain [RUN]               upstream/downstream runs (PDF -> extract -> reproduce)

RUN is a run id, any unique part of one, or 'latest'.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import LogConfig
from .summary import build_summary_md
from .utils import human_duration
from .writers import read_json


def _run_dirs(cfg: LogConfig) -> list[Path]:
    root = cfg.runs_dir()
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)


def resolve_run(ref: str | None, cfg: LogConfig) -> Path:
    runs = _run_dirs(cfg)
    if not runs:
        raise SystemExit(f"No runs found under {cfg.runs_dir()}")
    if ref in (None, "latest"):
        return runs[-1]
    exact = [r for r in runs if r.name == ref]
    if exact:
        return exact[0]
    matches = [r for r in runs if ref in r.name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"No run matches '{ref}'")
    names = "\n  ".join(m.name for m in matches)
    raise SystemExit(f"'{ref}' is ambiguous:\n  {names}")


def _print_tail(path: Path, tail: int | None) -> None:
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in (lines[-tail:] if tail else lines):
        print(line)


def cmd_list(cfg: LogConfig, args: argparse.Namespace) -> None:
    runs = _run_dirs(cfg)[-args.n:]
    if not runs:
        print(f"No runs under {cfg.runs_dir()}")
        return
    print(f"{'run id':<58} {'status':<16} {'duration':>9}  upstream")
    for r in reversed(runs):
        m = read_json(r / "manifest.json", {}) or {}
        print(f"{r.name:<58} {str(m.get('status')):<16} {human_duration(m.get('duration_s')):>9}  "
              f"{m.get('upstream_run_id') or '-'}")


def cmd_show(cfg: LogConfig, args: argparse.Namespace) -> None:
    print(build_summary_md(resolve_run(args.run, cfg)))


def cmd_phase(cfg: LogConfig, args: argparse.Namespace) -> None:
    run = resolve_run(args.run, cfg)
    phases = [p for p in sorted((run / "phases").iterdir()) if args.name in p.name]
    if not phases:
        raise SystemExit(f"No phase matching '{args.name}' in {run.name}")
    if len(phases) > 1:
        raise SystemExit("Ambiguous phase: " + ", ".join(p.name for p in phases))
    _print_tail(phases[0] / ("events.jsonl" if args.events else "phase.log"), args.tail)


def cmd_session(cfg: LogConfig, args: argparse.Namespace) -> None:
    run = resolve_run(args.run, cfg)
    target = run / "local_session" / ("docker/container_stdout.log" if args.container else "session.log")
    _print_tail(target, args.tail)


def cmd_chain(cfg: LogConfig, args: argparse.Namespace) -> None:
    run = resolve_run(args.run, cfg)
    manifests = {r.name: read_json(r / "manifest.json", {}) or {} for r in _run_dirs(cfg)}

    up = []
    cur = run.name
    while manifests.get(cur, {}).get("upstream_run_id"):
        cur = manifests[cur]["upstream_run_id"]
        up.append(cur)
        if len(up) > 20:
            break

    def children(name: str, depth: int = 0) -> None:
        for rid, m in manifests.items():
            if m.get("upstream_run_id") == name and depth < 20:
                print(f"{'  ' * (depth + 1)}-> {rid}  [{m.get('status')}]")
                children(rid, depth + 1)

    for i, rid in enumerate(reversed(up)):
        print(f"{'  ' * i}{rid}  [{manifests.get(rid, {}).get('status')}]")
    print(f"{'  ' * len(up)}{run.name}  [{manifests[run.name].get('status')}]  <- selected")
    children(run.name, len(up))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m Logger.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="log root (default: <project>/Logs or $AGENT_LOG_ROOT)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("-n", type=int, default=20)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show")
    p.add_argument("run", nargs="?")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("phase")
    p.add_argument("run")
    p.add_argument("name")
    p.add_argument("--tail", type=int)
    p.add_argument("--events", action="store_true", help="show events.jsonl instead of phase.log")
    p.set_defaults(fn=cmd_phase)

    p = sub.add_parser("session")
    p.add_argument("run", nargs="?")
    p.add_argument("--tail", type=int)
    p.add_argument("--container", action="store_true", help="show raw container output")
    p.set_defaults(fn=cmd_session)

    p = sub.add_parser("chain")
    p.add_argument("run", nargs="?")
    p.set_defaults(fn=cmd_chain)

    args = parser.parse_args(argv)
    cfg = LogConfig(root=Path(args.root)) if args.root else LogConfig()
    args.fn(cfg, args)


if __name__ == "__main__":
    main(sys.argv[1:])
