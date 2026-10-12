"""Entry point: a paper PDF in, a reproducibility report out.

    python main.py run paper.pdf                  # execute on Kaggle (default)
    python main.py run paper.pdf --local          # execute locally in Docker
    python main.py run paper.pdf --plan-only      # stop after the RunSpec, nothing is executed
    python main.py run --resume Outputs/<paper>/<run_id>    # reuse finished pdf/extract stages
    python main.py list                           # every run, newest first
    python main.py latest [paper-slug]            # print the newest report path

Everything for a run lands in Outputs/<paper>/<run_id>/ (see Pipeline/paths.py);
start with report.md in that folder.
"""
import argparse
import sys

from Pipeline import PreflightError, latest_run, list_runs, run_paper
from Pipeline.pdf_stage import READERS


def _print_event(ev: dict) -> None:
    if ev["status"] == "warning":
        print(f"[{ev['stage']}] warning: {ev['message']}")
    elif ev["stage"] == "preflight":
        if ev["status"] == "failed":
            print(f"[preflight] MISSING {ev['message']}")
    else:
        print(f"[{ev['stage']}] {ev['status']}" + (f": {ev['message']}" if ev["message"] else ""))


def cmd_run(args) -> int:
    if not args.pdf and not args.resume:
        print("error: give a PDF to process, or --resume <run folder>", file=sys.stderr)
        return 2
    try:
        result = run_paper(
            args.pdf,
            output_root=args.out,
            mode="local" if args.local else "kaggle",
            reader=args.reader,
            plan_only=args.plan_only,
            resume=args.resume,
            on_event=_print_event,
            skip_preflight=args.skip_preflight,
        )
    except PreflightError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    print(f"\n{result.headline}")
    print(f"Report: {result.report_md}")
    print(f"Folder: {result.run_dir}")
    return 1 if result.status == "failed" else 0


def cmd_list(args) -> int:
    runs = list_runs(args.out, args.paper)
    if not runs:
        print("no runs yet")
        return 0
    for r in runs:
        score = f"{r['reproduced']}/{r['total']}" if r["total"] is not None else "-"
        print(f"{r['created'] or '?':<26} {r['status']:<10} {score:<6} {r['path']}")
    return 0


def cmd_latest(args) -> int:
    run = latest_run(args.out, args.paper)
    if run is None:
        print("no runs yet", file=sys.stderr)
        return 1
    print(run / "report.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper PDF -> reproducibility report.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process a PDF end to end")
    r.add_argument("pdf", nargs="?", help="path to the paper PDF")
    r.add_argument("--resume", metavar="RUN_DIR", help="reuse an existing run folder's finished pdf/extract stages")
    r.add_argument("--out", metavar="DIR", help="output root (default: ./Outputs or $AGENT_OUTPUT_ROOT)")
    r.add_argument("--local", action="store_true", help="execute locally in Docker instead of Kaggle")
    r.add_argument("--reader", choices=sorted(READERS), default="marker", help="PDF reader (default: marker)")
    r.add_argument("--plan-only", action="store_true", help="stop after planning; do not execute anything")
    r.add_argument("--skip-preflight", action="store_true", help="skip the up-front key/tool checks")
    r.set_defaults(func=cmd_run)

    ls = sub.add_parser("list", help="list runs, newest first")
    ls.add_argument("paper", nargs="?", help="only this paper (its slug)")
    ls.add_argument("--out", metavar="DIR")
    ls.set_defaults(func=cmd_list)

    lt = sub.add_parser("latest", help="print the newest report path")
    lt.add_argument("paper", nargs="?", help="only this paper (its slug)")
    lt.add_argument("--out", metavar="DIR")
    lt.set_defaults(func=cmd_latest)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
