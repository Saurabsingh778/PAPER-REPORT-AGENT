"""Fetch the output of an already-pushed Kaggle run and write the Phase 2 report.

Use this after Ctrl+C on `Sandbox.main`, or any time a run finished while you weren't polling.

Usage:
  python -m Sandbox.collect [phase1_X.json | dir]        # waits until the kernel is done
  python -m Sandbox.collect [phase1_X.json | dir] --now  # fetch whatever exists right now
"""
import json
import sys
import time
from pathlib import Path

from .compare import compare
from .config import KAGGLE_USER
from .kaggle_runner import kaggle
from .main import find_phase1
from .report import write_report

SLUG = "repro-agent-run"


def main(arg=None, wait=True, poll=30, max_wait_min=240):
    f = find_phase1(arg)
    ex = json.loads(f.read_text())
    run_dir = f.parent / f"phase2_{f.stem.removeprefix('phase1_')}"
    ref = f"{KAGGLE_USER}/{SLUG}"

    t0 = time.time()
    while True:
        s = kaggle("kernels", "status", ref).stdout.strip()
        print(s)
        done = any(k in s for k in ("COMPLETE", "ERROR", "CANCEL"))
        if done or not wait or time.time() - t0 > max_wait_min * 60:
            break
        time.sleep(poll)

    out = run_dir / "kaggle_out"
    p = kaggle("kernels", "output", ref, "-p", str(out))
    print(p.stdout, p.stderr)
    res = out / "results.json"
    if not res.exists():
        print(f"No results.json yet in {out}. Check the log:\n  {out}/{SLUG}.log")
        return
    run_out = json.loads(res.read_text())

    plan_f = run_dir / "plan.json"
    unrep = json.loads(plan_f.read_text()).get("unreproducible", []) if plan_f.exists() else []
    rows, n_ok = compare(ex["main_results"], run_out)
    print(write_report(run_dir, rows, n_ok, unrep))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else None, wait="--now" not in sys.argv)