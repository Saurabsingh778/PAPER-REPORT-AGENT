"""Phase 2 entry point.

Usage:
  python -m Sandbox.main                      # latest phase1_*.json in Model_outputs
  python -m Sandbox.main <path/to/phase1_X.json>
  python -m Sandbox.main <dir containing phase1_*.json>
"""
import json
import sys
from pathlib import Path

from .compare import compare
from .github_client import inspect_repo, parse_repo
from .kaggle_runner import run_on_kaggle
from .planner import make_plan
from .report import write_report

DEFAULT_DIR = Path("/home/saurab/research-paper-agent/Model/Model_outputs")


def find_phase1(arg=None):
    p = Path(arg) if arg else DEFAULT_DIR
    if p.is_file():
        return p
    files = sorted(p.glob("phase1_*.json"))
    if not files:
        raise FileNotFoundError(f"no phase1_*.json in {p}")
    return files[-1]  # timestamped names sort chronologically


def main(arg=None):
    f = find_phase1(arg)
    print(f"Using {f.name}")
    ex = json.loads(f.read_text())

    stamp = f.stem.removeprefix("phase1_")
    run_dir = f.parent / f"phase2_{stamp}"
    run_dir.mkdir(exist_ok=True)

    # keep only complete, verified repo URLs (drops stubs like "https://github.com/")
    repos = [r["url"] for r in ex["code_repos"] if parse_repo(r["url"]) and r["verified"]]
    if not repos:
        print("No usable code repo -> every claim is unreproducible from code.")
        return

    repo = inspect_repo(repos[0])
    print("Kaggle datasets the scripts expect:", repo["kaggle_datasets"] or "none")
    plan = make_plan(ex["main_results"], repo)
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2))

    run_out = run_on_kaggle(plan, repos[0], run_dir, datasets=repo["kaggle_datasets"])
    rows, n_ok = compare(ex["main_results"], run_out)
    print(write_report(run_dir, rows, n_ok, plan.unreproducible))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)