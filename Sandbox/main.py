"""Phase 2 Orchestrator: Supports both Kaggle and Local Docker environments."""

import argparse
import json
import sys
from pathlib import Path

from RepoHandler.featch_repo import featch_and_save
from RepoHandler.repo_handler import generate_run_spec
from Sandbox.compare import compare
from Sandbox.github_client import parse_repo
from Sandbox.kaggle_runner import run_on_kaggle
from Sandbox.local_runner import run_locally
from Sandbox.report import write_report


def run_pipeline(phase1_path: str, use_local: bool = False):
    p1 = Path(phase1_path)
    extraction = json.loads(p1.read_text(encoding="utf-8"))

    # 1. Fetch repo and data locally
    print("[1/5] Fetching repositories and data...")
    featch_and_save(str(p1))

    code_repos = [r["url"] for r in extraction.get("code_repos", []) if parse_repo(r["url"])]
    if not code_repos:
        print("[!] No valid code repo found.")
        return

    repo_url = code_repos[0]
    owner, repo_name = parse_repo(repo_url)
    local_repo_dir = Path("./Downloads/Repo") / f"{owner}_{repo_name}"
    local_data_dir = Path("./Downloads/Data")

    out_dir = p1.parent / f"phase2_{p1.stem.removeprefix('phase1_')}"
    out_dir.mkdir(exist_ok=True)

    # 2. Run RepoHandler to build and validate RunSpec
    print("[2/5] Running RepoHandler AI planner...")
    spec = generate_run_spec(
        extraction_json_path=p1,
        repo_dir=local_repo_dir,
        repo_url=repo_url,
        data_dir=local_data_dir,
        log_dir=out_dir / "logs",
    )
    (out_dir / "run_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    # 3. Fire Execution Session
    if use_local:
        print(f"[3/5] Firing Local Docker session (GPU requirement: {spec.hardware.gpu})...")
        run_out = run_locally(
            spec=spec,
            workdir=out_dir,
            repo_dir=local_repo_dir,
            data_dir=local_data_dir,
        )
    else:
        print(f"[3/5] Firing Kaggle session (GPU requirement: {spec.hardware.gpu})...")
        run_out = run_on_kaggle(spec=spec, workdir=out_dir)

    # 4. Compare outputs & extract signals
    print("[4/5] Comparing reproduced results against paper claims...")
    rows, n_ok = compare(extraction["main_results"], run_out)

    # 5. Generate final report
    print("[5/5] Writing final report...")
    unreproducible = [c.reason for c in spec.claims if c.status != "mapped"]
    report_text = write_report(out_dir, rows, n_ok, unreproducible)
    print(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reproducibility agent pipeline.")
    parser.add_argument("phase1_json", help="Path to phase 1 extraction JSON file.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Execute locally in Docker with GPU rather than on Kaggle.",
    )
    args = parser.parse_args()

    run_pipeline(args.phase1_json, use_local=args.local)