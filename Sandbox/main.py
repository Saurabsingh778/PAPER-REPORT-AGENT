"""Phase 2 Orchestrator: Supports both Kaggle and Local Docker environments."""

import argparse
import json
import sys
from pathlib import Path

from Logger import call_logged, get_phase, link_artifact, phase, start_run
from RepoHandler.featch_repo import featch_and_save
from RepoHandler.repo_handler import generate_run_spec
from Sandbox.compare import compare
from Sandbox.matcher import collect_output, llm_match
from Sandbox.github_client import parse_repo
from Sandbox.kaggle_runner import run_on_kaggle
from Sandbox.local_runner import run_locally
from Sandbox.report import write_report


def _describe_spec(spec) -> str:
    return f"gpu={spec.hardware.gpu}, {len(spec.steps)} steps, {len(spec.claims)} claims"


def _describe_run(run_out) -> str:
    if not run_out:
        return "no results produced"
    return f"{len(run_out.get('steps', []))} steps executed"


def run_pipeline(phase1_path: str, use_local: bool = False):
    p1 = Path(phase1_path)
    extraction = json.loads(p1.read_text(encoding="utf-8"))

    meta = {"phase1_json": str(p1), "mode": "local_docker" if use_local else "kaggle"}
    with start_run("phase2_pipeline", upstream_artifact=p1, meta=meta):
        # 1. Fetch repo and data locally
        with phase("phase2_fetch", "download the code repository and data repositories"):
            print("[1/5] Fetching repositories and data...")
            call_logged(featch_and_save, str(p1), _name="featch_and_save")

            code_repos = [r["url"] for r in extraction.get("code_repos", []) if parse_repo(r["url"])]
            if not code_repos:
                print("[!] No valid code repo found.")
                get_phase().error("No valid code repo found in the phase 1 extraction")
                return

            repo_url = code_repos[0]
            owner, repo_name = parse_repo(repo_url)
            local_repo_dir = Path("./Downloads/Repo") / f"{owner}_{repo_name}"
            local_data_dir = Path("./Downloads/Data")
            get_phase().info("repository selected", repo_url=repo_url, repo_dir=str(local_repo_dir),
                             data_dir=str(local_data_dir), candidates=code_repos)

        out_dir = p1.parent / f"phase2_{p1.stem.removeprefix('phase1_')}"
        out_dir.mkdir(exist_ok=True)

        # 2. Run RepoHandler to build and validate RunSpec
        with phase("phase2_planner", "RepoHandler AI planner builds and validates the RunSpec"):
            print("[2/5] Running RepoHandler AI planner...")
            spec = call_logged(
                generate_run_spec,
                extraction_json_path=p1,
                repo_dir=local_repo_dir,
                repo_url=repo_url,
                data_dir=local_data_dir,
                log_dir=out_dir / "logs",
                _name="generate_run_spec",
                _summarize=_describe_spec,
            )
            (out_dir / "run_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
            get_phase().save_artifact("run_spec.json", out_dir / "run_spec.json")

        # 3. Fire Execution Session
        with phase("phase2_execution", "local Docker session" if use_local else "Kaggle session"):
            if use_local:
                print(f"[3/5] Firing Local Docker session (GPU requirement: {spec.hardware.gpu})...")
                run_out = call_logged(
                    run_locally,
                    spec=spec,
                    workdir=out_dir,
                    repo_dir=local_repo_dir,
                    data_dir=local_data_dir,
                    _name="run_locally",
                    _summarize=_describe_run,
                )
            else:
                print(f"[3/5] Firing Kaggle session (GPU requirement: {spec.hardware.gpu})...")
                run_out = call_logged(run_on_kaggle, spec=spec, workdir=out_dir,
                                      _name="run_on_kaggle", _summarize=_describe_run)
            get_phase().save_artifact("execution_results.json", run_out)

        # 4. Compare outputs & extract signals
        with phase("phase2_compare", "compare reproduced numbers against the paper's claims"):
            print("[4/5] Comparing reproduced results against paper claims...")
            claims = extraction["main_results"]
            raw_dir = out_dir / "local_run" / "out" / "raw" if use_local else None
            output = collect_output(run_out, raw_dir)
            get_phase().save_artifact("match_input_output.txt", output)
            matches = None
            try:
                found = call_logged(llm_match, claims, output, _name="llm_match",
                                    _summarize=lambda m: f"{sum(x.found for x in m)}/{len(m)} located")
                matches = {m.claim_index: m for m in found}
                get_phase().save_artifact("llm_matches.json", [m.model_dump() for m in found])
            except Exception as exc:  # fall back to the heuristic matcher
                print(f"[!] LLM matching failed ({exc}); using heuristic matching.")
            rows, n_ok = call_logged(
                compare, claims, run_out, matches=matches, output=output, _name="compare",
                _summarize=lambda r: f"{r[1]}/{len(r[0])} claims reproduced",
            )
            get_phase().save_artifact("comparison_rows.json", rows)

        # 5. Generate final report
        with phase("phase2_report", "write the final reproduction report"):
            print("[5/5] Writing final report...")
            unreproducible = [c.reason for c in spec.claims if c.status != "mapped"]
            report_text = call_logged(
                write_report, out_dir, rows, n_ok, unreproducible, _name="write_report",
                _log_result=False,
            )
            print(report_text)
            link_artifact(out_dir / "phase2.json", "phase2_json")
            link_artifact(out_dir / "phase2.md", "phase2_report")


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