"""Phase 2 Orchestrator: Supports both Kaggle and Local Docker environments.

Standalone:  python -m Sandbox.main <phase1.json> [--local]
Integrated:  Pipeline.pipeline_stage calls run_pipeline() with explicit folders, so every
             run keeps its own plan/execution output and its own download cache.
"""

import argparse
import json
import sys
from pathlib import Path

from Logger import call_logged, get_phase, link_artifact, phase, start_run
from RepoHandler.featch_repo import featch_and_save
from RepoHandler.repo_handler import generate_run_spec
from Sandbox.compare import compare
from Sandbox.feasibility import apply_gate, check_spec, probe_local_resources, resources_for_prompt
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


def _without_execution(claims, reasons, out_dir):
    """Rows + phase2 report for a run that never executed anything (no repo, nothing runnable).
    Every claim is 'not_attempted' and the reasons say why."""
    rows, n_ok = compare(claims, None)
    write_report(out_dir, rows, n_ok, reasons)
    return rows, n_ok


def run_pipeline(
    phase1_path: str,
    use_local: bool = False,
    *,
    out_dir=None,
    plan_dir=None,
    repo_root=None,
    data_root=None,
    plan_only: bool = False,
) -> dict:
    """Fetch -> plan -> (feasibility) -> execute (with 5x EnvirementSetup retry + latest fallback) -> compare -> report."""
    p1 = Path(phase1_path)
    extraction = json.loads(p1.read_text(encoding="utf-8"))
    claims = extraction.get("main_results", [])

    repo_root = Path(repo_root) if repo_root else Path("./Downloads/Repo")
    data_root = Path(data_root) if data_root else Path("./Downloads/Data")
    out_dir = Path(out_dir) if out_dir else p1.parent / f"phase2_{p1.stem.removeprefix('phase1_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_dir = Path(plan_dir) if plan_dir else out_dir
    plan_dir.mkdir(parents=True, exist_ok=True)

    result = {"status": "failed", "mode": "local_docker" if use_local else "kaggle",
              "repo_url": None, "spec": None, "run_out": None, "rows": [], "n_ok": 0,
              "unreproducible": [], "error": None, "out_dir": str(out_dir)}

    meta = {"phase1_json": str(p1), "mode": result["mode"]}
    with start_run("phase2_pipeline", upstream_artifact=p1, meta=meta):
        if not claims:
            reason = "The extraction found no quantitative results to reproduce."
            get_phase().warning(reason)
            result.update(status="no_claims", error=reason, unreproducible=[reason])
            write_report(out_dir, [], 0, [reason])
            return result

        # 1. Fetch repo and data locally
        with phase("phase2_fetch", "download the code repository and data repositories"):
            print("[1/5] Fetching repositories and data...")
            call_logged(featch_and_save, str(p1), str(repo_root), str(data_root), _name="featch_and_save")

            code_repos = [r["url"] for r in extraction.get("code_repos", []) if parse_repo(r["url"])]
            repo_url, local_repo_dir = None, None
            for url in code_repos:  # first repo that actually landed on disk
                owner, repo_name = parse_repo(url)
                candidate = repo_root / f"{owner}_{repo_name}"
                if candidate.is_dir():
                    repo_url, local_repo_dir = url, candidate
                    break
            if repo_url is None:
                reason = ("No code repository was found in the paper." if not code_repos else
                          f"None of the code repositories could be downloaded: {code_repos}")
                print(f"[!] {reason}")
                get_phase().error(reason)
                result.update(status="no_repo", error=reason, unreproducible=[reason])
                result["rows"], result["n_ok"] = _without_execution(claims, [reason], out_dir)
                return result

            result["repo_url"] = repo_url
            get_phase().info("repository selected", repo_url=repo_url, repo_dir=str(local_repo_dir),
                             data_dir=str(data_root), candidates=code_repos)

        resources_text = ""
        res = None
        if use_local:
            with phase("phase2_resources", "probe what this machine can run (GPU/VRAM/RAM/disk/Docker)"):
                res = call_logged(probe_local_resources, _name="probe_local_resources",
                                  _summarize=resources_for_prompt)
                resources_text = resources_for_prompt(res)
                get_phase().save_artifact("local_resources.json", res)
                print(f"[*] Local machine: {resources_text}")

        # 2. Run RepoHandler to build and validate RunSpec
        with phase("phase2_planner", "RepoHandler AI planner builds and validates the RunSpec"):
            print("[2/5] Running RepoHandler AI planner...")
            spec = call_logged(
                generate_run_spec,
                extraction_json_path=p1,
                repo_dir=local_repo_dir,
                repo_url=repo_url,
                data_dir=data_root,
                log_dir=plan_dir / "logs",
                resources=resources_text,
                _name="generate_run_spec",
                _summarize=_describe_spec,
            )
            (plan_dir / "run_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
            get_phase().save_artifact("run_spec.json", plan_dir / "run_spec.json")
            result["spec"] = spec

        # 2b. Feasibility gate: run what this machine can run, skip (and explain) the rest
        if use_local:
            with phase("phase2_feasibility", "check each planned step/claim against local resources"):
                report = call_logged(check_spec, spec, res, _name="check_spec",
                                     _summarize=lambda r: f"runnable={r['runnable']}, "
                                     f"{sum(v == 'feasible' for v in r['claims'].values())}/{len(r['claims'])} claims feasible")
                get_phase().save_artifact("feasibility.json", report)
                (plan_dir / "feasibility.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                for cid, v in report["claims"].items():
                    print(f"  claim {cid}: {v}")
                if not report["runnable"]:
                    reasons = report["global_blockers"] or [
                        f"step '{sid}': {'; '.join(p)}" for sid, p in report["steps"].items() if p]
                    print(f"[x] Nothing can run on this machine: {reasons}")
                    get_phase().error("nothing runnable locally", blockers=report["global_blockers"])
                    reason = "Nothing can run on this machine: " + "; ".join(reasons)
                    result.update(status="not_runnable", error=reason, unreproducible=reasons)
                    result["rows"], result["n_ok"] = _without_execution(claims, reasons, out_dir)
                    return result
                spec = apply_gate(spec, report)
                (plan_dir / "run_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
                result["spec"] = spec

        if plan_only:
            print("[*] --plan-only: stopping after the RunSpec; nothing was executed.")
            get_phase().info("plan only: execution skipped")
            result["rows"], result["n_ok"] = compare(claims, None)
            result.update(status="plan_only",
                          unreproducible=[c.reason for c in spec.claims if c.status != "mapped"])
            return result

        # 3. Fire Execution Session (with 5-attempt EnvirementSetup retry and latest-version fallback)
        with phase("phase2_execution", "local Docker session" if use_local else "Kaggle session"):
            if use_local:
                print(f"[3/5] Firing Local Docker session (GPU requirement: {spec.hardware.gpu})...")
                run_out = call_logged(
                    run_locally,
                    spec=spec,
                    workdir=out_dir,
                    repo_dir=local_repo_dir,
                    data_dir=data_root,
                    plan_dir=plan_dir,
                    _name="run_locally",
                    _summarize=_describe_run,
                )
            else:
                print(f"[3/5] Firing Kaggle session (GPU requirement: {spec.hardware.gpu})...")
                run_out = call_logged(run_on_kaggle, spec=spec, workdir=out_dir,
                                      _name="run_on_kaggle", _summarize=_describe_run)
            get_phase().save_artifact("execution_results.json", run_out)
            (out_dir / "execution_results.json").write_text(
                json.dumps(run_out, indent=2, default=str), encoding="utf-8")
            result["run_out"] = run_out
            result["spec"] = spec
            if not run_out:
                result["error"] = "The execution session returned no results.json."
                get_phase().warning(result["error"])

        # 4. Compare outputs & extract signals
        with phase("phase2_compare", "compare reproduced numbers against the paper's claims"):
            print("[4/5] Comparing reproduced results against paper claims...")
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
            (out_dir / "comparison_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

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

        result.update(status="completed", rows=rows, n_ok=n_ok, unreproducible=unreproducible)
        return result


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
