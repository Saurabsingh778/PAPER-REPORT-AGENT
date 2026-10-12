"""Self-test for the PDF -> report integration. No API keys, Docker, Kaggle or network needed:
the slow/external parts (PDF conversion, Gemini, Kaggle) are replaced by fakes, everything else
(folder layout, manifest, Logger, report, index, resume, the Sandbox orchestration) is real.

    python -m Pipeline.selftest
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("API_KEY", "selftest-key")  # the Gemini client is built on import; never called here

from Pipeline import runner  # noqa: E402
from Pipeline.index import latest_run, list_runs  # noqa: E402
from Pipeline.pipeline_stage import ReproductionResult  # noqa: E402
from Pipeline.preflight import PreflightError  # noqa: E402
from Pipeline.utils import read_json  # noqa: E402
from Schema.RunSpec import ClaimMap, Evidence, Hardware, RepoRef, RunSpec, Signal, Step  # noqa: E402

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


# ----------------------------------------------------------------------------- fixtures
CLAIMS = [
    {"claim": "Model reaches high accuracy.", "metric": "accuracy", "value": "95.10 \\pm 1.02\\%",
     "system_or_dataset": "toy set", "compared_to": None, "source": "Abstract", "verified": True},
    {"claim": "AUC is near perfect.", "metric": "AUC", "value": "0.9935",
     "system_or_dataset": "toy set", "compared_to": None, "source": "Table 1", "verified": True},
]
EXTRACTION = {"main_results": CLAIMS,
              "code_repos": [{"url": "https://github.com/o/r", "note": "code", "verified": True}],
              "data_links": [], "availability_statement": None}


def make_spec() -> RunSpec:
    return RunSpec(
        repo=RepoRef(url="https://github.com/o/r", commit="a" * 40),
        decided_by="readme",
        hardware=Hardware(gpu="none", reason="small CPU model"),
        steps=[Step(id="train", command="python train.py", covers=[1],
                    signals=[Signal(claim_id=1, metric="accuracy", line_pattern=r"acc=([0-9.]+)")],
                    gpu=False, timeout_min=5, confidence="high",
                    evidence=[Evidence(kind="readme", text="README: python train.py")])],
        claims=[ClaimMap(claim_id=1, status="mapped", step_ids=["train"]),
                ClaimMap(claim_id=2, status="unreproducible", reason="AUC is not printed by any script")],
        confidence="high",
    )


def fake_pdf(dir_: Path, name: str = "My Test Paper.pdf") -> Path:
    p = dir_ / name
    p.write_bytes(b"%PDF-1.4\n% selftest\n")
    return p


class Calls:
    """Counts stage calls and lets a test make one stage fail."""

    def __init__(self):
        self.convert = self.extract = self.reproduce = 0
        self.fail_extract = False


def install_fakes(calls: Calls, plan_only_status: bool = False):
    def convert(pdf, out_dir, reader="marker"):
        calls.convert += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paper.md").write_text("# Toy Glass Paper\n\n" + "body " * 100, encoding="utf-8")
        return {"markdown": out_dir / "paper.md", "chars": 600, "images": 0}

    def extract(pdf_dir, out_dir):
        calls.extract += 1
        if calls.fail_extract:
            raise RuntimeError("Gemini unavailable")
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "extraction.json"
        p.write_text(json.dumps(EXTRACTION), encoding="utf-8")
        return {"extraction_json": p, "claims": 2, "repos": 1, "data_links": 0}

    def reproduce(extraction_json, layout, *, use_local, plan_only=False):
        calls.reproduce += 1
        spec = make_spec()
        layout.run_spec_json.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        if plan_only:
            rows = [{"claim": c["claim"], "metric": c["metric"], "paper": c["value"], "found": None,
                     "matched_line": None, "status": "not_attempted", "phase1_verified": True} for c in CLAIMS]
            return ReproductionResult("plan_only", "kaggle", "https://github.com/o/r", rows, 0, spec.model_dump())
        rows = [
            {"claim": CLAIMS[0]["claim"], "metric": "accuracy", "paper": CLAIMS[0]["value"], "found": 95.3,
             "matched_line": "acc=95.3", "status": "reproduced", "phase1_verified": True,
             "match_reasoning": "same cross-validated mean"},
            {"claim": CLAIMS[1]["claim"], "metric": "AUC", "paper": CLAIMS[1]["value"], "found": None,
             "matched_line": None, "status": "not_attempted", "phase1_verified": True},
        ]
        return ReproductionResult(
            "completed", "kaggle", "https://github.com/o/r", rows, 1, spec.model_dump(),
            [{"id": "train", "command": "python train.py", "rc": 0, "seconds": 12}],
            ["AUC is not printed by any script"])

    runner.convert_pdf, runner.extract_paper, runner.run_reproduction = convert, extract, reproduce


# ----------------------------------------------------------------------------- tests
def test_happy_path(tmp: Path) -> Path:
    print("happy path")
    calls = Calls()
    install_fakes(calls)
    events: list[dict] = []
    res = runner.run_paper(fake_pdf(tmp), output_root=tmp / "out", skip_preflight=True, on_event=events.append)
    run = res.run_dir

    check(res.status == "completed" and (res.reproduced, res.total) == (1, 2), "1/2 claims, status completed")
    check(run.parent.name == "my-test-paper", "folder is Outputs/<slug>/<run_id>")
    for rel in ("report.md", "report.json", "manifest.json", "00_input/My Test Paper.pdf",
                "01_pdf/paper.md", "02_extraction/extraction.json", "03_plan/run_spec.json"):
        check((run / rel).is_file(), f"has {rel}")
    check((run / "logs" / "runs").is_dir(), "Logger output lives inside the run folder")

    m = read_json(run / "manifest.json")
    check(m["status"] == "completed" and all(s["status"] == "done" for s in m["stages"].values()),
          "manifest: every stage done")
    check(m["paper_title"] == "Toy Glass Paper", "title taken from the converted paper")
    check(len(m["input"]["sha256"]) == 64, "input PDF hash recorded")

    rj = read_json(run / "report.json")
    check(rj["summary"]["reproduced"] == 1 and rj["claims"][1]["status"] == "not_attempted", "report.json claim rows")
    check(rj["claims"][1]["planner_status"] == "unreproducible", "planner verdict merged into the claim")
    md = (run / "report.md").read_text(encoding="utf-8")
    check("1 of 2 claims reproduced (50%)" in md and "Reproduced" in md and "python train.py" in md,
          "report.md headline, claim table, execution steps")

    idx = read_json(tmp / "out" / "index.json")
    check(len(idx["runs"]) == 1 and idx["runs"][0]["report"].endswith("report.md"), "index.json lists the run")
    latest = tmp / "out" / "my-test-paper" / "latest"
    check(latest.exists() and latest.resolve() == run.resolve(), "<slug>/latest points at the run")
    check(latest_run(tmp / "out") == run, "latest_run() finds it")
    check(any(e["stage"] == "pdf" and e["status"] == "done" for e in events), "progress events emitted")
    return run


def test_failure_then_resume(tmp: Path) -> None:
    print("failure keeps a report, resume reuses finished stages")
    calls = Calls()
    calls.fail_extract = True
    install_fakes(calls)
    res = runner.run_paper(fake_pdf(tmp, "Second.pdf"), output_root=tmp / "out", skip_preflight=True)
    check(res.status == "failed" and res.failed_stage == "extract", "extract failure -> status failed")
    check(res.report_md.is_file() and "failed at stage 'extract'" in res.report_md.read_text(encoding="utf-8"),
          "a report is still written and says where it stopped")
    m = read_json(res.run_dir / "manifest.json")
    check(m["stages"]["pdf"]["status"] == "done" and m["stages"]["extract"]["status"] == "failed"
          and m["stages"]["pipeline"]["status"] == "pending", "manifest shows exactly where it stopped")

    calls.fail_extract = False
    res2 = runner.run_paper(resume=res.run_dir, skip_preflight=True)
    check(res2.run_dir == res.run_dir and res2.status == "completed", "resume continues in the same folder")
    check(calls.convert == 1, "PDF was not converted again")
    m2 = read_json(res.run_dir / "manifest.json")
    check(m2["stages"]["pdf"].get("reused") is True and m2["stages"]["extract"]["status"] == "done",
          "manifest marks the reused stage")


def test_plan_only(tmp: Path) -> None:
    print("plan only")
    calls = Calls()
    install_fakes(calls)
    res = runner.run_paper(fake_pdf(tmp, "Third.pdf"), output_root=tmp / "out", plan_only=True, skip_preflight=True)
    check(res.status == "plan_only" and "Plan only" in res.headline, "plan-only run is labelled as such")


def test_preflight(tmp: Path) -> None:
    print("preflight")
    before = set((tmp / "out").glob("*")) if (tmp / "out").exists() else set()
    try:
        runner.run_paper(tmp / "missing.pdf", output_root=tmp / "out")
        check(False, "missing PDF raises PreflightError")
    except PreflightError as exc:
        check("pdf" in str(exc), "missing PDF raises PreflightError")
    after = set((tmp / "out").glob("*")) if (tmp / "out").exists() else set()
    check(before == after, "nothing is created when preflight fails")


def test_listing(tmp: Path) -> None:
    print("listing")
    runs = list_runs(tmp / "out")
    check(len(runs) == 3 and runs[0]["created"] >= runs[-1]["created"], "list_runs: newest first")
    check(len(list_runs(tmp / "out", "my-test-paper")) == 1, "list_runs: filter by paper")


def test_sandbox_orchestration(tmp: Path) -> None:
    print("Sandbox.main.run_pipeline (real orchestration, fake fetch/plan/Kaggle)")
    import Sandbox.main as sm

    ext = tmp / "extraction.json"
    ext.write_text(json.dumps(EXTRACTION), encoding="utf-8")
    kaggle_out = {"steps": [{"step_id": "train", "rc": 0, "sec": 3, "cmd": "python train.py",
                             "stdout": "epoch 1\nacc=95.3\n", "stderr": "", "covers": [1], "claim_indices": [1]}]}

    def fetch(json_path, repo_dest, data_dest):
        (Path(repo_dest) / "o_r").mkdir(parents=True, exist_ok=True)
        Path(data_dest).mkdir(parents=True, exist_ok=True)

    sm.featch_and_save = fetch
    sm.generate_run_spec = lambda **kw: make_spec()
    sm.run_on_kaggle = lambda spec, workdir, **kw: kaggle_out
    sm.llm_match = lambda claims, output: (_ for _ in ()).throw(RuntimeError("no LLM in selftest"))

    r = sm.run_pipeline(str(ext), out_dir=tmp / "exec", plan_dir=tmp / "plan",
                        repo_root=tmp / "repos", data_root=tmp / "data")
    check(r["status"] == "completed" and r["n_ok"] == 1, "completed, claim 1 reproduced by the heuristic matcher")
    check(r["rows"][1]["status"] == "not_attempted", "claim without a step stays not_attempted")
    check((tmp / "plan" / "run_spec.json").is_file() and (tmp / "exec" / "phase2.md").is_file()
          and (tmp / "exec" / "comparison_rows.json").is_file(), "plan and execution outputs in their own folders")

    r = sm.run_pipeline(str(ext), out_dir=tmp / "exec2", plan_dir=tmp / "plan2",
                        repo_root=tmp / "repos", data_root=tmp / "data", plan_only=True)
    check(r["status"] == "plan_only" and r["run_out"] is None, "plan_only never executes")

    no_repo = dict(EXTRACTION, code_repos=[])
    ext2 = tmp / "extraction_norepo.json"
    ext2.write_text(json.dumps(no_repo), encoding="utf-8")
    r = sm.run_pipeline(str(ext2), out_dir=tmp / "exec3", plan_dir=tmp / "plan3",
                        repo_root=tmp / "repos3", data_root=tmp / "data3")
    check(r["status"] == "no_repo" and (tmp / "exec3" / "phase2.md").is_file(), "no code repo -> status no_repo + report")

    ext3 = tmp / "extraction_noclaims.json"
    ext3.write_text(json.dumps(dict(EXTRACTION, main_results=[])), encoding="utf-8")
    r = sm.run_pipeline(str(ext3), out_dir=tmp / "exec4", plan_dir=tmp / "plan4",
                        repo_root=tmp / "repos4", data_root=tmp / "data4")
    check(r["status"] == "no_claims", "no extracted claims -> status no_claims")


def test_fixes_in_existing_modules(tmp: Path) -> None:
    print("fixes in existing modules")
    import RepoHandler.featch_repo as fr
    from Sandbox import kaggle_runner

    check('r["claim_indices"] = s["covers"]' in kaggle_runner.KERNEL_TEMPLATE,
          "Kaggle runner records claim_indices (compare() needs it)")

    class Bad:
        returncode, stdout, stderr = 1, "", "401 Unauthorized"

    kaggle_runner.kaggle = lambda *a: Bad()
    try:
        kaggle_runner.run_on_kaggle(make_spec(), tmp / "kg", max_wait_min=0)
        check(False, "failed kaggle push raises")
    except RuntimeError as exc:
        check("401" in str(exc), "failed kaggle push raises instead of polling for hours")

    ext = tmp / "fetch.json"
    ext.write_text(json.dumps({"code_repos": [{"url": "https://github.com/o/have"}, {"url": "https://github.com/o/new"},
                                              {"url": "https://github.com/o/dead"}],
                               "data_links": [{"url": "https://example.org/data.zip"}]}), encoding="utf-8")
    (tmp / "R" / "o_have").mkdir(parents=True)
    downloaded: list[str] = []

    def dl(url, dest_dir):
        if url.endswith("dead"):
            raise RuntimeError("404")
        downloaded.append(url)

    fr.download_repo_locally = dl
    fr.featch_and_save(str(ext), str(tmp / "R"), str(tmp / "D"))
    check(downloaded == ["https://github.com/o/new"], "existing owner_repo skipped, dead link does not stop the rest")

    from Model import gemini
    from Schema.phase_1_schema import Extraction, Link, Result

    gemini.extract = lambda prompt: Extraction(
        main_results=[Result(claim="c", metric="m", value="95.1", system_or_dataset="d", compared_to=None, source="Abstract")],
        code_repos=[Link(url="https://github.com/o/r", note="code")], data_links=[], availability_statement=None)
    pdf_dir, res_dir = tmp / "pdfout", tmp / "extract"
    pdf_dir.mkdir()
    (pdf_dir / "paper.md").write_text("accuracy 95.1 code: https://github.com/o/r", encoding="utf-8")
    out = gemini.main(str(pdf_dir), results_dir=str(res_dir))
    check((res_dir / "extraction.json").is_file() and (res_dir / "extraction.md").is_file()
          and not list(res_dir.glob("phase1_*")), "gemini.main(results_dir=...) writes only into that folder")
    check(not (pdf_dir / "extraction.json").exists() and out["data"]["code_repos"][0]["verified"] is True,
          "nothing leaks next to paper.md; verification flags kept")


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_happy_path(tmp)
        test_failure_then_resume(tmp)
        test_plan_only(tmp)
        test_preflight(tmp)
        test_listing(tmp)
        test_sandbox_orchestration(tmp)
        test_fixes_in_existing_modules(tmp)
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
