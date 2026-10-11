"""Self-test / demo: simulates a pipeline run (no Docker, no API keys needed).

    python -m Logger.selftest            # writes to a temp folder and prints the tree
    python -m Logger.selftest --keep     # same, but under Logs/_selftest so you can browse it
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from .config import DEFAULT_LOG_ROOT, LogConfig
from .decorators import call_logged, logged
from .registry import get_phase, link_artifact, local_session, phase, start_run
from .utils import human_duration
from .writers import read_json, read_jsonl

FAKE_SECRET = "sk-selftest-0123456789abcdefghij"


@logged("fake_planner", summarize=lambda s: f"{len(s['steps'])} steps")
def fake_planner(repo: str, prompt: str) -> dict:
    get_phase().save_artifact("planner_prompt.txt", prompt)
    get_phase().info("planner produced a spec", steps=2)
    return {"steps": ["train", "evaluate"], "repo": repo}


def always_fails(x: int) -> None:
    raise ValueError(f"cannot handle {x}")


def _fake_container(session, results_dir: Path) -> None:
    raw = results_dir / "raw"
    raw.mkdir(parents=True)
    (raw / "step_train.log").write_text("$ python train.py\n--- STDOUT ---\nepoch 1 acc=0.91\n")
    results = {
        "setup": [{"cmd": "pip install -q numpy", "rc": 0, "sec": 3, "stdout": "ok", "stderr": ""}],
        "patches": [{"file": "train.py", "status": "applied"}],
        "steps": [
            {"cmd": "python train.py", "rc": 0, "sec": 12, "stdout": "acc=0.91", "stderr": "",
             "step_id": "train", "covers": [1], "signals": []},
            {"cmd": "python eval.py", "rc": 1, "sec": 2, "stdout": "", "stderr": "FileNotFoundError",
             "step_id": "evaluate", "covers": [2], "signals": []},
        ],
    }
    (results_dir / "results.json").write_text(json.dumps(results))

    session.docker_command(["docker", "run", "--rm", "img", "python", "/workspace/runner.py"])
    session.container_started()
    for line in [
        "Local GPU Info: GPU 0: Fake GPU",
        "[*] Installing pip package: numpy",
        "[>>] Running Step 'train': python train.py",
        "[>>] Running Step 'evaluate': python eval.py",
        "[✓] Local execution complete. results.json generated.",
    ]:
        session.container_line(line)
    session.container_exit(0)
    session.ingest_results(results_dir / "results.json", raw)


def run_selftest(root: Path) -> Path:
    cfg = LogConfig(root=root)
    os.environ["SELFTEST_API_KEY"] = FAKE_SECRET  # picked up by the redactor
    work = root / "_work"
    work.mkdir(parents=True, exist_ok=True)

    paper = work / "paper.md"
    paper.write_text("# paper")

    # ---- run 1: pretend PDF reader
    with start_run("selftest_pdf_read", config=cfg, meta={"pdf": "paper.pdf"}) as run1:
        with phase("phase0_pdf_read"):
            print("converting pdf ...")
            link_artifact(paper, "paper_markdown")

    # ---- run 2: reproduction pipeline, chained to run 1 through paper.md
    with start_run("selftest_pipeline", config=cfg, upstream_artifact=paper,
                   meta={"local": True}) as run2:
        print("output before any phase")  # lands in 00_unphased

        with phase("phase2_planner", "plan the reproduction"):
            spec = fake_planner("owner/repo", "PROMPT " * 5000)
            print(f"leaking key {FAKE_SECRET} on purpose")  # must be redacted
            get_phase().warning("pre-flight: script train.py not found")

        with phase("phase2_execution", "local docker run"):
            with local_session(image="fake/image:1") as sess:
                sess.workspace(work, work / "repo_src", work / "repo", work / "data", [
                    {"id": "dataset", "dest": "dataset", "source": None}])
                sess.save_spec(json.dumps(spec))
                sess.save_runner("print('runner')")
                sess.gpu_check(requested=True, supported=False, returncode=1, stderr="no runtime")
                _fake_container(sess, work / "out")

        try:
            with phase("phase2_compare"):
                call_logged(always_fails, 7, _name="always_fails")
        except ValueError:
            pass  # the phase failure is logged; the run carries on

    return run2.dir, run1.dir


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def verify(run_dir: Path, upstream_dir: Path) -> None:
    print("\nVerifying", run_dir.name)
    manifest = read_json(run_dir / "manifest.json")
    main_text = (run_dir / "main.log").read_text(encoding="utf-8")
    calls = [r for r in read_jsonl(run_dir / "main.jsonl") if r.get("type") == "call"]
    planner_phase = next(run_dir.glob("phases/*phase2_planner"))
    planner_log = (planner_phase / "phase.log").read_text(encoding="utf-8")
    events = list(read_jsonl(planner_phase / "events.jsonl"))
    sess = run_dir / "local_session"
    sess_summary = read_json(sess / "session_summary.json")

    _check(manifest["upstream_run_id"] == upstream_dir.name, "run chained to the PDF-read run")
    _check(len(calls) == 2, "main log has one summary per call")
    _check("fake_planner(" in main_text and "-> OK" in main_text, "main.log shows call summaries")
    _check("always_fails(" in main_text and "-> FAILED" in main_text, "failed call is flagged")
    _check("raw: phases/" in main_text, "summary points at the raw record")
    _check(any(e.get("type") == "call" and e["name"] == "fake_planner" for e in events),
           "phase events.jsonl has the full raw call")
    big = next(e for e in events if e.get("name") == "fake_planner")["args"]["prompt"]
    _check(isinstance(big, dict) and "$artifact" in big, "oversized prompt moved to artifacts/")
    _check("output before any phase" in (next(run_dir.glob("phases/00_unphased")) / "phase.log").read_text(),
           "console output outside phases kept in 00_unphased")
    _check("converting" not in planner_log and "leaking key" in planner_log, "console output goes to the right phase")
    _check(FAKE_SECRET not in planner_log and "[REDACTED]" in planner_log, "secret redacted in phase log")
    _check(all(FAKE_SECRET not in p.read_text(errors="ignore")
               for p in run_dir.rglob("*") if p.is_file()), "secret absent from every file")
    _check(sess_summary["status"] == "ok_with_failures", "local session flags failed step")
    _check(sess_summary["results"]["steps_ok"] == 1 and sess_summary["results"]["steps_failed"] == 1,
           "session step counts correct")
    for rel in ("docker/command.txt", "docker/container_stdout.log", "docker/exit.json",
                "gpu_check.json", "workspace.json", "spec.json", "runner.py",
                "results/results.json", "results/patches.json", "raw/step_train.log",
                "results/steps/01_train.json", "results/steps/02_evaluate.stderr.log"):
        _check((sess / rel).exists(), f"local_session/{rel}")
    statuses = {p["name"]: p["status"] for p in manifest["phases"]}
    _check(statuses.get("phase2_compare") == "failed", "failed phase recorded in manifest")
    _check(manifest["status"] == "failed", "run status reflects the failed phase")
    _check((run_dir / "summary.md").exists(), "summary.md written")
    print(f"  run took {human_duration(manifest['duration_s'])}")


def main() -> None:
    keep = "--keep" in sys.argv
    if keep:
        root = DEFAULT_LOG_ROOT / "_selftest"
        root.mkdir(parents=True, exist_ok=True)
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory(prefix="logger_selftest_")
        root = Path(tmp.name)

    run_dir, upstream = run_selftest(root)
    verify(run_dir, upstream)

    print("\nFile tree:")
    for p in sorted(run_dir.rglob("*")):
        rel = p.relative_to(run_dir)
        print("  " + ("  " * (len(rel.parts) - 1)) + rel.name + ("/" if p.is_dir() else ""))
    print(f"\nAll checks passed. Logs at: {run_dir}" if keep else "\nAll checks passed.")
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
