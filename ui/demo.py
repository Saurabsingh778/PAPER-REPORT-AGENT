"""Demo mode for the UI: python -m ui.server --demo

Replaces the three slow/external pieces (PDF conversion, Gemini extraction, repo fetch + plan +
Docker/Kaggle execution) with simulations that sleep, print realistic progress lines and write the
same files the real stages write. Everything else is the real code: run folders, manifest.json,
report.json/report.md, the index, resume, the Logger. The numbers in a demo report are SIMULATED
(the generated RunSpec says so in its warnings); nothing is executed.

Speed: DEMO_SPEED=2 runs twice as fast, DEMO_SPEED=0.5 half as fast.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "Pdf-Reader" / "Output"
SPEED = float(os.getenv("DEMO_SPEED", "1") or 1)

FALLBACK = {
    "main_results": [
        {"claim": "The model reaches high accuracy on the benchmark.", "metric": "accuracy", "value": "95.10 \\pm 1.02\\%",
         "system_or_dataset": "toy set", "compared_to": None, "source": "Abstract", "verified": True},
        {"claim": "AUC is near perfect.", "metric": "AUC", "value": "0.9935 \\pm 0.0021",
         "system_or_dataset": "toy set", "compared_to": None, "source": "Table 1", "verified": True},
        {"claim": "Pure topological features fail at near-random accuracy.", "metric": "accuracy", "value": "52\\%",
         "system_or_dataset": "toy set", "compared_to": None, "source": "Abstract", "verified": False},
    ],
    "code_repos": [{"url": "https://github.com/example/repo", "note": "code", "verified": True}],
    "data_links": [], "availability_statement": None,
}

# a believable spread, so every status chip in the UI gets exercised
STATUS_CYCLE = ["reproduced", "reproduced", "reproduced", "ran_but_mismatch", "reproduced",
                "reproduced", "ran_but_mismatch", "reproduced", "reproduced"]


def _sleep(sec: float) -> None:
    time.sleep(max(0.02, sec / SPEED))


def _number(text: str | None) -> float | None:
    s = re.sub(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?\s*(-?\d+)\s*\}?", r"\1e\2", text or "")
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None


def _extraction() -> dict:
    src = SAMPLE_DIR / "extraction.json"
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        if data.get("main_results"):
            return data
    except (OSError, ValueError):
        pass
    return FALLBACK


def install() -> None:
    from Pipeline import runner
    from Pipeline.pipeline_stage import ReproductionResult
    from Schema.RunSpec import (ClaimMap, DataSource, Environment, Evidence, Hardware, RepoRef,
                                RunSpec, Signal, Step)

    def convert(pdf: Path, out_dir: Path, reader: str = "marker") -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{reader}] loading layout + OCR models (simulated)")
        _sleep(1.0)
        for p in range(1, 9):
            print(f"[{reader}] page {p}/8  layout -> text -> tables")
            _sleep(0.35)
        md = SAMPLE_DIR / "paper.md"
        images = 0
        if md.is_file():
            shutil.copy2(md, out_dir / "paper.md")
            for img in sorted(SAMPLE_DIR.glob("*.jp*g"))[:12] + sorted(SAMPLE_DIR.glob("*.png"))[:4]:
                shutil.copy2(img, out_dir / img.name)
                images += 1
            chars = (out_dir / "paper.md").stat().st_size
        else:
            text = f"# {Path(pdf).stem}\n\n" + "Simulated paper body. " * 60
            (out_dir / "paper.md").write_text(text, encoding="utf-8")
            chars = len(text)
        print(f"[{reader}] wrote paper.md ({chars} chars, {images} images)")
        return {"markdown": out_dir / "paper.md", "chars": chars, "images": images}

    def extract(pdf_dir: Path, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("[gemini] extracting headline results, code repos and data links (simulated)")
        _sleep(1.2)
        data = _extraction()
        for i, c in enumerate(data["main_results"], 1):
            print(f"[gemini] claim {i}: {c.get('metric')} = {c.get('value')}")
            _sleep(0.18)
        print("[gemini] verifying each value against the paper text")
        _sleep(0.8)
        p = out_dir / "extraction.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return {"extraction_json": p, "claims": len(data["main_results"]),
                "repos": len(data.get("code_repos", [])), "data_links": len(data.get("data_links", []))}

    def reproduce(extraction_json: Path, layout, *, use_local: bool, plan_only: bool = False):
        data = json.loads(Path(extraction_json).read_text(encoding="utf-8"))
        claims = data["main_results"]
        n = len(claims)
        repo_url = (data.get("code_repos") or [{}])[0].get("url") or "https://github.com/example/repo"
        mode = "local_docker" if use_local else "kaggle"

        print("[1/5] Fetching repositories and data...")
        _sleep(1.0)
        print(f"[*] repository selected: {repo_url}")
        print("[*] cloned at commit " + "a3f9c21" + "e" * 33)
        _sleep(0.8)
        if use_local:
            print("[*] Local machine: GPU NVIDIA RTX (simulated), 16.0 GB VRAM, 32 GB RAM, Docker available")
            _sleep(0.4)

        print("[2/5] Running RepoHandler AI planner...")
        _sleep(1.4)
        chunks = [list(range(1, n + 1))[i::3] for i in range(3)] if n >= 3 else [list(range(1, n + 1))]
        chunks = [c for c in chunks if c]
        steps, claim_maps, step_defs = [], [], []
        last_unrepro = n >= 8
        covered_total = set()
        for k, ids in enumerate(chunks):
            ids = [i for i in ids if not (last_unrepro and i == n)]
            if not ids:
                continue
            sid = f"experiment_{k + 1}"
            sigs = [Signal(claim_id=i, metric=claims[i - 1].get("metric") or "metric",
                           line_pattern=rf"claim{i}[:=]\s*([0-9.eE+-]+)") for i in ids]
            steps.append(Step(
                id=sid, command=f"python scripts/run_experiment_{k + 1}.py --data {{DATA}}/glass",
                reads=["data:glass"], covers=ids, signals=sigs, gpu=True, timeout_min=45 + 15 * k,
                confidence="high" if k == 0 else "medium",
                evidence=[Evidence(kind="readme", file="README.md", text=f"README documents run_experiment_{k + 1}.py"),
                          Evidence(kind="print_line", file=f"scripts/run_experiment_{k + 1}.py",
                                   text="prints the metric on one line per fold")]))
            claim_maps += [ClaimMap(claim_id=i, status="mapped", step_ids=[sid]) for i in ids]
            covered_total.update(ids)
            step_defs.append((sid, ids))
        if last_unrepro:
            claim_maps.append(ClaimMap(claim_id=n, status="unreproducible",
                                       reason="No script in the repository prints this statistic"))
        spec = RunSpec(
            repo=RepoRef(url=repo_url, commit="a3f9c21" + "e" * 33), decided_by="readme",
            data=[DataSource(id="glass", kind="github_repo", ref=repo_url if repo_url.startswith("https://github.com/")
                             else "https://github.com/example/data", dest="glass", size_mb=140.0)],
            environment=Environment(pip=["torch", "torch_geometric", "scikit-learn"], notes="simulated plan"),
            hardware=Hardware(gpu="required", reason="GATv2 training is impractical on CPU", est_runtime_min=90, min_vram_gb=8),
            steps=steps, claims=sorted(claim_maps, key=lambda c: c.claim_id), confidence="medium",
            warnings=["DEMO RUN: this plan and its results are simulated; nothing was executed."])
        layout.run_spec_json.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        print(f"[*] RunSpec validated: {len(steps)} steps, {len(covered_total)}/{n} claims mapped")

        if plan_only:
            print("[*] --plan-only: stopping after the RunSpec; nothing was executed.")
            rows = [{"claim": c["claim"], "metric": c["metric"], "paper": c["value"], "found": None,
                     "matched_line": None, "status": "not_attempted", "phase1_verified": c.get("verified", True)}
                    for c in claims]
            return ReproductionResult("plan_only", mode, repo_url, rows, 0, spec.model_dump(),
                                      [], [c.reason for c in spec.claims if c.status != "mapped"])

        print("[*] feasibility gate")
        for cm in spec.claims:
            print(f"  claim {cm.claim_id}: " + ("runnable" if cm.status == "mapped" else "skipped (unreproducible)"))
            _sleep(0.12)

        where = "Local Docker" if use_local else "Kaggle"
        print(f"[3/5] Firing {where} session (GPU requirement: required)...")
        _sleep(1.0)
        results, run_steps = {}, []
        for sid, ids in step_defs:
            print(f"[step {sid}] $ python scripts/run_{sid}.py")
            t0 = time.time()
            for epoch in (1, 2, 3):
                print(f"  epoch {epoch}/3  loss={1.0 / (epoch + 1):.4f}")
                _sleep(0.9)
            for i in ids:
                paper = _number(claims[i - 1]["value"])
                status = STATUS_CYCLE[(i - 1) % len(STATUS_CYCLE)]
                if paper is None:
                    status = "ran_but_mismatch"
                found = None if paper is None else float(f"{paper * (1.004 if status == 'reproduced' else 0.93):.5g}")
                results[i] = (status, found)
                print(f"  claim{i}: {found}")
                _sleep(0.15)
            run_steps.append({"id": sid, "command": f"python scripts/run_{sid}.py", "rc": 0,
                              "seconds": round((time.time() - t0) * SPEED, 1)})

        print("[4/5] Comparing reproduced results against paper claims...")
        _sleep(1.0)
        rows = []
        for i, c in enumerate(claims, 1):
            status, found = results.get(i, ("not_attempted", None))
            rows.append({"claim": c["claim"], "metric": c["metric"], "paper": c["value"], "found": found,
                         "matched_line": f"claim{i}: {found}" if found is not None else None, "status": status,
                         "phase1_verified": c.get("verified", True),
                         "match_reasoning": ("Same metric and dataset; value within 3% of the paper." if status == "reproduced"
                                             else "Same metric printed, but the value differs by more than the 3% tolerance."
                                             if status == "ran_but_mismatch" else "")})
        n_ok = sum(r["status"] == "reproduced" for r in rows)
        (layout.execution_dir / "comparison_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

        print("[5/5] Writing final report...")
        _sleep(0.8)
        return ReproductionResult("completed", mode, repo_url, rows, n_ok, spec.model_dump(), run_steps,
                                  [c.reason for c in spec.claims if c.status != "mapped"])

    runner.convert_pdf, runner.extract_paper, runner.run_reproduction = convert, extract, reproduce
