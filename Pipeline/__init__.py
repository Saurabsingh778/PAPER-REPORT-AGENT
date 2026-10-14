"""Pipeline: the glue from an uploaded PDF to a finished reproducibility report.

    from Pipeline import run_paper
    result = run_paper("paper.pdf", on_event=print)
    result.report_md        # <Outputs>/<paper>/<run_id>/report.md

One module per concern:
    paths.py           run folder layout (the only place folder names are decided)
    manifest.py        manifest.json status board (per-stage status, timings, outputs)
    preflight.py       fast checks (keys, tools, the PDF) before any slow work
    pdf_stage.py       stage 1: PDF -> markdown (Marker / Docling)
    extract_stage.py   stage 2: markdown -> claims, repos, data links (Gemini)
    pipeline_stage.py  stage 3: fetch repo -> plan -> execute -> compare (Sandbox.main)
    report_stage.py    stage 4: report.md / report.json
    index.py           Outputs/index.json and <paper>/latest
    lookup.py          find an earlier run of the same PDF (by hash) so a re-upload can reuse its work
    runner.py          run_paper(): runs the stages in order, never skips the report
    selftest.py        `python -m Pipeline.selftest` (no API keys, Docker or Kaggle needed)
"""
from .index import latest_run, list_runs, rebuild_index
from .lookup import EXISTING_CHOICES, PriorRun, find_prior_run, plan_for_upload
from .paths import DEFAULT_OUTPUT_ROOT, RunLayout, resolve_output_root
from .preflight import PreflightError
from .runner import RunResult, run_paper

__all__ = [
    "run_paper", "RunResult", "PreflightError",
    "list_runs", "latest_run", "rebuild_index",
    "find_prior_run", "plan_for_upload", "PriorRun", "EXISTING_CHOICES",
    "RunLayout", "resolve_output_root", "DEFAULT_OUTPUT_ROOT",
]