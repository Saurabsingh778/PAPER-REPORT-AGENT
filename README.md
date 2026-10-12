# PAPER-REPORT-AGENT

Give it a paper PDF. It converts the paper, extracts the headline results and the code/data
links, plans how to reproduce them, runs the code (Kaggle or local Docker), compares the numbers
and writes a reproducibility report.

```
python main.py run paper.pdf                  # execute on Kaggle (default)
python main.py run paper.pdf --local          # execute locally in Docker
python main.py run paper.pdf --plan-only      # stop after the RunSpec; nothing is executed
python main.py run --resume Outputs/<paper>/<run_id>   # reuse the finished pdf/extract stages
python main.py list                           # every run, newest first
python main.py latest [paper-slug]            # path of the newest report
python -m Pipeline.selftest                   # checks the integration, no keys/Docker/Kaggle needed
```

Setup: copy `.env.example` to `.env` and fill it in. A missing key or tool is reported in one
second, before any slow work starts.

## Where the output goes

Everything for a run is in one folder. Start with `report.md`.

```
Outputs/                              (override with --out or AGENT_OUTPUT_ROOT)
  index.json                          every run, newest first
  <paper-slug>/
    latest -> <run_id>                newest run of this paper
    _cache/Repo, _cache/Data          downloaded code/data, reused by later runs of this paper
    <run_id>/                         e.g. 20261012-074500
      report.md                       the final report
      report.json                     the same report, machine-readable
      manifest.json                   run status + per-stage status, timings, outputs
      00_input/<paper>.pdf            the PDF as received
      01_pdf/                         paper.md, figures, block JSON
      02_extraction/                  extraction.json (claims, repos, data links), extraction.md
      03_plan/                        run_spec.json (what was planned) + planner logs
      04_execution/                   runner workspace, execution_results.json, comparison_rows.json, phase2.md
      logs/                           full Logger output for the run
```

A run that fails part-way still ends with a `report.md` that says which stage failed and why;
`manifest.json` shows which stages finished.

## From code (e.g. a UI)

```python
from Pipeline import run_paper, list_runs, latest_run

result = run_paper("paper.pdf", mode="kaggle", on_event=print)   # events: {stage, status, message, ts}
result.report_md, result.status, (result.reproduced, result.total)
```

## Layout

| Folder | Role |
|---|---|
| `Pipeline/` | the glue: run folder layout, manifest, preflight, one module per stage, `run_paper()` |
| `Pdf-Reader/` | PDF to markdown (Marker, Docling) |
| `Model/`, `Schema/` | Gemini extraction; `phase_1_schema` and the `RunSpec` contract |
| `RepoHandler/` | fetch the repo + data, plan the run (`RunSpec`) |
| `Sandbox/` | execute (Kaggle / local Docker), compare, phase 2 report |
| `Logger/` | hierarchical run logging |
