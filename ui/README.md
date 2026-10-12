# Web UI for PAPER-REPORT-AGENT

A live dashboard over `Pipeline.run_paper()`. It adds no pipeline logic: runs started here
produce the same `Outputs/<paper>/<run_id>/` folders as `python main.py run`, and CLI runs show up
in the UI's history.

```
pip install -r ui/requirements.txt
python -m ui.server                  # http://127.0.0.1:8000
python -m ui.server --demo           # simulated stages: no keys / Docker / Kaggle / GPU needed
python -m ui.server --out D:\runs --port 9000
```

## What you get

| Screen | What it does |
|---|---|
| **New run** | Drop a PDF, pick Kaggle / Local Docker, Marker / Docling, plan-only. The environment card runs the same preflight checks as the CLI and blocks *Start* until the fatal ones pass. |
| **Live run** | Stage rail (PDF → Extract → Reproduce → Report) with timings, sub-steps detected from the Sandbox output (`[1/5]` … `[5/5]`), streaming console with filter / follow / download, activity feed, and the extracted claims + planned steps appearing as soon as those files exist. Refreshing the page replays the run from the start. |
| **Runs** | History from `manifest.json` files: status, score bar, mode, duration. |
| **Report** | Overview (outcome breakdown, stage timings, sources found, warnings), Claims (paper vs reproduced value, Δ, matched output line, planner reason, flag when the extracted value was not found verbatim in the paper), Execution, Plan (RunSpec steps with evidence), Paper (figures + `paper.md`), `report.md`, and a file browser. **Re-run** uses the pipeline's `--resume` (reuses PDF conversion + extraction). |

## API (all under `/api`)

`GET health · env · runs · runs/{slug}/{run_id} · runs/{slug}/{run_id}/tree · runs/{slug}/{run_id}/file?path=`
`POST runs` (multipart: `pdf`, `mode`, `reader`, `plan_only`) · `POST runs/resume`
`GET jobs/{id}` · `GET jobs/{id}/events` (Server-Sent Events: `stage`, `log`, `run`, `done`, `error`, `preflight_failed`, `end`)

Interactive docs at `/api/docs`.

## Notes and limits

* **One run at a time** (a second start returns 409). Runs execute in a worker thread of the server process.
* **No cancel button**: `run_paper()` has no cancellation hook and a Python thread cannot be killed safely. Stop the server to abort.
* **No authentication, binds to 127.0.0.1.** The server can start Docker and spend Gemini/Kaggle quota; do not expose it on a network without auth in front.
* Live output is whatever the pipeline prints to stdout/stderr while the run is active (plus its stage events); the complete Logger output is in the run's `logs/` folder (Files tab).
* Demo mode numbers are **simulated** (the generated RunSpec carries a warning saying so). It reuses `Pdf-Reader/Output/` from the repo, when present, as the sample paper.
