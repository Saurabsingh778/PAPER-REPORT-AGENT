"""Web UI for PAPER-REPORT-AGENT: a thin FastAPI layer over Pipeline.run_paper().

    pip install fastapi "uvicorn[standard]" python-multipart
    python -m ui.server                     # http://127.0.0.1:8000
    python -m ui.server --demo              # simulated stages: no keys, Docker, Kaggle or GPU needed
    python -m ui.server --out D:\\runs --port 9000

What it does
  * POST /api/runs               upload a PDF + options -> starts run_paper() in a worker thread
  * GET  /api/jobs/<id>/events   Server-Sent Events: every stage event, plus every line the
                                 pipeline prints (stdout/stderr), replayed from the start so a
                                 page refresh rebuilds the live view
  * GET  /api/runs ...           everything already on disk (Outputs/index.json, manifest.json,
                                 report.json, extraction.json, run_spec.json, comparison_rows.json)

It adds no pipeline logic of its own: the run folder, manifest and report are exactly what
`python main.py run` produces, so CLI runs and UI runs show up side by side.

Security: the server can launch Docker and spend API quota, so it binds to 127.0.0.1 by default
and has no authentication. Do not expose it on a network without putting auth in front of it.
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from Pipeline import PreflightError, list_runs, resolve_output_root, run_paper  # noqa: E402
from Pipeline.paths import (EXECUTION_DIR, EXTRACTION_DIR, PDF_DIR, PLAN_DIR)  # noqa: E402
from Pipeline.pdf_stage import READERS  # noqa: E402
from Pipeline.preflight import run_preflight  # noqa: E402
from Pipeline.utils import read_json  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
MODES = ("kaggle", "local")
MAX_UPLOAD_MB = 60
MAX_LOG_EVENTS = 20000     # per job; a runaway tqdm loop must not eat the server's memory
MAX_JOBS_KEPT = 25
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".ipynb_checkpoints"}

OUTPUT_ROOT: Path = resolve_output_root(None)
DEMO = False

app = FastAPI(title="PAPER-REPORT-AGENT UI", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")


# ============================================================================ jobs
class Job:
    """One run_paper() call in flight (or finished), with a replayable event list."""

    def __init__(self, filename: str, options: dict):
        self.id = uuid.uuid4().hex[:12]
        self.filename = filename
        self.options = options
        self.created = time.time()
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.finished = False
        self.status = "running"          # running | completed | plan_only | failed
        self.run_key: dict | None = None  # {"slug", "run_id"} once the run folder exists
        self.result: dict | None = None
        self._n_logs = 0

    def push(self, kind: str, **data) -> None:
        with self.cond:
            if kind == "log":
                self._n_logs += 1
                if self._n_logs > MAX_LOG_EVENTS:
                    if self._n_logs == MAX_LOG_EVENTS + 1:
                        data = {"line": f"[ui] log limit reached ({MAX_LOG_EVENTS} lines); the full log is in the run's logs/ folder",
                                "stream": "ui"}
                    else:
                        return
            self.events.append({"i": len(self.events), "kind": kind, "t": time.time(), **data})
            self.cond.notify_all()

    def on_event(self, ev: dict) -> None:
        if ev.get("stage") == "run" and ev.get("status") == "started" and "/" in ev.get("message", ""):
            slug, run_id = ev["message"].split("/", 1)
            self.run_key = {"slug": slug, "run_id": run_id}
            self.push("run", **self.run_key)
        self.push("stage", **ev)

    def finish(self, status: str, result: dict | None = None) -> None:
        with self.cond:
            self.status = status
            self.result = result
            self.finished = True
            self.cond.notify_all()

    def snapshot(self) -> dict:
        return {"id": self.id, "filename": self.filename, "options": self.options, "created": self.created,
                "status": self.status, "finished": self.finished, "run": self.run_key,
                "result": self.result, "events": len(self.events)}


JOBS: dict[str, Job] = {}
ACTIVE: Job | None = None
LOCK = threading.Lock()


def _job(job_id: str) -> Job:
    j = JOBS.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job (the server may have been restarted)")
    return j


# ============================================================================ stdout/stderr -> job log
class _Tee(io.TextIOBase):
    """Forwards everything the pipeline prints to the active job as `log` events, line by line."""

    def __init__(self, real, stream: str):
        self.real, self.stream, self._buf = real, stream, ""

    def write(self, s):
        try:
            self.real.write(s)
        except Exception:
            pass
        job = ACTIVE
        if job is not None and s:
            self._buf += s
            parts = re.split(r"[\r\n]+", self._buf)
            self._buf = parts.pop()                       # incomplete tail waits for its newline
            for line in parts:
                if line.strip():
                    job.push("log", line=line.rstrip(), stream=self.stream)
        return len(s)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.real, name)


def _install_tee():
    saved = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = _Tee(saved[0], "out"), _Tee(saved[1], "err")
    return saved


def _remove_tee(saved):
    for t in (sys.stdout, sys.stderr):
        if isinstance(t, _Tee) and t._buf.strip() and ACTIVE is not None:
            ACTIVE.push("log", line=t._buf.rstrip(), stream=t.stream)
    sys.stdout, sys.stderr = saved


def _worker(job: Job, pdf: Path | None, mode: str, reader: str, plan_only: bool,
            resume: Path | None, cleanup: Path | None) -> None:
    global ACTIVE
    saved = _install_tee()
    try:
        res = run_paper(
            pdf, output_root=OUTPUT_ROOT, mode=mode, reader=reader, plan_only=plan_only,
            resume=resume, on_event=job.on_event, skip_preflight=DEMO)
        result = {"status": res.status, "headline": res.headline, "reproduced": res.reproduced,
                  "total": res.total, "failed_stage": res.failed_stage, "error": res.error}
        job.push("done", **result)
        job.finish(res.status, result)
    except PreflightError as exc:
        job.push("preflight_failed", message=str(exc), checks=[asdict(c) for c in exc.checks])
        job.finish("failed", {"status": "failed", "headline": "Preflight failed: fix the missing items and start again",
                              "error": str(exc)})
    except BaseException as exc:  # noqa: BLE001  (a UI worker must always report, never die silently)
        tb = traceback.format_exc()
        job.push("error", message=f"{type(exc).__name__}: {exc}", trace=tb)
        job.finish("failed", {"status": "failed", "headline": f"Run crashed: {type(exc).__name__}: {exc}",
                              "error": str(exc)})
    finally:
        _remove_tee(saved)
        with LOCK:
            ACTIVE = None
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


def _start_job(job: Job, **kw) -> None:
    global ACTIVE
    with LOCK:
        if ACTIVE is not None and not ACTIVE.finished:
            raise HTTPException(409, f"another run is in progress ({ACTIVE.filename}); one run at a time")
        ACTIVE = job
        JOBS[job.id] = job
        for old in sorted(JOBS.values(), key=lambda j: j.created)[:-MAX_JOBS_KEPT]:
            if old.finished:
                JOBS.pop(old.id, None)
    threading.Thread(target=_worker, args=(job,), kwargs=kw, daemon=True, name=f"run-{job.id}").start()


# ============================================================================ helpers
def _check_options(mode: str, reader: str) -> None:
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    if reader not in READERS:
        raise HTTPException(422, f"reader must be one of {sorted(READERS)}")


def _run_dir(slug: str, run_id: str) -> Path:
    d = (OUTPUT_ROOT / slug / run_id).resolve()
    if not d.is_relative_to(OUTPUT_ROOT.resolve()) or not (d / "manifest.json").is_file():
        raise HTTPException(404, "run not found")
    return d


def _safe_name(name: str) -> str:
    base = re.sub(r"[^\w.\- ]+", "_", Path(name or "paper.pdf").name).strip(" .") or "paper.pdf"
    return base if base.lower().endswith(".pdf") else base + ".pdf"


# ============================================================================ API
@app.get("/api/health")
def health():
    act = ACTIVE if (ACTIVE and not ACTIVE.finished) else None
    return {"ok": True, "demo": DEMO, "output_root": str(OUTPUT_ROOT), "readers": sorted(READERS),
            "modes": list(MODES), "active_job": act.snapshot() if act else None}


@app.get("/api/env")
def env(mode: str = "kaggle", reader: str = "marker", plan_only: bool = False):
    """The same fast checks run_paper() does first, minus the PDF check, so the UI can show them
    before anything is uploaded."""
    _check_options(mode, reader)
    if DEMO:
        names = [("gemini key", "simulated"), ("pdf reader", "simulated"),
                 ("docker" if mode == "local" else "kaggle credentials", "simulated")]
        return [{"name": n, "ok": True, "detail": d, "fatal": True} for n, d in names]
    checks = run_preflight(Path("__ui_check__.pdf"), mode=mode, reader=reader, plan_only=plan_only)
    return [asdict(c) for c in checks if c.name != "pdf"]


@app.post("/api/runs")
async def start_run(pdf: UploadFile = File(...), mode: str = Form("kaggle"), reader: str = Form("marker"),
                    plan_only: bool = Form(False)):
    _check_options(mode, reader)
    head = await pdf.read(5)
    if head != b"%PDF-":
        raise HTTPException(422, f"{pdf.filename} does not look like a PDF")
    tmp = Path(tempfile.mkdtemp(prefix="paper-ui-"))
    dest = tmp / _safe_name(pdf.filename)
    size = len(head)
    with dest.open("wb") as fh:
        fh.write(head)
        while chunk := await pdf.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                shutil.rmtree(tmp, ignore_errors=True)
                raise HTTPException(413, f"PDF is larger than {MAX_UPLOAD_MB} MB")
            fh.write(chunk)
    job = Job(dest.name, {"mode": mode, "reader": reader, "plan_only": plan_only, "size_bytes": size})
    try:
        _start_job(job, pdf=dest, mode=mode, reader=reader, plan_only=plan_only, resume=None, cleanup=tmp)
    except HTTPException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return job.snapshot()


class ResumeBody(BaseModel):
    slug: str
    run_id: str
    mode: str | None = None
    reader: str | None = None
    plan_only: bool = False


@app.post("/api/runs/resume")
def resume_run(body: ResumeBody):
    d = _run_dir(body.slug, body.run_id)
    m = read_json(d / "manifest.json", {}) or {}
    opts = m.get("options") or {}
    mode = body.mode or opts.get("mode") or "kaggle"
    reader = body.reader or opts.get("reader") or "marker"
    _check_options(mode, reader)
    job = Job((m.get("input") or {}).get("name", body.slug),
              {"mode": mode, "reader": reader, "plan_only": body.plan_only, "resume": f"{body.slug}/{body.run_id}"})
    _start_job(job, pdf=None, mode=mode, reader=reader, plan_only=body.plan_only, resume=d, cleanup=None)
    return job.snapshot()


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str):
    return _job(job_id).snapshot()


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str, request: Request):
    """SSE. Replays the job from the first event (or from Last-Event-ID on reconnect)."""
    job = _job(job_id)
    try:
        i = int(request.headers.get("last-event-id", "-1")) + 1
    except ValueError:
        i = 0

    def gen():
        nonlocal i
        yield "retry: 2000\n\n"
        while True:
            ping = False
            with job.cond:
                if i >= len(job.events) and not job.finished:
                    ping = not job.cond.wait(timeout=10)
                batch, fin = job.events[i:], job.finished
            for ev in batch:
                yield f"id: {ev['i']}\nevent: {ev['kind']}\ndata: {json.dumps(ev)}\n\n"
                i = ev["i"] + 1
            if fin and i >= len(job.events):
                yield f"event: end\ndata: {json.dumps(job.snapshot())}\n\n"
                return
            if ping:
                yield ": ping\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs")
def runs():
    return {"output_root": str(OUTPUT_ROOT), "runs": list_runs(OUTPUT_ROOT)}


@app.get("/api/runs/{slug}/{run_id}")
def run_detail(slug: str, run_id: str):
    d = _run_dir(slug, run_id)
    return {
        "slug": slug, "run_id": run_id,
        "manifest": read_json(d / "manifest.json"),
        "report": read_json(d / "report.json"),
        "extraction": read_json(d / EXTRACTION_DIR / "extraction.json"),
        "run_spec": read_json(d / PLAN_DIR / "run_spec.json"),
        "comparison_rows": read_json(d / EXECUTION_DIR / "comparison_rows.json"),
        "has_report_md": (d / "report.md").is_file(),
    }


@app.get("/api/runs/{slug}/{run_id}/tree")
def run_tree(slug: str, run_id: str):
    d = _run_dir(slug, run_id)
    out = []
    for dirpath, dirnames, filenames in os.walk(d):
        dirnames[:] = sorted(x for x in dirnames if x not in SKIP_DIRS)
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.is_symlink() or f.endswith(".tmp"):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            out.append({"path": p.relative_to(d).as_posix(), "size": st.st_size, "mtime": st.st_mtime})
            if len(out) >= 3000:
                return {"files": out, "truncated": True}
    return {"files": out, "truncated": False}


_TEXT_EXT = {".md", ".json", ".txt", ".log", ".jsonl", ".yaml", ".yml", ".py", ".csv", ".toml"}


@app.get("/api/runs/{slug}/{run_id}/file")
def run_file(slug: str, run_id: str, path: str = Query(...), download: bool = False):
    d = _run_dir(slug, run_id)
    p = (d / path).resolve()
    if not p.is_relative_to(d) or not p.is_file():
        raise HTTPException(404, "file not found")
    ext = p.suffix.lower()
    media = "text/plain; charset=utf-8" if ext in _TEXT_EXT else (mimetypes.guess_type(p.name)[0] or "application/octet-stream")
    return FileResponse(p, media_type=media, filename=p.name if download else None)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


# ============================================================================ entry point
def main() -> None:
    global OUTPUT_ROOT, DEMO
    ap = argparse.ArgumentParser(description="Web UI for PAPER-REPORT-AGENT")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; there is no auth)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", metavar="DIR", help="output root (default: ./Outputs or $AGENT_OUTPUT_ROOT)")
    ap.add_argument("--demo", action="store_true", help="simulate the slow stages; nothing external is called")
    args = ap.parse_args()

    OUTPUT_ROOT = resolve_output_root(args.out)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.demo:
        DEMO = True
        os.environ.setdefault("API_KEY", "demo-key")
        from ui import demo
        demo.install()
        print("[ui] DEMO mode: PDF conversion, Gemini, Docker and Kaggle are simulated")

    import uvicorn
    print(f"[ui] output root: {OUTPUT_ROOT}")
    print(f"[ui] open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
