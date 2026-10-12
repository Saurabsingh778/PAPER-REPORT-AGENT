"""The integration: run_paper() takes a PDF and returns a finished run folder with a report.

    pdf -> [01 pdf stage] -> [02 extract] -> [03 pipeline: repo, plan, execute, compare] -> [04 report]

Rules the runner enforces so a UI (or a person) can rely on it:
  * one folder per run (see paths.py), created before any slow work starts;
  * manifest.json is updated at every stage boundary, so progress can be polled;
  * a failing stage never skips the report: report.md/report.json are always written and say
    where the run stopped and why;
  * stages 'pdf' and 'extract' can be reused with resume= (they are slow and need no infrastructure).
"""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from Logger import LogConfig, phase, start_run

from .extract_stage import extract_paper
from .index import point_latest, rebuild_index
from .manifest import Manifest
from .paths import RunLayout, layout_from_dir, new_layout
from .pdf_stage import READERS, convert_pdf
from .pipeline_stage import ReproductionResult, run_reproduction
from .preflight import assert_ready, run_preflight
from .report_stage import build_report, guess_title
from .utils import now_iso, read_json, sha256_file

MODES = ("kaggle", "local")
EventCallback = Callable[[dict], None]


@dataclass
class RunResult:
    run_dir: Path
    run_id: str
    status: str                 # completed | plan_only | failed
    report_md: Path
    report_json: Path
    reproduced: int | None
    total: int | None
    headline: str
    failed_stage: str | None = None
    error: str | None = None


class StageFailed(Exception):
    def __init__(self, stage: str, error: str):
        super().__init__(f"{stage}: {error}")
        self.stage = stage
        self.error = error


def _emitter(callback: EventCallback | None):
    def emit(stage: str, status: str, message: str = "") -> None:
        if callback is None:
            return
        try:
            callback({"stage": stage, "status": status, "message": message, "ts": now_iso()})
        except Exception:  # a broken progress listener must never kill a run
            pass
    return emit


@contextmanager
def _stage(manifest: Manifest, emit, name: str, description: str, log_phase: str | None):
    """Record a stage in the manifest and (optionally) in the Logger. The body fills the yielded
    dict: 'outputs' (list of run-relative paths) and any extra facts worth keeping."""
    manifest.start(name)
    emit(name, "started", description)
    info: dict = {}
    try:
        if log_phase:
            with phase(log_phase, description):
                yield info
        else:
            yield info
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        manifest.fail(name, err)
        emit(name, "failed", err)
        raise StageFailed(name, err) from exc
    manifest.done(name, outputs=info.pop("outputs", []), **info)
    emit(name, "done", str(info.get("summary", "")))


def _find_input_pdf(layout: RunLayout) -> Path:
    pdfs = sorted(layout.input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDF in {layout.input_dir}; cannot resume")
    return pdfs[0]


def run_paper(
    pdf_path: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    mode: str = "kaggle",
    reader: str = "marker",
    plan_only: bool = False,
    resume: str | Path | None = None,
    on_event: EventCallback | None = None,
    skip_preflight: bool = False,
) -> RunResult:
    """Run the whole pipeline for one PDF.

    pdf_path     the paper to process (omit when resuming)
    output_root  where run folders go (default: <project>/Outputs or $AGENT_OUTPUT_ROOT)
    mode         'kaggle' (default) or 'local' (Docker)
    reader       'marker' (default) or 'docling'
    plan_only    stop after the RunSpec is planned; nothing is executed
    resume       an existing run folder: reuse its finished pdf/extract stages, redo the rest
    on_event     called as on_event({"stage", "status", "message", "ts"}) at every stage change

    Raises PreflightError (nothing created yet) or ValueError for bad arguments. Any failure
    after the run folder exists is returned as RunResult(status='failed') with a report.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if reader not in READERS:
        raise ValueError(f"reader must be one of {sorted(READERS)}")
    emit = _emitter(on_event)

    # ---- locate the input and decide the run folder
    if resume:
        layout = layout_from_dir(resume)
        pdf = _find_input_pdf(layout)
    else:
        if pdf_path is None:
            raise ValueError("pass a PDF path (or resume=<run folder>)")
        pdf = Path(pdf_path).expanduser().resolve()
        layout = None

    # ---- preflight: fail in a second, before anything is created or converted
    if not skip_preflight:
        checks = run_preflight(pdf, mode=mode, reader=reader, plan_only=plan_only)
        for c in checks:
            emit("preflight", "ok" if c.ok else ("failed" if c.fatal else "warning"), f"{c.name}: {c.detail}")
        assert_ready(checks)

    # ---- create (or reopen) the run folder + manifest
    options = {"mode": mode, "reader": reader, "plan_only": plan_only}
    if layout is None:
        layout = new_layout(pdf, output_root)
        layout.create()
        stored = layout.input_dir / pdf.name
        shutil.copy2(pdf, stored)
        manifest = Manifest.create(
            layout,
            input_info={"name": pdf.name, "path": layout.rel(stored), "sha256": sha256_file(stored),
                        "size_bytes": stored.stat().st_size},
            options=options,
        )
        reuse = set()
    else:
        layout.create()
        manifest = Manifest.load(layout)
        manifest.update_options(**options)
        reuse = {s for s in ("pdf", "extract") if manifest.is_done(s) and (
            layout.paper_md if s == "pdf" else layout.extraction_json).is_file()}
        manifest.reset_for_resume(keep=reuse)
    emit("run", "started", f"{layout.slug}/{layout.run_id}")

    # ---- the stages
    extraction: dict | None = None
    result: ReproductionResult | None = None
    failure: StageFailed | None = None

    try:
        with start_run("paper_pipeline", config=LogConfig(root=layout.logs_dir),
                       meta={"pdf": str(pdf), "run_dir": str(layout.root), **options}):
            try:
                # 1. PDF -> markdown
                if "pdf" in reuse:
                    manifest.mark_reused("pdf")
                    emit("pdf", "reused", "paper.md from the earlier run")
                else:
                    with _stage(manifest, emit, "pdf", "convert the PDF to markdown and figures",
                                "phase0_pdf_read") as info:
                        out = convert_pdf(layout.input_dir / pdf.name, layout.pdf_dir, reader)
                        info["outputs"] = [layout.rel(out["markdown"])]
                        info["summary"] = f"{out['chars']} characters, {out['images']} images ({reader})"
                manifest.set_title(guess_title(layout.paper_md, Path(pdf).stem))

                # 2. markdown -> claims, repos, data links
                if "extract" in reuse:
                    manifest.mark_reused("extract")
                    emit("extract", "reused", "extraction.json from the earlier run")
                else:
                    with _stage(manifest, emit, "extract", "extract results, code repos and data links",
                                "phase1_extraction") as info:
                        out = extract_paper(layout.pdf_dir, layout.extraction_dir)
                        info["outputs"] = [layout.rel(out["extraction_json"])]
                        info["summary"] = (f"{out['claims']} claims, {out['repos']} code repos, "
                                           f"{out['data_links']} data links")
                extraction = read_json(layout.extraction_json)

                # 3. fetch repo, plan, execute, compare (Sandbox.main logs its own phases)
                with _stage(manifest, emit, "pipeline",
                            "plan only" if plan_only else "fetch repo, plan, execute and compare",
                            None) as info:
                    result = run_reproduction(layout.extraction_json, layout,
                                              use_local=(mode == "local"), plan_only=plan_only)
                    info["outputs"] = [layout.rel(p) for p in (layout.run_spec_json,) if p.exists()]
                    info["outcome"] = result.status
                    info["summary"] = (f"{result.n_ok}/{len(result.rows)} claims reproduced"
                                       if result.status == "completed" else result.status)
            except StageFailed as exc:
                failure = exc
                if extraction is None:
                    extraction = read_json(layout.extraction_json)

            # 4. report (always)
            run_status = ("failed" if failure else "plan_only" if (result and result.status == "plan_only")
                          else "completed")
            with _stage(manifest, emit, "report", "write report.md and report.json", "report") as info:
                report = build_report(
                    layout, manifest, extraction=extraction, result=result, run_status=run_status,
                    failed_stage=failure.stage if failure else None,
                    error=failure.error if failure else None)
                info["outputs"] = ["report.md", "report.json"]
                info["summary"] = report["headline"]
    except KeyboardInterrupt:
        manifest.finish("interrupted")
        rebuild_index(layout.output_root)
        raise
    except Exception as exc:  # a bug in the runner/report itself: don't leave the run 'running'
        manifest.finish("failed", result={"error": f"{type(exc).__name__}: {exc}"})
        rebuild_index(layout.output_root)
        raise

    summary = report["summary"]
    manifest.finish(run_status, result={
        "reproduced": summary["reproduced"], "total": summary["claims_total"],
        "headline": report["headline"], "outcome": report["run"]["reproduction_outcome"],
        "report": "report.md"})
    point_latest(layout)
    rebuild_index(layout.output_root)
    emit("run", run_status, report["headline"])

    return RunResult(
        run_dir=layout.root, run_id=layout.run_id, status=run_status,
        report_md=layout.report_md, report_json=layout.report_json,
        reproduced=summary["reproduced"], total=summary["claims_total"], headline=report["headline"],
        failed_stage=failure.stage if failure else None, error=failure.error if failure else None)
