"""Finding earlier runs of the same PDF, so uploading it again reuses finished work.

Every run folder remembers the SHA-256 of the PDF it was started from (manifest.json ->
input.sha256). find_prior_run() looks an uploaded file up by that hash, so a renamed copy of the
same paper still matches and a different paper that happens to share a file name never does.

What an earlier run is worth, best first:

    report_ready   a finished report.md/report.json exists -> the caller should ask before redoing it
    extracted      01_pdf/paper.md and 02_extraction/extraction.json exist -> resume skips both stages
    converted      only 01_pdf/paper.md exists -> resume skips the (slow) PDF conversion

A run with none of these is not returned: there is nothing to reuse, so the upload is a fresh run.

This module only reads the output folder. Acting on the result (resume, show the report, start
fresh) is up to the caller; plan_for_upload() is the one place that decides which of those it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import layout_from_dir, resolve_output_root
from .utils import read_json

# What the caller may say about an upload that matches an earlier run with a finished report.
EXISTING_CHOICES = ("ask", "reuse", "regenerate", "new")


@dataclass(frozen=True)
class PriorRun:
    slug: str
    run_id: str
    run_dir: Path
    report_ready: bool
    pdf_done: bool          # paper.md is on disk and the manifest says the stage finished
    extract_done: bool      # extraction.json likewise
    status: str | None      # the run's manifest status: completed | plan_only | failed | ...
    title: str | None
    created: str | None
    finished: str | None
    mode: str | None
    reader: str | None
    plan_only: bool         # whether that earlier run was a plan-only run
    headline: str | None
    reproduced: int | None
    total: int | None

    @property
    def state(self) -> str:
        return "report_ready" if self.report_ready else "extracted" if self.extract_done else "converted"

    @property
    def reusable_stages(self) -> list[str]:
        """The stages a resume of this run will skip (same rule as Pipeline.runner)."""
        return [s for s, ok in (("pdf", self.pdf_done), ("extract", self.extract_done)) if ok]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for the UI."""
        return {
            "slug": self.slug, "run_id": self.run_id, "state": self.state, "status": self.status,
            "title": self.title, "created": self.created, "finished": self.finished,
            "mode": self.mode, "reader": self.reader, "plan_only": self.plan_only,
            "headline": self.headline, "reproduced": self.reproduced, "total": self.total,
            "reuses": self.reusable_stages,
        }


def _stage_done(manifest: dict, stage: str, output: Path) -> bool:
    status = ((manifest.get("stages") or {}).get(stage) or {}).get("status")
    return status == "done" and output.is_file()


def _classify(manifest: dict, run_dir: Path, plan_only: bool) -> PriorRun | None:
    layout = layout_from_dir(run_dir)
    status = manifest.get("status")
    options = manifest.get("options") or {}
    has_report = layout.report_md.is_file() and layout.report_json.is_file()
    # A plan-only report is a finished report only for a plan-only request: asked for a full
    # reproduction, "plan only" is not the report the person is after.
    report_ready = has_report and (status == "completed" or (status == "plan_only" and plan_only))
    pdf_done = _stage_done(manifest, "pdf", layout.paper_md)
    extract_done = pdf_done and _stage_done(manifest, "extract", layout.extraction_json)
    if not (report_ready or pdf_done):
        return None
    result = manifest.get("result") or {}
    return PriorRun(
        slug=layout.slug, run_id=layout.run_id, run_dir=run_dir,
        report_ready=report_ready, pdf_done=pdf_done, extract_done=extract_done,
        status=status, title=manifest.get("paper_title"), created=manifest.get("created"),
        finished=manifest.get("finished"), mode=options.get("mode"), reader=options.get("reader"),
        plan_only=bool(options.get("plan_only")), headline=result.get("headline"),
        reproduced=result.get("reproduced"), total=result.get("total"))


def find_prior_run(sha256: str, output_root: str | Path | None = None, *, reader: str,
                   plan_only: bool = False) -> PriorRun | None:
    """The most useful earlier run of the PDF with this hash, or None.

    Only runs converted with the same `reader` count: reusing Marker's markdown for a Docling
    request would silently ignore the reader the person picked. Preference order: a finished
    report, then a run that also has the extraction, then the newest.
    """
    root = resolve_output_root(output_root)
    best: PriorRun | None = None
    best_key: tuple | None = None
    for manifest_path in root.glob("*/*/manifest.json"):
        run_dir = manifest_path.parent
        if run_dir.is_symlink():  # <slug>/latest points at a run folder that is listed itself
            continue
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or (manifest.get("input") or {}).get("sha256") != sha256:
            continue
        if (manifest.get("options") or {}).get("reader") not in (None, reader):
            continue
        prior = _classify(manifest, run_dir, plan_only)
        if prior is None:
            continue
        key = (prior.report_ready, prior.extract_done, prior.created or "", prior.run_id)
        if best_key is None or key > best_key:
            best, best_key = prior, key
    return best


def plan_for_upload(prior: PriorRun | None, existing: str = "ask") -> str:
    """What to do with an upload, given the earlier run found for it (or None) and the person's
    answer so far (`existing`, one of EXISTING_CHOICES).

        fresh        no usable earlier run, or the person wants to start from scratch
        ask          a finished report exists and nobody has said what to do yet
        show_report  keep the finished report and run nothing
        resume       continue the earlier run: reuse its converted/extracted files, redo the rest
    """
    if existing not in EXISTING_CHOICES:
        raise ValueError(f"existing must be one of {EXISTING_CHOICES}")
    if prior is None or existing == "new":
        return "fresh"
    if prior.report_ready:
        return {"reuse": "show_report", "regenerate": "resume"}.get(existing, "ask")
    return "resume"