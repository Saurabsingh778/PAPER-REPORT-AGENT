"""Where everything for one paper run lives. This is the only module that decides folder names.

    <output_root>/                         default: <project>/Outputs  (override: --out or AGENT_OUTPUT_ROOT)
      index.json                           every run, newest first (rebuilt after each run)
      <paper-slug>/
        latest -> <run_id>                 symlink to the newest run of this paper
        _cache/Repo, _cache/Data           downloaded code/data, reused by later runs of the same paper
        <run_id>/
          report.md                        START HERE: the final reproducibility report
          report.json                      same report, machine-readable
          manifest.json                    run status + per-stage status/timings/outputs
          00_input/<paper>.pdf             the uploaded PDF, as received
          01_pdf/                          paper.md, extracted figures, block JSON
          02_extraction/                   extraction.json (claims, repos, data links), extraction.md
          03_plan/                         run_spec.json (what was planned) + planner logs
          04_execution/                    runner workspace, raw results, comparison_rows.json, phase2.md/json
          logs/                            Logger output for the whole run
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Outputs"

INPUT_DIR = "00_input"
PDF_DIR = "01_pdf"
EXTRACTION_DIR = "02_extraction"
PLAN_DIR = "03_plan"
EXECUTION_DIR = "04_execution"
LOGS_DIR = "logs"
CACHE_DIR = "_cache"


def resolve_output_root(override: str | Path | None = None) -> Path:
    raw = override or os.getenv("AGENT_OUTPUT_ROOT")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_OUTPUT_ROOT


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return slug[:60].strip("-") or "paper"


@dataclass(frozen=True)
class RunLayout:
    output_root: Path
    slug: str
    run_id: str

    # ---- run folder and its parts
    @property
    def paper_root(self) -> Path:
        return self.output_root / self.slug

    @property
    def root(self) -> Path:
        return self.paper_root / self.run_id

    @property
    def input_dir(self) -> Path:
        return self.root / INPUT_DIR

    @property
    def pdf_dir(self) -> Path:
        return self.root / PDF_DIR

    @property
    def extraction_dir(self) -> Path:
        return self.root / EXTRACTION_DIR

    @property
    def plan_dir(self) -> Path:
        return self.root / PLAN_DIR

    @property
    def execution_dir(self) -> Path:
        return self.root / EXECUTION_DIR

    @property
    def logs_dir(self) -> Path:
        return self.root / LOGS_DIR

    # ---- shared per-paper download cache (not inside a run, so reruns skip re-downloading)
    @property
    def repo_cache(self) -> Path:
        return self.paper_root / CACHE_DIR / "Repo"

    @property
    def data_cache(self) -> Path:
        return self.paper_root / CACHE_DIR / "Data"

    # ---- key files
    @property
    def paper_md(self) -> Path:
        return self.pdf_dir / "paper.md"

    @property
    def extraction_json(self) -> Path:
        return self.extraction_dir / "extraction.json"

    @property
    def run_spec_json(self) -> Path:
        return self.plan_dir / "run_spec.json"

    @property
    def report_md(self) -> Path:
        return self.root / "report.md"

    @property
    def report_json(self) -> Path:
        return self.root / "report.json"

    @property
    def manifest_json(self) -> Path:
        return self.root / "manifest.json"

    def create(self) -> None:
        for d in (self.input_dir, self.pdf_dir, self.extraction_dir, self.plan_dir,
                  self.execution_dir, self.logs_dir, self.repo_cache, self.data_cache):
            d.mkdir(parents=True, exist_ok=True)

    def rel(self, path: str | Path) -> str:
        """Path relative to the run folder, forward slashes (what manifests and reports store)."""
        try:
            return Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return Path(path).as_posix()


def new_layout(pdf_path: str | Path, output_root: str | Path | None = None,
               run_id: str | None = None) -> RunLayout:
    """Layout for a fresh run of `pdf_path`. Never reuses an existing run folder."""
    root = resolve_output_root(output_root)
    slug = slugify(Path(pdf_path).stem)
    base = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    layout = RunLayout(root, slug, base)
    n = 1
    while layout.root.exists():
        n += 1
        layout = RunLayout(root, slug, f"{base}-{n}")
    return layout


def layout_from_dir(run_dir: str | Path) -> RunLayout:
    """Layout for an existing run folder (used by --resume)."""
    d = Path(run_dir).expanduser().resolve()
    if not (d / "manifest.json").is_file():
        raise FileNotFoundError(f"{d} is not a run folder (no manifest.json)")
    return RunLayout(d.parent.parent, d.parent.name, d.name)
