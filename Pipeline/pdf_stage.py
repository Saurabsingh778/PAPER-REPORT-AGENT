"""Stage 1: PDF -> markdown (+ figures/blocks) with Marker or Docling.

The reader scripts live in "Pdf-Reader/", which is not an importable package name (hyphen),
so they are loaded by file path. They are imported lazily: Marker pulls in torch and the model
stack, which a `--help` or a resumed run should never pay for.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from .paths import PROJECT_ROOT

READER_DIR = PROJECT_ROOT / "Pdf-Reader"
READERS = {"marker": "marker_pdf.py", "docling": "pdf_reader.py"}
MIN_MARKDOWN_CHARS = 200


class StageError(RuntimeError):
    """A stage ran but its output is unusable."""


def _load_reader(reader: str) -> ModuleType:
    if reader not in READERS:
        raise ValueError(f"unknown reader '{reader}' (choose from {sorted(READERS)})")
    path = READER_DIR / READERS[reader]
    spec = importlib.util.spec_from_file_location(f"pdf_reader_{reader}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_pdf(pdf_path: Path, out_dir: Path, reader: str = "marker") -> dict:
    """Convert `pdf_path` into `out_dir`. Returns {"markdown": Path, "chars": int, "images": int}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    module = _load_reader(reader)
    module.read_pdf(str(pdf_path), str(out_dir))

    md = out_dir / "paper.md"
    if not md.is_file():
        raise StageError(f"{reader} finished but wrote no paper.md")
    text = md.read_text(encoding="utf-8", errors="replace")
    if len(text.strip()) < MIN_MARKDOWN_CHARS:
        raise StageError(f"{reader} produced only {len(text.strip())} characters of text "
                         f"(scanned PDF without OCR, or an unreadable file?)")
    images = [p for p in out_dir.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png")]
    return {"markdown": md, "chars": len(text), "images": len(images)}
