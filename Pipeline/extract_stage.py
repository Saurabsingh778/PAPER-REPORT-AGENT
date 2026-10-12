"""Stage 2: paper.md -> extraction.json (headline results, code repos, data links) via Gemini."""
from __future__ import annotations

import json
from pathlib import Path

from .pdf_stage import StageError


def extract_paper(pdf_dir: Path, out_dir: Path) -> dict:
    """Run the Gemini extraction. Reads <pdf_dir>/paper.md, writes extraction.json/.md to out_dir.
    Returns {"extraction_json": Path, "claims": int, "repos": int, "data_links": int}."""
    from Model import gemini  # lazy: builds the Gemini client, needs API_KEY

    written = gemini.main(str(pdf_dir), results_dir=str(out_dir))
    path = Path(written["extraction_json"])
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("main_results"), list):
        raise StageError("extraction.json has no 'main_results' list")
    return {
        "extraction_json": path,
        "claims": len(data["main_results"]),
        "repos": len(data.get("code_repos", [])),
        "data_links": len(data.get("data_links", [])),
    }
