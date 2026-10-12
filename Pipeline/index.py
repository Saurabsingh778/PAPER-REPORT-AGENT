"""Finding runs again: Outputs/index.json (all runs, newest first) and Outputs/<slug>/latest.

The index is rebuilt by scanning the manifests rather than edited in place, so two runs
finishing at the same time cannot corrupt it and a deleted run folder simply disappears.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .paths import RunLayout, resolve_output_root
from .utils import now_iso, read_json, write_json_atomic


def _summarise(manifest_path: Path, output_root: Path) -> dict[str, Any] | None:
    m = read_json(manifest_path)
    if not isinstance(m, dict):
        return None
    run_dir = manifest_path.parent
    result = m.get("result") or {}
    return {
        "slug": m.get("slug"),
        "run_id": m.get("run_id"),
        "status": m.get("status"),
        "title": m.get("paper_title"),
        "pdf": (m.get("input") or {}).get("name"),
        "mode": (m.get("options") or {}).get("mode"),
        "created": m.get("created"),
        "finished": m.get("finished"),
        "reproduced": result.get("reproduced"),
        "total": result.get("total"),
        "headline": result.get("headline"),
        "path": run_dir.relative_to(output_root).as_posix(),
        "report": (run_dir / "report.md").relative_to(output_root).as_posix()
        if (run_dir / "report.md").exists() else None,
    }


def list_runs(output_root: str | Path | None = None, slug: str | None = None) -> list[dict[str, Any]]:
    """All runs under the output root (optionally one paper), newest first."""
    root = resolve_output_root(output_root)
    runs = []
    for mp in root.glob("*/*/manifest.json"):
        if mp.parent.is_symlink():  # <slug>/latest points at a real run folder that is listed itself
            continue
        if slug and mp.parent.parent.name != slug:
            continue
        s = _summarise(mp, root)
        if s:
            runs.append(s)
    runs.sort(key=lambda r: (r.get("created") or "", r.get("run_id") or ""), reverse=True)
    return runs


def rebuild_index(output_root: str | Path | None = None) -> Path:
    root = resolve_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "index.json"
    write_json_atomic(path, {"schema_version": "1", "updated": now_iso(), "runs": list_runs(root)})
    return path


def point_latest(layout: RunLayout) -> None:
    """<slug>/latest -> newest run folder. A symlink where the OS allows it, otherwise latest.txt."""
    link = layout.paper_root / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(layout.run_id, link, target_is_directory=True)
    except OSError:
        (layout.paper_root / "latest.txt").write_text(layout.run_id + "\n", encoding="utf-8")


def latest_run(output_root: str | Path | None = None, slug: str | None = None) -> Path | None:
    """Folder of the newest run (of one paper if `slug` is given), or None."""
    root = resolve_output_root(output_root)
    runs = list_runs(root, slug)
    return (root / runs[0]["path"]) if runs else None
