"""Global cross-run index.

  Logs/index.jsonl  one line when a run starts, one when it ends
  Logs/links.json   artifact path -> run that produced it

The links are how separate CLI invocations are chained into one story: the PDF reader
registers paper.md, the extraction step finds it (its upstream run) and registers the
phase1 JSON, and the Phase 2 pipeline finds that in turn.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import LogConfig
from .utils import iso, never_raise
from .writers import read_json, read_jsonl, write_json_atomic


def _key(path: str | Path) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(path)


class RunIndex:
    def __init__(self, cfg: LogConfig):
        self.cfg = cfg

    @never_raise
    def _append(self, record: dict[str, Any]) -> None:
        path = self.cfg.index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")

    def register_start(self, run_id: str, entry: str, run_dir: Path, upstream: str | None) -> None:
        self._append(
            {"event": "start", "ts": iso(), "run_id": run_id, "entry": entry,
             "dir": str(run_dir), "upstream_run_id": upstream}
        )

    def register_end(self, run_id: str, status: str, duration_s: float) -> None:
        self._append(
            {"event": "end", "ts": iso(), "run_id": run_id, "status": status,
             "duration_s": round(duration_s, 3)}
        )

    @never_raise
    def link_artifact(self, path: str | Path, run_id: str, kind: str) -> None:
        links = read_json(self.cfg.links_path(), {}) or {}
        links[_key(path)] = {"run_id": run_id, "kind": kind, "ts": iso()}
        write_json_atomic(self.cfg.links_path(), links)

    def find_run_for_artifact(self, path: str | Path) -> str | None:
        links = read_json(self.cfg.links_path(), {}) or {}
        entry = links.get(_key(path))
        return entry.get("run_id") if entry else None

    def runs(self) -> list[dict[str, Any]]:
        """Merge start/end lines into one record per run, oldest first."""
        merged: dict[str, dict[str, Any]] = {}
        for rec in read_jsonl(self.cfg.index_path()):
            rid = rec.get("run_id")
            if not rid:
                continue
            row = merged.setdefault(rid, {"run_id": rid})
            if rec.get("event") == "start":
                row.update(entry=rec.get("entry"), started=rec.get("ts"), dir=rec.get("dir"),
                           upstream_run_id=rec.get("upstream_run_id"))
            elif rec.get("event") == "end":
                row.update(status=rec.get("status"), ended=rec.get("ts"),
                           duration_s=rec.get("duration_s"))
        for row in merged.values():
            row.setdefault("status", "running/unknown")
        return list(merged.values())
