"""manifest.json: the run's status board. Written after every change, so a UI (or a person)
can watch a run, and `--resume` can tell which stages are already finished."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from .paths import RunLayout
from .utils import now_iso, read_json, write_json_atomic

SCHEMA_VERSION = "1"
STAGES = ("pdf", "extract", "pipeline", "report")


class Manifest:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data
        self._t0: dict[str, float] = {}

    # ---- construction
    @classmethod
    def create(cls, layout: RunLayout, *, input_info: dict[str, Any],
               options: dict[str, Any]) -> "Manifest":
        data = {
            "schema_version": SCHEMA_VERSION,
            "run_id": layout.run_id,
            "slug": layout.slug,
            "status": "running",
            "created": now_iso(),
            "updated": now_iso(),
            "finished": None,
            "input": input_info,
            "options": options,
            "paper_title": None,
            "stages": {s: {"status": "pending"} for s in STAGES},
            "result": None,
        }
        m = cls(layout.manifest_json, data)
        m.save()
        return m

    @classmethod
    def load(cls, layout: RunLayout) -> "Manifest":
        data = read_json(layout.manifest_json)
        if not isinstance(data, dict):
            raise ValueError(f"unreadable manifest: {layout.manifest_json}")
        for s in STAGES:
            data.setdefault("stages", {}).setdefault(s, {"status": "pending"})
        return cls(layout.manifest_json, data)

    def save(self) -> None:
        self.data["updated"] = now_iso()
        write_json_atomic(self.path, self.data)

    # ---- stages
    def stage(self, name: str) -> dict[str, Any]:
        return self.data["stages"][name]

    def is_done(self, name: str) -> bool:
        return self.stage(name).get("status") == "done"

    def start(self, name: str) -> None:
        self._t0[name] = time.perf_counter()
        self.data["stages"][name] = {"status": "running", "started": now_iso()}
        self.save()

    def done(self, name: str, outputs: Iterable[str] = (), **info: Any) -> None:
        st = self.stage(name)
        st.update(status="done", finished=now_iso(), outputs=list(outputs), error=None, **info)
        if name in self._t0:
            st["duration_s"] = round(time.perf_counter() - self._t0[name], 2)
        self.save()

    def fail(self, name: str, error: str) -> None:
        st = self.stage(name)
        st.update(status="failed", finished=now_iso(), error=error)
        if name in self._t0:
            st["duration_s"] = round(time.perf_counter() - self._t0[name], 2)
        self.save()

    def mark_reused(self, name: str) -> None:
        """Resume: the stage's earlier output is kept as it is."""
        self.stage(name)["reused"] = True
        self.save()

    def reset_for_resume(self, keep: Iterable[str]) -> None:
        """Stages in `keep` stay as they are; every other stage goes back to pending."""
        keep = set(keep)
        for s in STAGES:
            if s not in keep:
                self.data["stages"][s] = {"status": "pending"}
        self.data.update(status="running", finished=None, result=None)
        self.save()

    # ---- run level
    def set_title(self, title: str | None) -> None:
        self.data["paper_title"] = title
        self.save()

    def update_options(self, **options: Any) -> None:
        self.data["options"].update(options)
        self.save()

    def finish(self, status: str, result: dict[str, Any] | None = None) -> None:
        self.data.update(status=status, finished=now_iso(), result=result)
        self.save()
