"""SessionLogger: base class for an execution-environment layer (local Docker, Kaggle...).

A session is logged separately from the phase that launched it, because it has its own
lifecycle (workspace prep, container start/stop, per-command raw output). Subclasses add
environment-specific methods; this base provides the shared plumbing:

<session dir>/
    session.log         timestamped lifecycle log
    session.jsonl       the same events, structured
    session_summary.json
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from .config import LogConfig
from .formatters import phase_line
from .redaction import Redactor
from .utils import clock, dumps, iso, never_raise, now
from .writers import JsonlLog, TextLog, copy_redacted, write_json_atomic

FinishHook = Callable[[dict[str, Any]], None]


class SessionLogger:
    kind = "session"

    def __init__(
        self,
        run_dir: Path,
        dir_name: str,
        cfg: LogConfig,
        redactor: Redactor,
        phase_name: str | None = None,
        meta: dict[str, Any] | None = None,
        on_finish: FinishHook | None = None,
    ):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / dir_name
        self.rel = self.dir.relative_to(self.run_dir).as_posix()
        self.phase_name = phase_name
        self.meta = dict(meta or {})
        self._redactor = redactor
        self._on_finish = on_finish
        self._lock = threading.RLock()
        self._finished = False
        self._forced_status: str | None = None
        self.status = "running"
        self.started_at = now()
        self.ended_at = None
        self.details: dict[str, Any] = {}  # subclass-specific facts for the summary

        self.log_file = TextLog(self.dir / "session.log", redactor)
        self.events = JsonlLog(self.dir / "session.jsonl", redactor)
        self.log_file.writeline(phase_line(clock(), "INFO", f"=== {self.kind} session started ==="))
        if self.meta:
            self.event("session_start", **self.meta)

    # ------------------------------------------------------------------ logging
    @never_raise
    def event(self, type_: str, level: str = "INFO", msg: str | None = None, **fields: Any) -> None:
        ts = now()
        text = msg or type_
        line = phase_line(clock(ts), level, text)
        if fields:
            line += "  " + dumps(fields)
        self.log_file.writeline(line)
        record = {"type": type_, "ts": iso(ts), "level": level}
        if msg:
            record["msg"] = msg
        record.update(fields)
        self.events.append(record)

    def info(self, msg: str, **fields: Any) -> None:
        self.event("log", "INFO", msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self.event("log", "WARNING", msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self.event("log", "ERROR", msg, **fields)

    def mark(self, status: str) -> None:
        """Force the final status (e.g. 'failed' when the container exits non-zero)."""
        self._forced_status = status

    # ------------------------------------------------------------------ files
    @never_raise
    def save_text(self, rel: str, text: str) -> Path:
        dest = self.dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self._redactor(text), encoding="utf-8")
        return dest

    @never_raise
    def save_json(self, rel: str, obj: Any) -> Path:
        dest = self.dir / rel
        write_json_atomic(dest, obj, self._redactor)
        return dest

    @never_raise
    def copy_file(self, rel: str, src: Path) -> Path | None:
        src = Path(src)
        if not src.is_file() or src.stat().st_size > self.cfg.max_copy_bytes:
            return None
        dest = self.dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy_redacted(src, dest, self._redactor)
        return dest

    # ------------------------------------------------------------------ lifecycle
    def summary(self) -> dict[str, Any]:
        end = self.ended_at or now()
        return {
            "kind": self.kind,
            "dir": self.rel,
            "launched_by_phase": self.phase_name,
            "status": self.status,
            "started": iso(self.started_at),
            "ended": iso(self.ended_at) if self.ended_at else None,
            "duration_s": round((end - self.started_at).total_seconds(), 3),
            **self.details,
        }

    @never_raise
    def finish(self, status: str = "ok") -> dict[str, Any]:
        with self._lock:
            if self._finished:
                return self.summary()
            self._finished = True
            self.ended_at = now()
            self.status = self._forced_status or status
        s = self.summary()
        self.event("session_end", status=self.status, duration_s=s["duration_s"])
        write_json_atomic(self.dir / "session_summary.json", s, self._redactor)
        self.log_file.close()
        self.events.close()
        if self._on_finish:
            self._on_finish(s)
        return s
