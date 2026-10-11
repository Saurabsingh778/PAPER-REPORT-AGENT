"""RunLogger: one pipeline invocation = one run directory.

Logs/runs/<run_id>/
    main.log            human-readable: one summary line per call + phase/session boundaries
                        and every warning/error from any phase
    main.jsonl          the same, structured (type = run_start|phase_start|call|phase_end|...)
    manifest.json       run metadata, upstream run, phase + session index (always current)
    summary.md          readable report generated when the run ends
    phases/<NN>_<name>/ raw output of each phase (see phase_logger.py)
    local_session/      the local Docker layer, logged separately (see local_session.py)
"""
from __future__ import annotations

import atexit
import os
import platform
import socket
import subprocess
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .capture import ConsoleCapture
from .config import PROJECT_ROOT, LogConfig
from .formatters import call_summary_text, main_line
from .index import RunIndex
from .local_session import LocalSessionLogger
from .phase_logger import PhaseLogger
from .redaction import Redactor
from .summary import build_summary_md
from .utils import iso, never_raise, new_run_id, now, short_repr, truncate
from .writers import JsonlLog, TextLog, write_json_atomic

UNPHASED = "unphased"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


class RunLogger:
    def __init__(
        self,
        entry: str,
        *,
        config: LogConfig | None = None,
        meta: dict[str, Any] | None = None,
        upstream_artifact: str | Path | None = None,
    ):
        self.cfg = config or LogConfig()
        self.redactor = Redactor(self.cfg.redact)
        self.entry = entry
        self.run_id = new_run_id(entry)
        self.dir = self.cfg.runs_dir() / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)

        self.started_at = now()
        self.ended_at = None
        self.status = "running"
        self.finished = False
        self._lock = threading.RLock()
        self._phase_stack: list[PhaseLogger] = []
        self._phases: dict[str, dict[str, Any]] = {}       # key: "NN_name" -> summary
        self._phase_objects: list[PhaseLogger] = []
        self._phase_counter = 0
        self._call_counter = 0
        self._sessions: list[dict[str, Any]] = []
        self._default_phase: PhaseLogger | None = None
        self._capture: ConsoleCapture | None = None

        self.main_log = TextLog(self.dir / "main.log", self.redactor)
        self.main_jsonl = JsonlLog(self.dir / "main.jsonl", self.redactor)
        self.index = RunIndex(self.cfg)

        self.upstream_run_id = None
        if upstream_artifact is not None:
            self.upstream_run_id = self.index.find_run_for_artifact(upstream_artifact)

        self.manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "entry": entry,
            "status": self.status,
            "started": iso(self.started_at),
            "ended": None,
            "duration_s": None,
            "upstream_run_id": self.upstream_run_id,
            "meta": meta or {},
            "environment": {
                "argv": sys.argv,
                "cwd": os.getcwd(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "host": socket.gethostname(),
                "git_commit": _git_commit(),
            },
            "files": {"main_log": "main.log", "main_jsonl": "main.jsonl", "summary": "summary.md"},
            "phases": [],
            "sessions": [],
            "outputs": [],
        }
        self._write_manifest()
        self.index.register_start(self.run_id, entry, self.dir, self.upstream_run_id)
        self._main("INFO", "run", f"run started: {self.run_id}", type="run_start",
                   entry=entry, upstream_run_id=self.upstream_run_id, meta=meta or {})

        if self.cfg.capture_console:
            self._capture = ConsoleCapture(self._console_sink)
            self._capture.install()
        atexit.register(self._atexit)

    # ------------------------------------------------------------------ main log
    @never_raise
    def _main(self, level: str, phase: str, text: str, type: str = "event", **fields: Any) -> None:
        ts = now()
        line = main_line(ts.strftime("%Y-%m-%d %H:%M:%S"), level, phase, text)
        self.main_log.writeline(line)
        record = {"type": type, "ts": iso(ts), "level": level, "phase": phase, "msg": text}
        record.update(fields)
        self.main_jsonl.append(record)
        if self.cfg.echo_main:
            sys.__stdout__.write(line + "\n")

    def event(self, level: str, msg: str, **fields: Any) -> None:
        """Run-level event in the main log (outside any phase)."""
        self._main(level.upper(), self.current_phase().name if self._phase_stack else "run", msg, **fields)

    def next_call_id(self) -> str:
        with self._lock:
            self._call_counter += 1
            return f"c{self._call_counter:04d}"

    @never_raise
    def log_call(self, rec: dict[str, Any]) -> None:
        """Summary of one instrumented call -> main.log + main.jsonl."""
        text = call_summary_text(rec)
        level = "INFO" if rec.get("status") == "ok" else "ERROR"
        fields = {k: v for k, v in rec.items() if k not in ("phase",)}
        self._main(level, rec.get("phase", UNPHASED), text, type="call", **fields)

    # ------------------------------------------------------------------ phases
    def _problem(self, phase_name: str):
        def hook(level: str, msg: str) -> None:
            self._main(level, phase_name, truncate(msg, self.cfg.summary_max_chars) + "  (see phase log)")
        return hook

    def _open_phase(self, name: str, description: str = "", index: int | None = None) -> PhaseLogger:
        with self._lock:
            if index is None:
                self._phase_counter += 1
                index = self._phase_counter
            pl = PhaseLogger(self.dir, name, index, self.cfg, self.redactor, description,
                             on_problem=self._problem(name))
            self._phase_objects.append(pl)
            self._phases[pl.rel] = pl.summary()
        self._sync_manifest_phases()
        return pl

    def current_phase(self) -> PhaseLogger:
        """Active phase, or the run's catch-all 'unphased' phase if none is open."""
        with self._lock:
            if self._phase_stack:
                return self._phase_stack[-1]
            if self._default_phase is None:
                # Always folder 00_unphased so the numbered phases keep their own order.
                self._default_phase = self._open_phase(
                    UNPHASED, "output produced outside any named phase", index=0)
            return self._default_phase

    @contextmanager
    def phase(self, name: str, description: str = "") -> Iterator[PhaseLogger]:
        pl = self._open_phase(name, description)
        with self._lock:
            self._phase_stack.append(pl)
        self._main("INFO", name, f"phase started{': ' + description if description else ''}",
                   type="phase_start", phase_dir=pl.rel)
        status = "ok"
        try:
            yield pl
        except KeyboardInterrupt as exc:
            status = "interrupted"
            pl.fail(exc)
            raise
        except SystemExit as exc:
            status = "ok" if exc.code in (0, None) else "failed"
            raise
        except BaseException as exc:
            status = "failed"
            pl.fail(exc)
            raise
        finally:
            with self._lock:
                if self._phase_stack and self._phase_stack[-1] is pl:
                    self._phase_stack.pop()
            summary = pl.finish(status)
            self._phases[pl.rel] = summary
            self._sync_manifest_phases()
            level = "INFO" if summary["status"].startswith("ok") else "ERROR"
            self._main(
                level, name,
                f"phase ended: {summary['status']} in {summary['duration_s']}s "
                f"(calls={summary['calls']} failed={summary['calls_failed']} "
                f"warnings={summary['warnings']} errors={summary['errors']}) -> {pl.rel}/",
                type="phase_end", **summary)

    # ------------------------------------------------------------------ sessions
    @contextmanager
    def local_session(self, **meta: Any) -> Iterator[LocalSessionLogger]:
        phase_name = self._phase_stack[-1].name if self._phase_stack else None
        sess = LocalSessionLogger(self.dir, self.cfg, self.redactor, phase_name, meta,
                                  on_finish=self._session_finished)
        self._main("INFO", phase_name or "run", "local session started -> local_session/",
                   type="session_start", kind=sess.kind, **meta)
        status = "ok"
        try:
            yield sess
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            sess.error(f"{type(exc).__name__}: {exc}",
                       traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            raise
        finally:
            sess.finish(status)

    @never_raise
    def _session_finished(self, summary: dict[str, Any]) -> None:
        with self._lock:
            self._sessions.append(summary)
            self.manifest["sessions"] = list(self._sessions)
        self._write_manifest()
        level = "INFO" if str(summary.get("status", "")).startswith("ok") else "ERROR"
        results = summary.get("results") or {}
        docker = summary.get("docker") or {}
        gpu = summary.get("gpu") or {}
        text = (f"local session ended: {summary.get('status')} in {summary.get('duration_s')}s "
                f"(docker rc={docker.get('returncode')}, "
                f"steps ok={results.get('steps_ok')} failed={results.get('steps_failed')}, "
                f"gpu={'on' if gpu.get('enabled') else 'off'}) -> {summary.get('dir')}/")
        self._main(level, summary.get("launched_by_phase") or "run", text, type="session_end", **summary)

    # ------------------------------------------------------------------ links / outputs
    @never_raise
    def link_artifact(self, path: str | Path, kind: str) -> None:
        """Record that this run produced `path`, so a later run can find its upstream."""
        self.index.link_artifact(path, self.run_id, kind)
        with self._lock:
            self.manifest["outputs"].append({"path": str(path), "kind": kind})
        self._write_manifest()
        self._main("INFO", self.current_phase().name, f"output registered [{kind}]: {path}",
                   type="output", path=str(path), kind=kind)

    # ------------------------------------------------------------------ console capture
    def _console_sink(self, stream: str, line: str) -> None:
        if self.finished:
            return
        self.current_phase().console(stream, line)

    # ------------------------------------------------------------------ manifest
    @never_raise
    def _write_manifest(self) -> None:
        with self._lock:
            write_json_atomic(self.dir / "manifest.json", self.manifest, self.redactor)

    def _sync_manifest_phases(self) -> None:
        with self._lock:
            rows = []
            for pl in self._phase_objects:
                rows.append(self._phases.get(pl.rel) or pl.summary())
            self.manifest["phases"] = sorted(rows, key=lambda r: r["index"])
        self._write_manifest()

    # ------------------------------------------------------------------ lifecycle
    def finish(self, status: str | None = None, error: BaseException | None = None) -> None:
        with self._lock:
            if self.finished:
                return
            self.finished = True
        if self._capture:
            self._capture.uninstall()
        if error is not None:
            self._main("ERROR", "run", f"run failed: {type(error).__name__}: {short_repr(str(error), 300)}",
                       type="error", error_type=type(error).__name__)
            tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            (self.dir / "error.txt").write_text(self.redactor(tb), encoding="utf-8")

        for pl in list(self._phase_stack):
            self._phases[pl.rel] = pl.finish("aborted")
        self._phase_stack.clear()
        if self._default_phase is not None:
            self._phases[self._default_phase.rel] = self._default_phase.finish("ok")

        self.ended_at = now()
        if status is None:
            bad = [p for p in self._phases.values() if p.get("status") in ("failed", "aborted", "interrupted")]
            soft = [p for p in self._phases.values() if p.get("status") == "ok_with_errors"]
            if error is not None or bad:
                status = "failed"
            else:
                status = "ok_with_errors" if soft else "ok"
        self.status = status
        duration = round((self.ended_at - self.started_at).total_seconds(), 3)
        self.manifest.update(status=status, ended=iso(self.ended_at), duration_s=duration)
        self._sync_manifest_phases()
        self._main("INFO" if status == "ok" else "ERROR", "run",
                   f"run ended: {status} in {duration}s -> {self.dir}", type="run_end",
                   status=status, duration_s=duration)

        try:
            (self.dir / "summary.md").write_text(
                self.redactor(build_summary_md(self.dir)), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            sys.__stderr__.write(f"[Logger] could not write summary.md: {exc!r}\n")
        self.index.register_end(self.run_id, status, duration)
        self.main_log.close()
        self.main_jsonl.close()

    def _atexit(self) -> None:
        if not self.finished:
            self.finish("aborted")

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(error=exc if exc is not None else None)
        return False
