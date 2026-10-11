"""PhaseLogger: the raw record of one pipeline phase.

phases/<NN>_<name>/
    phase.log          every log line + all captured console output, timestamped
    events.jsonl       structured events; every instrumented call with FULL args/result
    artifacts/         raw prompts, model responses, copied files, oversized values
    phase_summary.json written when the phase ends
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from .config import LogConfig
from .formatters import console_line, phase_line, raw_block
from .redaction import Redactor
from .utils import clock, dumps, iso, never_raise, now, slug, to_jsonable, truncate
from .writers import JsonlLog, TextLog, copy_redacted, write_json_atomic

ProblemHook = Callable[[str, str], None]


class PhaseLogger:
    def __init__(
        self,
        run_dir: Path,
        name: str,
        index: int,
        cfg: LogConfig,
        redactor: Redactor,
        description: str = "",
        on_problem: ProblemHook | None = None,
    ):
        self.name = name
        self.index = index
        self.description = description
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "phases" / f"{index:02d}_{slug(name)}"
        self.rel = self.dir.relative_to(self.run_dir).as_posix()
        self.artifacts_dir = self.dir / "artifacts"
        self._redactor = redactor
        self._on_problem = on_problem
        self._lock = threading.RLock()
        self._artifact_names: set[str] = set()

        self.log_file = TextLog(self.dir / "phase.log", redactor)
        self.events = JsonlLog(self.dir / "events.jsonl", redactor)

        self.started_at = now()
        self.ended_at = None
        self.status = "running"
        self.counters = {"calls": 0, "calls_failed": 0, "warnings": 0, "errors": 0, "artifacts": 0}
        self._finished = False

        self.log_file.writeline(phase_line(clock(self.started_at), "INFO", f"=== phase '{name}' started ==="))
        if description:
            self.log_file.writeline(phase_line(clock(), "INFO", description))

    # ------------------------------------------------------------------ logging
    @never_raise
    def _emit(self, level: str, msg: str, fields: dict[str, Any] | None = None) -> None:
        ts = now()
        line = phase_line(clock(ts), level, msg)
        if fields:
            line += "  " + dumps(fields)
        self.log_file.writeline(line)
        record: dict[str, Any] = {"type": "log", "ts": iso(ts), "level": level, "msg": msg}
        if fields:
            record["fields"] = fields
        self.events.append(record)
        if level in ("WARNING", "ERROR"):
            with self._lock:
                self.counters["warnings" if level == "WARNING" else "errors"] += 1
            if self._on_problem:
                self._on_problem(level, msg)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit("DEBUG", msg, fields or None)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit("INFO", msg, fields or None)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit("WARNING", msg, fields or None)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit("ERROR", msg, fields or None)

    @never_raise
    def raw(self, label: str, data: Any) -> None:
        """Dump a full raw value (model response, subprocess output...) into phase.log."""
        text = data if isinstance(data, str) else dumps(data, indent=2)
        self.log_file.writeline(raw_block(label, text))
        self.events.append({"type": "raw", "ts": iso(), "label": label, "chars": len(text)})

    @never_raise
    def console(self, stream: str, line: str) -> None:
        """Captured print()/stderr output (called by the console tee)."""
        self.log_file.writeline(console_line(clock(), stream, line))

    @never_raise
    def fail(self, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.error(f"{type(exc).__name__}: {exc}")
        self.raw("traceback", tb)

    # ------------------------------------------------------------------ artifacts
    def _unique_name(self, name: str) -> str:
        base = Path(name)
        candidate, n = base.name, 1
        with self._lock:
            while candidate in self._artifact_names or (self.artifacts_dir / candidate).exists():
                n += 1
                candidate = f"{base.stem}__{n}{base.suffix}"
            self._artifact_names.add(candidate)
        return candidate

    @never_raise
    def save_artifact(self, name: str, data: Any, *, quiet: bool = False) -> str | None:
        """Store a raw artifact. Returns its path relative to the run directory.

        str -> text, bytes -> binary, an existing file Path -> copied, anything else -> JSON.
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(name).suffix
        if isinstance(data, Path):
            if not data.is_file():
                self.warning(f"artifact source missing, not copied: {data}")
                return None
            if data.stat().st_size > self.cfg.max_copy_bytes:
                self.warning(f"artifact too large to copy ({data.stat().st_size} bytes): {data}")
                return None
            dest = self.artifacts_dir / self._unique_name(name or data.name)
            copy_redacted(data, dest, self._redactor)
        elif isinstance(data, (bytes, bytearray)):
            dest = self.artifacts_dir / self._unique_name(name if suffix else name + ".bin")
            dest.write_bytes(bytes(data))
        elif isinstance(data, str):
            dest = self.artifacts_dir / self._unique_name(name if suffix else name + ".txt")
            dest.write_text(self._redactor(data), encoding="utf-8")
        else:
            dest = self.artifacts_dir / self._unique_name(name if suffix else name + ".json")
            dest.write_text(self._redactor(dumps(data, indent=2)), encoding="utf-8")

        rel = dest.relative_to(self.run_dir).as_posix()
        with self._lock:
            self.counters["artifacts"] += 1
        self.events.append({"type": "artifact", "ts": iso(), "name": dest.name, "path": rel,
                            "bytes": dest.stat().st_size})
        if not quiet:
            self.log_file.writeline(
                phase_line(clock(), "INFO", f"artifact saved: {rel} ({dest.stat().st_size} bytes)"))
        return rel

    def _externalize(self, label: str, value: Any) -> Any:
        """Inline small values; park large ones in artifacts/ and return a pointer."""
        j = to_jsonable(value)
        text = j if isinstance(j, str) else json.dumps(j, ensure_ascii=False, default=repr)
        if len(text) <= self.cfg.inline_raw_max_chars:
            return j
        rel = self.save_artifact(label, j, quiet=True)
        return {"$artifact": rel, "chars": len(text), "preview": truncate(text, 200)}

    # ------------------------------------------------------------------ calls
    @never_raise
    def record_call(
        self,
        call_id: str,
        name: str,
        status: str,
        started: Any,
        ended: Any,
        duration_s: float,
        args: dict[str, Any] | None,
        result: Any = None,
        has_result: bool = False,
        error: dict[str, Any] | None = None,
    ) -> str | None:
        """Full raw record of one call. Returns '<phase dir>/events.jsonl#L<n>'."""
        record: dict[str, Any] = {
            "type": "call",
            "call_id": call_id,
            "name": name,
            "status": status,
            "started": iso(started),
            "ended": iso(ended),
            "duration_s": round(duration_s, 4),
            "args": {k: self._externalize(f"{call_id}_arg_{slug(k)}", v) for k, v in (args or {}).items()},
        }
        if has_result:
            record["result"] = self._externalize(f"{call_id}_result", result)
        if error:
            record["error"] = error
        with self._lock:
            self.counters["calls"] += 1
            if status != "ok":
                self.counters["calls_failed"] += 1
        line_no = self.events.append(record)
        return f"{self.rel}/events.jsonl#L{line_no}"

    # ------------------------------------------------------------------ lifecycle
    def summary(self) -> dict[str, Any]:
        end = self.ended_at or now()
        return {
            "name": self.name,
            "index": self.index,
            "description": self.description,
            "status": self.status,
            "dir": self.rel,
            "started": iso(self.started_at),
            "ended": iso(self.ended_at) if self.ended_at else None,
            "duration_s": round((end - self.started_at).total_seconds(), 3),
            **self.counters,
        }

    @never_raise
    def finish(self, status: str = "ok") -> dict[str, Any]:
        with self._lock:
            if self._finished:
                return self.summary()
            self._finished = True
            self.ended_at = now()
            if status == "ok" and self.counters["errors"]:
                status = "ok_with_errors"
            self.status = status
        s = self.summary()
        self.log_file.writeline(
            phase_line(clock(), "INFO",
                       f"=== phase '{self.name}' ended: {status} in {s['duration_s']}s "
                       f"(calls={s['calls']} failed={s['calls_failed']} "
                       f"warnings={s['warnings']} errors={s['errors']}) ==="))
        write_json_atomic(self.dir / "phase_summary.json", s, self._redactor)
        self.log_file.close()
        self.events.close()
        return s
