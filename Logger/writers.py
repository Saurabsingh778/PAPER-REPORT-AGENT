"""Thread-safe append-only writers (text, JSON lines) and atomic JSON files."""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

from .utils import to_jsonable

Redact = Callable[[str], str]


def _identity(text: str) -> str:
    return text


class TextLog:
    """Append-only text file. Each write is flushed so a crash never loses recent lines."""

    def __init__(self, path: Path, redact: Redact | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._redact = redact or _identity
        self._lock = threading.RLock()
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> None:
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(self._redact(text))
            self._fh.flush()

    def writeline(self, text: str = "") -> None:
        self.write(text if text.endswith("\n") else text + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class JsonlLog:
    """Append-only JSON-lines file. append() returns the 1-based line number so other
    logs can point at the exact record (e.g. 'events.jsonl#L12')."""

    def __init__(self, path: Path, redact: Redact | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._redact = redact or _identity
        self._lock = threading.RLock()
        self._count = 0
        if self.path.exists():
            with open(self.path, "rb") as fh:
                self._count = sum(1 for _ in fh)
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def append(self, record: dict[str, Any]) -> int:
        line = json.dumps(to_jsonable(record), ensure_ascii=False, default=repr)
        with self._lock:
            if self._fh.closed:
                return self._count
            self._fh.write(self._redact(line) + "\n")
            self._fh.flush()
            self._count += 1
            return self._count

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_json_atomic(path: Path, obj: Any, redact: Redact | None = None) -> None:
    """Write JSON via temp file + rename so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2, default=repr)
    if redact:
        text = redact(text)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def copy_redacted(src: Path, dest: Path, redact: Redact | None = None) -> None:
    """Copy a file; text files are scrubbed of secrets on the way, binary files copied as-is."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if redact:
        try:
            text = src.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            shutil.copy2(src, dest)
            return
        dest.write_text(redact(text), encoding="utf-8")
        return
    shutil.copy2(src, dest)
