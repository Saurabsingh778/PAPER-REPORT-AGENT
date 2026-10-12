"""Small helpers shared by the Pipeline modules (time, atomic JSON/text writes, hashing)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """Local time with UTC offset, second precision."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so a reader (a UI polling the run) never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(path: str | Path, data: Any) -> None:
    _atomic_write(Path(path), json.dumps(data, indent=2, ensure_ascii=False, default=str))


def write_text_atomic(path: str | Path, text: str) -> None:
    _atomic_write(Path(path), text)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def posix_rel(path: str | Path, base: str | Path) -> str:
    """`path` relative to `base`, with forward slashes (stable in JSON and links)."""
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()
