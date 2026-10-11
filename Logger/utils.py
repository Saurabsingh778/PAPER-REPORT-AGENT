"""Small helpers shared by every logger module (no project imports, stdlib only)."""
from __future__ import annotations

import dataclasses
import enum
import functools
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_warned: set[str] = set()


def now() -> datetime:
    return datetime.now().astimezone()


def iso(dt: datetime | None = None) -> str:
    return (dt or now()).isoformat(timespec="milliseconds")


def clock(dt: datetime | None = None) -> str:
    return (dt or now()).strftime("%H:%M:%S.%f")[:-3]


def slug(text: str, max_len: int = 60) -> str:
    s = _SLUG_RE.sub("_", str(text)).strip("._-")
    return (s or "unnamed")[:max_len]


def new_run_id(entry: str) -> str:
    stamp = now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{slug(entry, 30)}_{stamp}_{uuid.uuid4().hex[:6]}"


def truncate(text: str, limit: int) -> str:
    """Cut text to `limit` chars and say how much was dropped."""
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} chars]"


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of arbitrary objects (pydantic models, Paths, exceptions...)
    into something json.dumps accepts. Never raises."""
    if _depth > 8:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, _depth + 1)
    if isinstance(obj, BaseException):
        return {"type": type(obj).__name__, "message": str(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if hasattr(obj, "model_dump"):  # pydantic v2
        try:
            return to_jsonable(obj.model_dump(mode="json"), _depth + 1)
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return to_jsonable(dataclasses.asdict(obj), _depth + 1)
        except Exception:
            pass
    return repr(obj)


def dumps(obj: Any, indent: int | None = None) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, indent=indent, default=repr)


def short_repr(obj: Any, limit: int = 300) -> str:
    """One-line, length-capped description of any value, for main-log summaries."""
    if isinstance(obj, str):
        text = obj.replace("\n", "\\n")
    elif isinstance(obj, (Path, int, float, bool)) or obj is None:
        text = str(obj)
    else:
        try:
            text = dumps(obj)
        except Exception:
            text = repr(obj)
        name = type(obj).__name__
        if name not in ("dict", "list", "tuple", "set"):
            text = f"{name} {text}"
    return truncate(text, limit)


def describe(obj: Any, limit: int = 80) -> str:
    """Content-light description for summary lines: small values are shown as-is, large
    structures collapse to their type and size (the full value lives in the raw record)."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return str(obj)
    if isinstance(obj, Path):
        return truncate(str(obj), max(limit, 160))
    if isinstance(obj, str):
        one_line = obj.replace("\n", "\\n")
        looks_like_path = "/" in obj and "\n" not in obj and " " not in obj.strip()
        return truncate(one_line, max(limit, 160) if looks_like_path else limit)
    if isinstance(obj, (bytes, bytearray)):
        return f"bytes[{len(obj)}]"
    if isinstance(obj, dict):
        compact = dumps(obj)
        return compact if len(compact) <= limit else f"dict[{len(obj)} keys]"
    if isinstance(obj, (list, tuple, set, frozenset)):
        compact = dumps(list(obj))
        return compact if len(compact) <= limit else f"{type(obj).__name__}[{len(obj)}]"
    text = repr(obj)
    return text if len(text) <= limit else type(obj).__name__


def never_raise(fn: Callable) -> Callable:
    """Logging must never take the pipeline down: swallow (and report once) any error
    raised by a logger method."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            key = f"{fn.__qualname__}:{type(exc).__name__}"
            if key not in _warned:
                _warned.add(key)
                try:
                    sys.__stderr__.write(f"[Logger] {fn.__qualname__} failed: {exc!r}\n")
                except Exception:
                    pass
            return None

    return wrapper
