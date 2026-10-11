"""Process-wide access to the active run.

Pipeline code never passes a logger around. It asks the registry:

    with start_run("phase2_pipeline", meta={...}):      # entry point owns the run
        with phase("phase2_planner"):                   # one block per pipeline stage
            get_phase().info("hello")                   # raw output goes to the phase log

Everything degrades to a no-op when no run is active, so modules still work when they are
imported or run on their own.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import LogConfig
from .local_session import LocalSessionLogger
from .null import NULL_PHASE, NULL_SESSION
from .phase_logger import PhaseLogger
from .run_logger import RunLogger

_RUN: RunLogger | None = None
_LOCK = threading.RLock()


def init_run(
    entry: str,
    *,
    meta: dict[str, Any] | None = None,
    upstream_artifact: str | Path | None = None,
    config: LogConfig | None = None,
) -> RunLogger:
    """Start a run. If one is already active it is reused (nested entry points)."""
    global _RUN
    with _LOCK:
        if _RUN is not None and not _RUN.finished:
            _RUN.event("WARNING", f"init_run('{entry}') called while a run is active; reusing it")
            return _RUN
        _RUN = RunLogger(entry, config=config, meta=meta, upstream_artifact=upstream_artifact)
        return _RUN


def get_run() -> RunLogger | None:
    with _LOCK:
        return _RUN if (_RUN is not None and not _RUN.finished) else None


def end_run(status: str | None = None, error: BaseException | None = None) -> None:
    global _RUN
    with _LOCK:
        run = _RUN
        _RUN = None
    if run is not None:
        run.finish(status, error)


@contextmanager
def start_run(entry: str, **kwargs: Any) -> Iterator[RunLogger]:
    """Create the run for this entry point and finish it on exit (success or failure).
    If a run is already active the existing one is yielded and left for its owner to end."""
    existing = get_run()
    if existing is not None:
        yield existing
        return
    run = init_run(entry, **kwargs)
    try:
        yield run
    except BaseException as exc:
        end_run("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exc)
        raise
    else:
        end_run()


def get_phase() -> PhaseLogger:
    """The active phase logger (a harmless no-op object when no run is active)."""
    run = get_run()
    return run.current_phase() if run else NULL_PHASE  # type: ignore[return-value]


@contextmanager
def phase(name: str, description: str = "") -> Iterator[PhaseLogger]:
    run = get_run()
    if run is None:
        yield NULL_PHASE  # type: ignore[misc]
        return
    with run.phase(name, description) as pl:
        yield pl


@contextmanager
def local_session(**meta: Any) -> Iterator[LocalSessionLogger]:
    run = get_run()
    if run is None:
        yield NULL_SESSION  # type: ignore[misc]
        return
    with run.local_session(**meta) as sess:
        yield sess


def link_artifact(path: str | Path, kind: str) -> None:
    run = get_run()
    if run:
        run.link_artifact(path, kind)


def log_info(msg: str, **fields: Any) -> None:
    get_phase().info(msg, **fields)


def log_warning(msg: str, **fields: Any) -> None:
    get_phase().warning(msg, **fields)


def log_error(msg: str, **fields: Any) -> None:
    get_phase().error(msg, **fields)
