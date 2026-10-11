"""Logger: unified, hierarchical logging for the whole paper-reproduction pipeline.

This file is the single entry point - import everything from here:

    from Logger import start_run, phase, get_phase, call_logged, logged, local_session

Layout of one run (Logs/runs/<run_id>/):

    main.log, main.jsonl     summary of every call + phase/session boundaries + all warnings
    manifest.json            run metadata and the phase/session index
    summary.md               readable report, written when the run ends
    phases/<NN>_<name>/      raw output of each phase (phase.log, events.jsonl, artifacts/)
    local_session/           the local Docker layer, logged separately

Modules (one responsibility each):
    config.py          LogConfig - root dir, truncation limits, switches
    utils.py           time, slug, truncate, JSON-safe conversion
    redaction.py       secret scrubbing
    writers.py         thread-safe text / JSONL / atomic JSON writers
    formatters.py      line and block layouts
    capture.py         stdout/stderr tee into the active phase
    phase_logger.py    PhaseLogger - raw per-phase record
    session_logger.py  SessionLogger base for execution environments
    local_session.py   LocalSessionLogger - local Docker layer
    run_logger.py      RunLogger - main log, manifest, phase/session lifecycle
    index.py           cross-run index and artifact links
    summary.py         summary.md generation
    decorators.py      @logged and call_logged call instrumentation
    registry.py        process-wide access (init_run, start_run, phase, get_phase, ...)
    null.py            no-op stand-ins when no run is active
    cli.py             `python -m Logger.cli ...` log browser
"""
from .config import DEFAULT_LOG_ROOT, PROJECT_ROOT, LogConfig
from .decorators import call_logged, logged
from .index import RunIndex
from .local_session import LocalSessionLogger
from .null import NULL_PHASE, NULL_SESSION
from .phase_logger import PhaseLogger
from .redaction import Redactor
from .registry import (
    end_run,
    get_phase,
    get_run,
    init_run,
    link_artifact,
    local_session,
    log_error,
    log_info,
    log_warning,
    phase,
    start_run,
)
from .run_logger import RunLogger
from .session_logger import SessionLogger
from .summary import build_summary_md

__all__ = [
    "DEFAULT_LOG_ROOT", "PROJECT_ROOT", "LogConfig",
    "init_run", "start_run", "end_run", "get_run", "get_phase", "phase", "local_session",
    "link_artifact", "log_info", "log_warning", "log_error",
    "logged", "call_logged",
    "RunLogger", "PhaseLogger", "SessionLogger", "LocalSessionLogger", "RunIndex",
    "Redactor", "build_summary_md", "NULL_PHASE", "NULL_SESSION",
]
