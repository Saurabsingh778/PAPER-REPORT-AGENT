"""Logging configuration: where logs live and how much of each value is kept inline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Logger/ sits directly under the project root, so this is stable no matter which
# directory a script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "Logs"


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _default_root() -> Path:
    raw = os.getenv("AGENT_LOG_ROOT")
    if not raw:
        return DEFAULT_LOG_ROOT
    p = Path(raw).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class LogConfig:
    """Tunables. Every field can be overridden per run via init_run(config=LogConfig(...)).

    Environment overrides: AGENT_LOG_ROOT, AGENT_LOG_REDACT, AGENT_LOG_CAPTURE,
    AGENT_LOG_ECHO.
    """

    root: Path = field(default_factory=_default_root)

    # Secrets (API keys, tokens) are scrubbed from every text/JSON line before it hits disk.
    redact: bool = field(default_factory=lambda: _env_bool("AGENT_LOG_REDACT", True))

    # Tee sys.stdout / sys.stderr into the active phase log so plain print() output is kept.
    capture_console: bool = field(default_factory=lambda: _env_bool("AGENT_LOG_CAPTURE", True))

    # Also print main-log summary lines to the real console (off: the pipeline already prints).
    echo_main: bool = field(default_factory=lambda: _env_bool("AGENT_LOG_ECHO", False))

    # Main log keeps short summaries: any single value is cut to this many characters.
    summary_max_chars: int = 300

    # Phase events.jsonl keeps raw values inline up to this size; bigger values are written
    # to the phase's artifacts/ folder and referenced from the event instead.
    inline_raw_max_chars: int = 20_000

    # Largest file a phase/session will copy into its artifacts folder.
    max_copy_bytes: int = 50 * 1024 * 1024

    def runs_dir(self) -> Path:
        return Path(self.root) / "runs"

    def index_path(self) -> Path:
        return Path(self.root) / "index.jsonl"

    def links_path(self) -> Path:
        return Path(self.root) / "links.json"
