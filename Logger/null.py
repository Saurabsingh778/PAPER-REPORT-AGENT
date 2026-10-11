"""No-op stand-ins used when no run is active, so library code can call the logger
unconditionally (e.g. when a module is imported or run outside the main pipeline)."""
from __future__ import annotations

from typing import Any


class _Null:
    name = "none"
    dir = None
    counters: dict[str, int] = {}

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any):
        return False

    def __bool__(self) -> bool:
        return False


class NullPhase(_Null):
    """Stands in for a PhaseLogger."""


class NullSession(_Null):
    """Stands in for a LocalSessionLogger."""


NULL_PHASE = NullPhase()
NULL_SESSION = NullSession()
