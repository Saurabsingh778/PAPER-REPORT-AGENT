"""Call instrumentation.

Each instrumented call produces two records:
  * a one-line SUMMARY in the run's main log (name, short args, status, duration, result)
  * a FULL raw record (complete args, result, traceback) in the active phase's events.jsonl
The summary carries a `raw:` pointer to the exact line of the raw record.
"""
from __future__ import annotations

import functools
import inspect
import time
import traceback
from typing import Any, Callable

from .formatters import format_args
from .registry import get_run
from .utils import describe, iso, never_raise, now, truncate


def _bind(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        mapping = dict(bound.arguments)
        for implicit in ("self", "cls"):
            mapping.pop(implicit, None)
        return mapping
    except (TypeError, ValueError):
        out: dict[str, Any] = {f"arg{i}": v for i, v in enumerate(args)}
        out.update(kwargs)
        return out


@never_raise
def _record(
    run,
    name: str,
    call_id: str,
    phase,
    arg_map: dict[str, Any],
    started,
    ended,
    duration_s: float,
    result: Any,
    has_result: bool,
    exc: BaseException | None,
    summarize: Callable[[Any], str] | None,
) -> None:
    status = "ok" if exc is None else "error"
    error_full = None
    error_text = None
    if exc is not None:
        error_text = truncate(f"{type(exc).__name__}: {exc}", run.cfg.summary_max_chars)
        error_full = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }

    raw_ref = phase.record_call(
        call_id=call_id, name=name, status=status, started=started, ended=ended,
        duration_s=duration_s, args=arg_map, result=result, has_result=has_result, error=error_full,
    )

    result_text = None
    if has_result and exc is None and result is not None:
        try:
            result_text = summarize(result) if summarize else describe(result, run.cfg.summary_max_chars)
        except Exception:  # a bad summarizer must not hide the call
            result_text = describe(result, run.cfg.summary_max_chars)

    run.log_call({
        "call_id": call_id,
        "name": name,
        "phase": phase.name,
        "status": status,
        "started": iso(started),
        "duration_s": round(duration_s, 4),
        "args": {k: describe(v, run.cfg.summary_max_chars) for k, v in arg_map.items()},
        "args_text": format_args(arg_map),
        "result": result_text,
        "error": error_text,
        "raw_ref": raw_ref,
    })


def _invoke(
    fn: Callable,
    name: str,
    args: tuple,
    kwargs: dict,
    summarize: Callable[[Any], str] | None,
    log_result: bool,
) -> Any:
    run = get_run()
    if run is None:
        return fn(*args, **kwargs)

    phase = run.current_phase()
    call_id = run.next_call_id()
    arg_map = _bind(fn, args, kwargs)
    started = now()
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:
        _record(run, name, call_id, phase, arg_map, started, now(), time.perf_counter() - t0,
                None, False, exc, summarize)
        raise
    _record(run, name, call_id, phase, arg_map, started, now(), time.perf_counter() - t0,
            result, log_result, None, summarize)
    return result


def logged(
    name: str | None = None,
    *,
    summarize: Callable[[Any], str] | None = None,
    log_result: bool = True,
) -> Callable:
    """Decorator form:

        @logged("fetch_repo", summarize=lambda path: f"saved to {path}")
        def download(url): ...
    """

    def decorator(fn: Callable) -> Callable:
        label = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _invoke(fn, label, args, kwargs, summarize, log_result)

        return wrapper

    return decorator


def call_logged(
    fn: Callable,
    *args: Any,
    _name: str | None = None,
    _summarize: Callable[[Any], str] | None = None,
    _log_result: bool = True,
    **kwargs: Any,
) -> Any:
    """Call-site form, for functions you do not want to edit:

        rows, n_ok = call_logged(compare, claims, run_out, _name="compare")
    """
    label = _name or getattr(fn, "__qualname__", getattr(fn, "__name__", "call"))
    return _invoke(fn, label, args, kwargs, _summarize, _log_result)
