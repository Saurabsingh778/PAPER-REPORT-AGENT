"""Tee sys.stdout / sys.stderr so every print() and library warning lands in the active
phase log, while still appearing on the real console."""
from __future__ import annotations

import re
import sys
import threading
from typing import Callable

_SPLIT = re.compile(r"(\r\n|\n|\r)")


class LineBuffer:
    """Turns arbitrary write() chunks into whole lines.

    A bare carriage return (tqdm-style progress bars) discards the half-written line, so
    only the final state of a progress bar is logged instead of hundreds of updates.
    """

    def __init__(self, emit: Callable[[str], None]):
        self._emit = emit
        self._buf = ""

    def feed(self, text: str) -> None:
        for part in _SPLIT.split(text):
            if part in ("\n", "\r\n"):
                self._emit(self._buf)
                self._buf = ""
            elif part == "\r":
                self._buf = ""
            else:
                self._buf += part

    def flush(self) -> None:
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


class _Tee:
    def __init__(self, stream, buffer: LineBuffer, guard: threading.local):
        self._stream = stream
        self._buffer = buffer
        self._guard = guard

    def write(self, data):
        written = self._stream.write(data)
        if isinstance(data, str) and not getattr(self._guard, "busy", False):
            self._guard.busy = True
            try:
                self._buffer.feed(data)
            except Exception:  # logging must never break printing
                pass
            finally:
                self._guard.busy = False
        return written

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):  # isatty(), fileno(), encoding, ...
        return getattr(self._stream, name)


class ConsoleCapture:
    """sink(stream_name, line) is called once per complete line of console output."""

    def __init__(self, sink: Callable[[str, str], None]):
        self._sink = sink
        self._guard = threading.local()
        self._orig_out = None
        self._orig_err = None
        self._tee_out = None
        self._tee_err = None
        self._buf_out = LineBuffer(lambda line: self._sink("stdout", line))
        self._buf_err = LineBuffer(lambda line: self._sink("stderr", line))

    def install(self) -> None:
        if self._orig_out is not None:
            return
        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        self._tee_out = _Tee(sys.stdout, self._buf_out, self._guard)
        self._tee_err = _Tee(sys.stderr, self._buf_err, self._guard)
        sys.stdout, sys.stderr = self._tee_out, self._tee_err

    def uninstall(self) -> None:
        if self._orig_out is None:
            return
        self._buf_out.flush()
        self._buf_err.flush()
        # Only restore a stream if nobody replaced our tee in the meantime.
        if sys.stdout is self._tee_out:
            sys.stdout = self._orig_out
        if sys.stderr is self._tee_err:
            sys.stderr = self._orig_err
        self._orig_out = self._orig_err = None
