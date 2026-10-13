"""Stop the `llama-server` that Surya/Marker starts for a PDF conversion.

Why this exists: with Surya's `llamacpp` backend, `llama-server` is launched detached (its own
session) and Surya only stops it when the Python process exits. In a long-lived process such as
the UI server it therefore keeps holding VRAM after the conversion has finished. torch's
`empty_cache()` cannot release that memory because it belongs to a different process.

How it stays safe (this must never take the workflow down):
  * Surya records the server it spawned in ~/.cache/datalab/surya/llamacpp_server.json. We read the
    pid there BEFORE the conversion and again AFTER it. Only a server that appeared during this
    conversion is stopped; one that was already running (started by you or another program) is left
    alone.
  * A pid is only signalled after its command line is checked to really be a llama-server, so a
    recycled pid can never hit an unrelated process.
  * Only the llama-server pid is signalled: never this process, its parent, or other Python/CUDA
    processes.
  * Nothing here raises. Every outcome is returned as a dict so the caller can log it.
  * Surya starts a fresh server on the next conversion if none is running, so stopping it between
    conversions is transparent (the cost is reloading the model into the server next time).

Set MARKER_KEEP_LLAMA_SERVER=1 (or Surya's own SURYA_INFERENCE_KEEP_ALIVE=1) to leave it running.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

KEEP_ENV_VARS = ("MARKER_KEEP_LLAMA_SERVER", "SURYA_INFERENCE_KEEP_ALIVE")
_TRUTHY = {"1", "true", "yes", "on"}

DEFAULT_GRACE_S = 15.0   # time a server gets to exit after SIGTERM before SIGKILL


@dataclass(frozen=True)
class LlamaServerSnapshot:
    """What was recorded as running before the conversion started."""

    pid: int | None


# ----------------------------------------------------------------------------- Surya's sentinel
def _sentinel_path() -> Path:
    # Same file Surya's attach_or_spawn() writes for the "llamacpp" backend.
    return Path("~/.cache/datalab/surya/llamacpp_server.json").expanduser()


def _read_sentinel_pid() -> int | None:
    try:
        data = json.loads(_sentinel_path().read_text())
        pid = data.get("pid")
        return int(pid) if pid else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _remove_sentinel_if(pid: int) -> None:
    """Delete the sentinel, but only if it still points at the server we just stopped."""
    try:
        if _read_sentinel_pid() == pid:
            _sentinel_path().unlink()
    except OSError:
        pass


# ----------------------------------------------------------------------------- process checks
def _command_line(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        pass
    try:  # no /proc (macOS)
        res = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=5)
        return res.stdout.strip()
    except Exception:
        return ""


def is_llama_server(pid: int) -> bool:
    """True only if `pid` is alive and its command line is a llama-server."""
    parts = _command_line(pid).split()
    if not parts:
        return False
    exe = Path(parts[0]).name.lower()
    return "llama-server" in exe or ("llama" in exe and "--mmproj" in parts)


def _is_zombie(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rpartition(")")[2].split()[0] == "Z"
    except (OSError, IndexError):
        pass
    try:
        res = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, timeout=5)
        return res.stdout.strip().startswith("Z")
    except Exception:
        return False


def _exited(pid: int) -> bool:
    """True once the process is gone. A finished child that nobody reaped yet is reaped here."""
    if hasattr(os, "waitpid") and hasattr(os, "WNOHANG"):
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return True
        except OSError:  # not our child, or already reaped
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return _is_zombie(pid)


def _wait_exit(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _exited(pid):
            return True
        time.sleep(0.2)
    return _exited(pid)


def _terminate(pid: int, grace_s: float) -> bool:
    """SIGTERM, wait, then SIGKILL. Returns True once the process is gone (its VRAM is freed)."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    if _wait_exit(pid, grace_s):
        return True
    try:
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return _wait_exit(pid, 5.0)


def _keep_requested() -> str | None:
    for name in KEEP_ENV_VARS:
        if os.getenv(name, "").strip().lower() in _TRUTHY:
            return name
    return None


# ----------------------------------------------------------------------------- public API
def snapshot_llama_server() -> LlamaServerSnapshot:
    """Call BEFORE the models are created, so a pre-existing server can be told apart later."""
    return LlamaServerSnapshot(pid=_read_sentinel_pid())


def stop_new_llama_server(before: LlamaServerSnapshot, grace_s: float = DEFAULT_GRACE_S) -> dict:
    """Stop the llama-server started during this conversion. Never raises.

    Returns {"action": "stopped" | "kept" | "none" | "failed", "pid": int | None, "reason": str}.
    """
    pid: int | None = None
    try:
        keep = _keep_requested()
        if keep:
            return {"action": "kept", "pid": None, "reason": f"{keep} is set"}

        pid = _read_sentinel_pid()
        if pid is None:
            return {"action": "none", "pid": None, "reason": "no llama-server was recorded"}
        if pid == before.pid:
            return {"action": "kept", "pid": pid,
                    "reason": "this llama-server was already running before the conversion; left alone"}
        if not is_llama_server(pid):
            return {"action": "none", "pid": pid, "reason": "recorded pid is not a running llama-server"}

        if _terminate(pid, grace_s):
            _remove_sentinel_if(pid)
            return {"action": "stopped", "pid": pid, "reason": "llama-server stopped, its VRAM is released"}
        return {"action": "failed", "pid": pid, "reason": "llama-server did not exit"}
    except Exception as exc:  # cleanup must never break the pipeline
        return {"action": "failed", "pid": pid, "reason": f"{type(exc).__name__}: {exc}"}