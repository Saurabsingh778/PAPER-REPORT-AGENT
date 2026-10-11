"""LocalSessionLogger: the local Docker execution layer, logged on its own.

<run>/local_session/
    session.log / session.jsonl     lifecycle: workspace prep, GPU check, container start/exit
    session_summary.json            final status, timings, per-step table
    workspace.json                  what was copied / symlinked / mounted
    spec.json  runner.py            exactly what was handed to the container
    gpu_check.json                  nvidia-container-toolkit probe (raw stdout/stderr)
    docker/command.txt              the docker run command
    docker/container_stdout.log     live container output, timestamped line by line
    docker/exit.json                return code and wall time
    results/results.json            raw results.json produced inside the container
    results/steps/NN_<id>.json      per-step metadata (+ .stdout.log / .stderr.log)
    results/setup/NN_pip.json       per-package install records (+ logs)
    results/patches.json            path rewrites that were applied
    raw/                            FULL untruncated stdout/stderr per command (written by
                                    the in-container runner; results.json only keeps tails)
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .config import LogConfig
from .formatters import console_line
from .redaction import Redactor
from .session_logger import FinishHook, SessionLogger
from .utils import clock, human_duration, iso, never_raise, now, slug
from .writers import TextLog, read_json

_STEP_RE = re.compile(r"^\[>>\] Running Step '([^']+)': (.*)$")
_PIP_RE = re.compile(r"^\[\*\] Installing pip package: (.*)$")
_GPU_RE = re.compile(r"^Local GPU Info:\s*(.*)$")
_DONE_RE = re.compile(r"^\[✓\] Local execution complete")


def _step_status(rc: Any) -> str:
    if rc == 0:
        return "ok"
    if rc == -9:
        return "timeout"
    return "failed"


class LocalSessionLogger(SessionLogger):
    kind = "local_docker"

    def __init__(
        self,
        run_dir: Path,
        cfg: LogConfig,
        redactor: Redactor,
        phase_name: str | None = None,
        meta: dict[str, Any] | None = None,
        on_finish: FinishHook | None = None,
    ):
        super().__init__(run_dir, "local_session", cfg, redactor, phase_name, meta, on_finish)
        self._container_log: TextLog | None = None
        self._timeline: list[dict[str, Any]] = []
        self._container_started = None
        self._line_count = 0

    # ------------------------------------------------------------------ preparation
    @never_raise
    def workspace(
        self,
        runner_dir: Path,
        repo_src: Path,
        repo_copy: Path,
        data_src: Path,
        data_mounts: list[dict[str, Any]],
    ) -> None:
        """Record how the sandbox folder was assembled (repo copy, data symlinks)."""
        info = {
            "runner_dir": str(runner_dir),
            "repo_source": str(repo_src),
            "repo_working_copy": str(repo_copy),
            "data_source_dir": str(data_src),
            "data_mounts": data_mounts,
        }
        self.save_json("workspace.json", info)
        self.info("workspace prepared", repo_copy=str(repo_copy), data_mounts=len(data_mounts))
        for m in data_mounts:
            if not m.get("source"):
                self.warning(f"no downloaded data matched spec data '{m.get('id')}'", **m)
        self.details["workspace"] = {"data_mounts": len(data_mounts)}

    def save_spec(self, spec_json: str) -> None:
        self.save_text("spec.json", spec_json)

    def save_runner(self, script: str) -> None:
        self.save_text("runner.py", script)

    @never_raise
    def gpu_check(
        self,
        requested: bool,
        supported: bool | None,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        duration_s: float | None = None,
        error: str | None = None,
    ) -> None:
        record = {
            "gpu_requested": requested,
            "docker_gpu_supported": supported,
            "returncode": returncode,
            "duration_s": duration_s,
            "error": error,
            "stdout": stdout,
            "stderr": stderr,
        }
        self.save_json("gpu_check.json", record)
        if requested and not supported:
            self.warning("GPU requested but Docker NVIDIA runtime not available; falling back to CPU")
        else:
            self.info("GPU check", requested=requested, supported=supported)
        self.details["gpu"] = {"requested": requested, "docker_supported": supported,
                               "enabled": bool(requested and supported)}

    # ------------------------------------------------------------------ container
    @never_raise
    def docker_command(self, cmd: list[str]) -> None:
        self.save_text("docker/command.txt", shlex.join(str(c) for c in cmd) + "\n")
        self.info("docker command recorded", image=self.meta.get("image"))
        self.details.setdefault("docker", {})["command_file"] = "docker/command.txt"

    @never_raise
    def container_started(self) -> None:
        self._container_started = now()
        self._container_log = TextLog(self.dir / "docker" / "container_stdout.log", self._redactor)
        self.event("container_start")

    @never_raise
    def container_line(self, line: str) -> None:
        """Feed one line of live container output (called for every line)."""
        line = line.rstrip("\n")
        self._line_count += 1
        if self._container_log is None:
            self.container_started()
        self._container_log.writeline(console_line(clock(), "stdout", line))

        m = _STEP_RE.match(line)
        if m:
            self._close_open_step()
            self._timeline.append({"step_id": m.group(1), "command": m.group(2),
                                   "started": iso(), "_t0": now()})
            self.event("step_start", step_id=m.group(1), command=m.group(2))
            return
        m = _PIP_RE.match(line)
        if m:
            self.event("pip_install", package=m.group(1))
            return
        m = _GPU_RE.match(line)
        if m:
            self.event("container_gpu_info", info=m.group(1))
            return
        if _DONE_RE.match(line):
            self._close_open_step()
            self.event("runner_complete")

    def _close_open_step(self) -> None:
        if self._timeline and "ended" not in self._timeline[-1]:
            last = self._timeline[-1]
            last["ended"] = iso()
            last["wall_s"] = round((now() - last.pop("_t0")).total_seconds(), 3)

    @never_raise
    def container_exit(self, returncode: int | None) -> None:
        self._close_open_step()
        wall = None
        if self._container_started:
            wall = round((now() - self._container_started).total_seconds(), 3)
        self.save_json("docker/exit.json",
                       {"returncode": returncode, "wall_s": wall, "output_lines": self._line_count})
        level = "INFO" if returncode == 0 else "ERROR"
        self.event("container_exit", level, returncode=returncode, wall_s=wall)
        self.details.setdefault("docker", {}).update(returncode=returncode, wall_s=wall)
        if returncode not in (0, None):
            self.mark("failed")
        if self._container_log:
            self._container_log.close()

    # ------------------------------------------------------------------ results
    @never_raise
    def ingest_results(self, results_file: Path | None, raw_dir: Path | None = None) -> dict[str, Any] | None:
        """Split the container's results.json into per-step / per-setup files and copy the
        untruncated raw logs the in-container runner wrote."""
        if raw_dir and Path(raw_dir).is_dir():
            copied = 0
            for f in sorted(Path(raw_dir).iterdir()):
                if f.is_file() and self.copy_file(f"raw/{f.name}", f):
                    copied += 1
            self.info("raw command logs copied", count=copied)

        results_file = Path(results_file) if results_file else None
        if not results_file or not results_file.exists():
            self.error("results.json was not produced by the container")
            self.mark("failed")
            self.details["results"] = {"present": False}
            return None

        self.copy_file("results/results.json", results_file)
        results = read_json(results_file, {}) or {}

        steps_table = []
        for i, step in enumerate(results.get("steps", []), 1):
            base = f"results/steps/{i:02d}_{slug(step.get('step_id', 'step'))}"
            meta = {k: v for k, v in step.items() if k not in ("stdout", "stderr")}
            meta["stdout_chars"] = len(step.get("stdout") or "")
            meta["stderr_chars"] = len(step.get("stderr") or "")
            self.save_json(base + ".json", meta)
            self.save_text(base + ".stdout.log", step.get("stdout") or "")
            self.save_text(base + ".stderr.log", step.get("stderr") or "")
            status = _step_status(step.get("rc"))
            steps_table.append({"step_id": step.get("step_id"), "rc": step.get("rc"),
                                "status": status, "sec": step.get("sec"),
                                "covers": step.get("covers"), "file": base + ".json"})
            self.event("step_result", "INFO" if status == "ok" else "ERROR",
                       step_id=step.get("step_id"), rc=step.get("rc"), status=status,
                       sec=step.get("sec"))

        setup_table = []
        for i, item in enumerate(results.get("setup", []), 1):
            base = f"results/setup/{i:02d}_{slug(item.get('cmd', 'setup'))}"
            meta = {k: v for k, v in item.items() if k not in ("stdout", "stderr")}
            self.save_json(base + ".json", meta)
            self.save_text(base + ".stdout.log", item.get("stdout") or "")
            self.save_text(base + ".stderr.log", item.get("stderr") or "")
            setup_table.append({"cmd": item.get("cmd"), "rc": item.get("rc"), "sec": item.get("sec")})
            if item.get("rc") not in (0, None):
                self.warning(f"setup command failed: {item.get('cmd')}", rc=item.get("rc"))

        patches = results.get("patches", [])
        self.save_json("results/patches.json", patches)

        failed = [s for s in steps_table if s["status"] != "ok"]
        summary = {
            "present": True,
            "steps": steps_table,
            "setup": setup_table,
            "patches_applied": len(patches),
            "steps_ok": len(steps_table) - len(failed),
            "steps_failed": len(failed),
            "timeline": [{k: v for k, v in t.items() if not k.startswith("_")} for t in self._timeline],
        }
        self.details["results"] = summary
        if failed and self._forced_status is None:
            self.mark("ok_with_failures")
        return summary

    # ------------------------------------------------------------------ finish
    def describe(self) -> str:
        """One-line summary for the main log."""
        res = self.details.get("results", {})
        docker = self.details.get("docker", {})
        gpu = self.details.get("gpu", {})
        parts = [f"docker rc={docker.get('returncode')}"]
        if res.get("present"):
            parts.append(f"{res['steps_ok']}/{len(res['steps'])} steps ok")
        else:
            parts.append("no results.json")
        if gpu:
            parts.append(f"gpu={'on' if gpu.get('enabled') else 'off'}")
        parts.append(human_duration(self.summary().get("duration_s")))
        return ", ".join(parts)

    def finish(self, status: str = "ok"):
        if self._container_log:
            self._container_log.close()
        return super().finish(status)
