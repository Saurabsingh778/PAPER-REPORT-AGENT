"""Local Docker execution engine for RunSpec."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from Logger import local_session
from Schema.RunSpec import RunSpec

DOCKER_RUNNER_SCRIPT = r'''
import subprocess, json, time, os, sys, shutil

with open("/workspace/spec.json", "r", encoding="utf-8") as f:
    SPEC = json.load(f)

W = "/workspace"
REPO_DIR = os.path.join(W, "repo")
DATA_DIR = os.path.join(W, "data")
OUT_DIR = os.path.join(W, "out")
RAW_DIR = os.path.join(OUT_DIR, "raw")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

out = {"setup": [], "steps": [], "patches": []}

def _txt(x):
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else x

def write_raw(tag, cmd, cwd, rc, sec, so, se):
    """Full, untruncated output of one command (results.json only keeps the tails)."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)[:80]
    with open(os.path.join(RAW_DIR, safe + ".log"), "w", encoding="utf-8") as f:
        f.write(f"$ {cmd}\n# cwd={cwd} rc={rc} sec={sec}\n\n--- STDOUT ---\n{so}\n\n--- STDERR ---\n{se}\n")

def run(cmd, cwd, timeout, tag=None):
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        if tag:
            write_raw(tag, cmd, cwd, p.returncode, round(time.time() - t), p.stdout, p.stderr)
        return {"cmd": cmd, "rc": p.returncode, "sec": round(time.time() - t),
                "stdout": p.stdout[-8000:], "stderr": p.stderr[-4000:]}
    except subprocess.TimeoutExpired as e:
        so = e.stdout if isinstance(e.stdout, str) else ""
        if tag:
            write_raw(tag, cmd, cwd, -9, timeout, _txt(e.stdout), _txt(e.stderr) + "\nTIMEOUT")
        return {"cmd": cmd, "rc": -9, "sec": timeout, "stdout": so[-4000:], "stderr": "TIMEOUT"}

def save():
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

# Check GPU availability
if SPEC.get("hardware", {}).get("gpu") != "none":
    smi = run("nvidia-smi -L", W, 30)
    print("Local GPU Info:", smi["stdout"].strip())

# 1. Apply PathRewrite patches
for step in SPEC.get("steps", []):
    for patch in step.get("patches", []):
        fpath = os.path.join(REPO_DIR, patch["file"])
        if os.path.exists(fpath):
            content = open(fpath, "r", encoding="utf-8", errors="replace").read()
            old_val = patch["old"]
            new_val = patch["new"].replace("{REPO}", REPO_DIR).replace("{DATA}", DATA_DIR)
            if old_val in content:
                content = content.replace(old_val, new_val)
                open(fpath, "w", encoding="utf-8").write(content)
                out["patches"].append({"file": patch["file"], "status": "applied"})

# 2. Install pip dependencies
for n, pkg in enumerate(SPEC.get("environment", {}).get("pip", []), 1):
    print(f"[*] Installing pip package: {pkg}")
    res = run(f"pip install -q {pkg}", REPO_DIR, 600, tag=f"pip_{n:02d}_{pkg}")
    out["setup"].append(res)
save()

# 3. Execute steps in order
done = {}
for s in SPEC["steps"]:
    cmd = s["command"].replace("{REPO}", REPO_DIR).replace("{DATA}", DATA_DIR)
    cwd = s["cwd"].replace("{REPO}", REPO_DIR).replace("{DATA}", DATA_DIR)
    timeout = s["timeout_min"] * 60

    print(f"\n[>>] Running Step '{s['id']}': {cmd}")
    if cmd in done:
        r = dict(done[cmd])
    else:
        r = run(cmd, cwd, timeout, tag=f"step_{s['id']}")
        done[cmd] = r

    r["step_id"] = s["id"]
    r["covers"] = s["covers"]
    r["claim_indices"] = s["covers"]
    r["signals"] = s["signals"]
    out["steps"].append(r)
    save()

print("\n[✓] Local execution complete. results.json generated.")
'''


def probe_docker_gpu() -> Dict[str, Any]:
    """Probes Docker's NVIDIA runtime and returns the raw outcome (for logging)."""
    t0 = time.time()
    try:
        res = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.0.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {"supported": res.returncode == 0, "returncode": res.returncode,
                "stdout": res.stdout, "stderr": res.stderr, "duration_s": round(time.time() - t0, 3)}
    except Exception as exc:
        return {"supported": False, "returncode": None, "stdout": "", "stderr": "",
                "duration_s": round(time.time() - t0, 3), "error": f"{type(exc).__name__}: {exc}"}


def check_docker_gpu_support() -> bool:
    """Verifies whether Docker has access to NVIDIA GPUs via nvidia-container-toolkit."""
    return probe_docker_gpu()["supported"]


def run_locally(
    spec: RunSpec,
    workdir: Path,
    repo_dir: Path,
    data_dir: Path,
    image: str = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
) -> Optional[Dict[str, Any]]:
    """Runs the RunSpec inside a local Docker container.

    Everything about the container session (workspace prep, GPU probe, docker command,
    live output, per-step results, raw command logs) is recorded in the run's separate
    local_session/ log layer.
    """
    with local_session(image=image, workdir=str(workdir), gpu_policy=spec.hardware.gpu) as sess:
        return _run_locally(spec, workdir, repo_dir, data_dir, image, sess)


def _run_locally(
    spec: RunSpec,
    workdir: Path,
    repo_dir: Path,
    data_dir: Path,
    image: str,
    sess,
) -> Optional[Dict[str, Any]]:
    workdir = Path(workdir).resolve()
    repo_dir = Path(repo_dir).resolve()
    data_dir = Path(data_dir).resolve()

    runner_dir = workdir / "local_run"
    runner_dir.mkdir(parents=True, exist_ok=True)
    out_dir = runner_dir / "out"
    out_dir.mkdir(exist_ok=True)
    # Drop leftovers from an earlier run so stale results are never read or logged as new.
    shutil.rmtree(out_dir / "raw", ignore_errors=True)
    (out_dir / "results.json").unlink(missing_ok=True)

    # 1. Create a working copy of repo to avoid modifying source downloads
    working_repo_dir = runner_dir / "repo"
    if working_repo_dir.exists():
        shutil.rmtree(working_repo_dir)
    shutil.copytree(repo_dir, working_repo_dir)

    # 2. Prepare data directory mount (matching spec.data dest mapping)
    working_data_dir = runner_dir / "data"
    working_data_dir.mkdir(exist_ok=True)
    data_mounts = []
    for d in spec.data:
        dest_name = d.dest or d.id
        target = working_data_dir / dest_name
        # Find matching downloaded data folder
        available = list(data_dir.iterdir()) if data_dir.exists() else []
        candidates = list(data_dir.glob(f"*{dest_name}*")) if data_dir.exists() else []
        candidates = candidates or available
        mount = {"id": d.id, "dest": dest_name, "kind": d.kind, "source": None, "mode": "none",
                 "candidates": [str(c) for c in candidates]}
        if candidates and not target.exists():
            # Link or copy to working_data_dir/dest
            src = candidates[0]
            if os.name == "posix":
                target.symlink_to(src.resolve(), target_is_directory=True)
                mount.update(source=str(src.resolve()), mode="symlink")
            else:
                shutil.copytree(src, target)
                mount.update(source=str(src), mode="copy")
        data_mounts.append(mount)
    sess.workspace(runner_dir, repo_dir, working_repo_dir, data_dir, data_mounts)

    # 3. Write runner files
    (runner_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (runner_dir / "runner.py").write_text(DOCKER_RUNNER_SCRIPT, encoding="utf-8")
    sess.save_spec(spec.model_dump_json(indent=2))
    sess.save_runner(DOCKER_RUNNER_SCRIPT)

    # 4. Configure Docker command
    gpu_requested = spec.hardware.gpu in ("required", "helpful")
    gpu_args = []
    if gpu_requested:
        probe = probe_docker_gpu()
        sess.gpu_check(
            requested=True,
            supported=probe["supported"],
            returncode=probe.get("returncode"),
            stdout=probe.get("stdout", ""),
            stderr=probe.get("stderr", ""),
            duration_s=probe.get("duration_s"),
            error=probe.get("error"),
        )
        if probe["supported"]:
            gpu_args = ["--gpus", "all"]
            print("[✓] NVIDIA Container Toolkit detected. Enabling local GPU (--gpus all).")
        else:
            print("[!] Warning: GPU was requested but Docker NVIDIA runtime was not detected. Falling back to CPU.")
    else:
        sess.gpu_check(requested=False, supported=None)

    docker_cmd = [
        "docker", "run", "--rm",
        *gpu_args,
        "-v", f"{runner_dir / 'spec.json'}:/workspace/spec.json:ro",
        "-v", f"{runner_dir / 'runner.py'}:/workspace/runner.py:ro",
        "-v", f"{working_repo_dir}:/workspace/repo",
        "-v", f"{working_data_dir}:/workspace/data",
        "-v", f"{out_dir}:/workspace/out",
        "-w", "/workspace",
        image,
        "python", "/workspace/runner.py",
    ]

    sess.docker_command(docker_cmd)
    print(f"[*] Starting Docker container ({image})...")
    process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sess.container_started()

    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        sess.container_line(line)
    process.wait()
    sess.container_exit(process.returncode)

    results_file = out_dir / "results.json"
    sess.ingest_results(results_file if results_file.exists() else None, out_dir / "raw")
    if results_file.exists():
        return json.loads(results_file.read_text(encoding="utf-8"))
    return None