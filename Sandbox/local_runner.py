"""Local Docker execution engine for RunSpec."""

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from Logger import local_session
from Schema.RunSpec import RunSpec, find_data_folder

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


DEFAULT_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"


def probe_docker_gpu(image: str = DEFAULT_IMAGE, timeout: int = 180) -> Dict[str, Any]:
    """Probes Docker's NVIDIA runtime and returns the raw outcome (for logging).

    Uses the image the run will use anyway (so no second image is pulled) and a generous
    timeout: a first-time pull easily exceeds a few seconds and must not read as "no GPU".
    """
    t0 = time.time()
    try:
        res = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", image, "nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {"supported": res.returncode == 0, "returncode": res.returncode,
                "stdout": res.stdout, "stderr": res.stderr, "duration_s": round(time.time() - t0, 3)}
    except Exception as exc:
        return {"supported": False, "returncode": None, "stdout": "", "stderr": "",
                "duration_s": round(time.time() - t0, 3), "error": f"{type(exc).__name__}: {exc}"}


def check_docker_gpu_support(image: str = DEFAULT_IMAGE) -> bool:
    """Verifies whether Docker has access to NVIDIA GPUs via nvidia-container-toolkit."""
    return probe_docker_gpu(image)["supported"]


def run_locally(
    spec: RunSpec,
    workdir: Path,
    repo_dir: Path,
    data_dir: Path,
    image: str = DEFAULT_IMAGE,
) -> Optional[Dict[str, Any]]:
    """Runs the RunSpec inside a local Docker container.

    Everything about the container session (workspace prep, GPU probe, docker command,
    live output, per-step results, raw command logs) is recorded in the run's separate
    local_session/ log layer.
    """
    with local_session(image=image, workdir=str(workdir), gpu_policy=spec.hardware.gpu) as sess:
        return _run_locally(spec, workdir, repo_dir, data_dir, image, sess)


_IGNORE_VCS = shutil.ignore_patterns(".git", ".hg", "__pycache__")


def _stage_data(spec: RunSpec, repo_dir: Path, data_dir: Path, working_data_dir: Path) -> List[Dict[str, Any]]:
    """Copy everything under data_dir into working_data_dir (same relative layout, so
    {DATA}/<anything the planner saw> resolves), then alias each spec.data entry whose
    dest/id is not already a folder there (relative symlink, valid inside the container).

    Real copies, NOT absolute symlinks: an absolute host path does not exist in the container
    (it dangled, so the step saw no data), and steps may write next to their input data,
    which must not touch ./Downloads.
    """
    if data_dir.exists():
        for child in sorted(data_dir.iterdir()):
            if child.name.startswith("."):
                continue
            target = working_data_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, ignore=_IGNORE_VCS)
            else:
                shutil.copy2(child, target)

    mounts: List[Dict[str, Any]] = []
    for d in spec.data:
        name = d.dest or d.id
        entry = {"id": d.id, "dest": name, "kind": d.kind, "source": None, "mode": "none",
                 "candidates": sorted(p.name for p in working_data_dir.iterdir())}
        if d.kind == "in_repo" and (repo_dir / d.ref).exists():
            entry.update(source=str(repo_dir / d.ref), mode="in_code_repo")   # reached via {REPO}
        elif (working_data_dir / name).exists():
            entry.update(source=str(working_data_dir / name), mode="direct")
        else:
            first_seg = d.ref.split("/", 1)[0] if d.kind == "in_repo" else ""
            hit = find_data_folder(working_data_dir, name, d.id, first_seg)
            if hit is not None:
                alias = working_data_dir / name
                try:
                    alias.symlink_to(hit.name, target_is_directory=True)   # relative link
                    entry.update(source=str(hit), mode="alias")
                except OSError:
                    shutil.copytree(hit, alias)
                    entry.update(source=str(hit), mode="copy")
        mounts.append(entry)
    return mounts


def missing_data_paths(spec: RunSpec, working_data_dir: Path) -> List[tuple]:
    """(step_id, relative path) for every {DATA}/... argument in a step command that does not
    exist in the staged data. Output locations a step creates itself are not flagged: only
    paths whose parent folder is also absent are reported."""
    out = []
    for s in spec.steps:
        try:
            toks = shlex.split(s.command)
        except ValueError:
            continue
        for t in toks:
            bare = t.split("=", 1)[-1]
            if not bare.startswith("{DATA}"):
                continue
            rel = bare.removeprefix("{DATA}").lstrip("/")
            p = working_data_dir / rel
            if not p.exists() and not p.parent.exists():
                out.append((s.id, rel))
    return out


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

    # 2. Stage data. The planner is shown the downloaded data as paths relative to {DATA}
    #    (e.g. "digital-synthesis-lab_DM2/demo/..."), so {DATA} inside the container must have
    #    exactly that layout. Staging one folder under spec.data's dest/id instead (the old
    #    behaviour) moved it to /workspace/data/<id>/ and every {DATA}/<repo>/... path broke.
    working_data_dir = runner_dir / "data"
    shutil.rmtree(working_data_dir, ignore_errors=True)  # rebuilt each run (drops stale copies/links)
    working_data_dir.mkdir(exist_ok=True)
    data_mounts = _stage_data(spec, repo_dir, data_dir, working_data_dir)
    missing = missing_data_paths(spec, working_data_dir)
    for step_id, path in missing:
        print(f"[!] Step '{step_id}' uses {{DATA}}/{path}, which does not exist in the staged data.")
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
        probe = probe_docker_gpu(image)
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