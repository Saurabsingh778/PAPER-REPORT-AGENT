"""Local Docker execution engine for RunSpec, with 5x LLM environment retry and latest-version fallback."""

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from EnvirementSetup import (
    MAX_SETUP_ATTEMPTS,
    run_with_env_retry,
    unpin_repo_manifests,
)
from Logger import local_session
from Sandbox.docker_resolver import resolve_docker_image
from Schema.RunSpec import Environment, RunSpec, find_data_folder

DOCKER_RUNNER_SCRIPT = r'''
import subprocess, json, time, os, sys, shutil, shlex

with open("/workspace/spec.json", "r", encoding="utf-8") as f:
    SPEC = json.load(f)

W = "/workspace"
REPO_DIR = os.path.join(W, "repo")
DATA_DIR = os.path.join(W, "data")
OUT_DIR = os.path.join(W, "out")
RAW_DIR = os.path.join(OUT_DIR, "raw")
COMPAT_DIR = os.path.join(W, "compat")
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

ENV = SPEC.get("environment", {})
PACKAGES = ENV.get("packages", {})
METADATA = {
    m.get("key"): m.get("value")
    for m in ENV.get("metadata", [])
    if isinstance(m, dict) and m.get("key")
}
IS_LATEST_FALLBACK = METADATA.get("setup_mode") == "latest_fallback"

for item in ENV.get("env_vars", []):
    if isinstance(item, dict):
        key, value = item.get("name"), item.get("value")
        if key is not None:
            os.environ[str(key)] = str(value)

out["environment"] = {
    "requested_language": ENV.get("language", "python"),
    "requested_language_version": ENV.get("language_version"),
    "requirements_policy": ENV.get("requirements_policy", "resolved"),
    "setup_mode": "latest_fallback" if IS_LATEST_FALLBACK else "standard",
}

if not IS_LATEST_FALLBACK and ENV.get("language", "python").lower() == "python" and ENV.get("language_version"):
    expected = ".".join(str(ENV["language_version"]).split(".")[:2])
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != expected:
        out["setup"].append({"cmd": "python --version", "rc": 2, "sec": 0,
                             "stdout": actual,
                             "stderr": f"RunSpec requires Python {expected}, image provides Python {actual}"})
        out["setup_failed"] = True
        save()
        print(f"[x] RunSpec requires Python {expected}, image provides Python {actual}")
        sys.exit(2)

# In latest-version fallback mode, inject a lightweight sitecustomize compatibility bridge
# so legacy scripts that rely on deprecated NumPy/SciPy/NetworkX/TF1 aliases still run
# when started with the latest versions of all modules.
if IS_LATEST_FALLBACK:
    os.makedirs(COMPAT_DIR, exist_ok=True)
    compat_lines = [
        "import sys",
        "try:",
        "    import numpy as _np",
        "    for _attr, _builtin in (('bool', bool), ('int', int), ('float', float), ('object', object), ('str', str)):",
        "        if _attr not in _np.__dict__:",
        "            setattr(_np, _attr, _builtin)",
        "except Exception:",
        "    pass",
        "try:",
        "    import scipy.sparse as _sp",
        "    import scipy.sparse.linalg as _spla",
        "    import types as _types",
        "    if 'scipy.sparse.linalg.eigen' not in sys.modules:",
        "        _eig_mod = _types.ModuleType('scipy.sparse.linalg.eigen')",
        "        _eig_mod.arpack = _spla",
        "        sys.modules['scipy.sparse.linalg.eigen'] = _eig_mod",
        "        sys.modules['scipy.sparse.linalg.eigen.arpack'] = _spla",
        "except Exception:",
        "    pass",
        "try:",
        "    import networkx as _nx",
        "    import scipy.sparse as _sp",
        "    _orig_adj = getattr(_nx, 'adjacency_matrix', None)",
        "    if _orig_adj is not None:",
        "        def _compat_adj(*a, **kw):",
        "            m = _orig_adj(*a, **kw)",
        "            return _sp.csr_matrix(m) if not isinstance(m, _sp.spmatrix) else m",
        "        _nx.adjacency_matrix = _compat_adj",
        "except Exception:",
        "    pass",
    ]
    with open(os.path.join(COMPAT_DIR, "sitecustomize.py"), "w", encoding="utf-8") as sc:
        sc.write("\n".join(compat_lines) + "\n")
    existing_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{COMPAT_DIR}:{REPO_DIR}" + (f":{existing_pp}" if existing_pp else "")

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

# 2. Install the resolved environment
apt_packages = PACKAGES.get("apt", ENV.get("system_packages", []))
if apt_packages:
    out["setup"].append(run("apt-get update -qq", REPO_DIR, 600, tag="apt_update"))
    cmd = "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends " + " ".join(
        shlex.quote(str(pkg)) for pkg in apt_packages)
    out["setup"].append(run(cmd, REPO_DIR, 900, tag="apt_install"))

policy = ENV.get("requirements_policy", "resolved")
if policy == "install_as_is" and not IS_LATEST_FALLBACK:
    for n, rel in enumerate(ENV.get("requirements_files", []), 1):
        path = os.path.join(REPO_DIR, rel)
        res = run(f"{shlex.quote(sys.executable)} -m pip install -q --retries 5 --timeout 1200 -r {shlex.quote(path)}", REPO_DIR, 900,
                  tag=f"requirements_{n:02d}")
        out["setup"].append(res)

pip_packages = PACKAGES.get("pip", ENV.get("pip", []))
if pip_packages:
    mode_label = "latest unpinned" if IS_LATEST_FALLBACK else "resolved"
    print(f"[*] Installing {mode_label} pip environment ({len(pip_packages)} packages)")
    upgrade_flag = "--upgrade " if IS_LATEST_FALLBACK else ""
    requirements = " ".join(shlex.quote(str(pkg)) for pkg in pip_packages)
    res = run(f"{shlex.quote(sys.executable)} -m pip install -q {upgrade_flag}--retries 5 --timeout 1200 {requirements}", REPO_DIR, 1800,
              tag="pip_resolved")
    out["setup"].append(res)

conda_packages = PACKAGES.get("conda", [])
if conda_packages:
    cmd = "conda install -y " + " ".join(shlex.quote(str(pkg)) for pkg in conda_packages)
    out["setup"].append(run(cmd, REPO_DIR, 1200, tag="conda_install"))

custom_managers = PACKAGES.get("custom", [])
if isinstance(custom_managers, dict):
    custom_managers = [{"manager": name, "packages": packages} for name, packages in custom_managers.items()]
for custom in custom_managers:
    manager, packages = custom.get("manager"), custom.get("packages", [])
    if packages:
        cmd = f"{shlex.quote(str(manager))} install " + " ".join(shlex.quote(str(pkg)) for pkg in packages)
        out["setup"].append(run(cmd, REPO_DIR, 1200, tag=f"custom_{manager}"))

setup_failures = [result for result in out["setup"] if result.get("rc", 0) != 0]
if setup_failures:
    out["setup_failed"] = True
    out["setup_failure_count"] = len(setup_failures)
    save()
    print(f"[x] Environment setup failed ({len(setup_failures)} command(s)); steps will not run.")
    sys.exit(2)
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
        # If `python setup.py install` / `develop` failed due to deprecated easy_install dependency
        # resolution while packages are already installed, retry with `pip install --no-build-isolation --no-deps .`
        if r["rc"] != 0 and ("setup.py install" in cmd or "setup.py develop" in cmd):
            fallback_cmd = f"{shlex.quote(sys.executable)} -m pip install --no-build-isolation --no-deps ."
            print(f"[*] Retrying failed setup.py step with: {fallback_cmd}")
            r2 = run(fallback_cmd, cwd, timeout, tag=f"step_{s['id']}_pip_nodeps")
            if r2["rc"] == 0:
                r = r2
        done[cmd] = r

    r["step_id"] = s["id"]
    r["covers"] = s["covers"]
    r["claim_indices"] = s["covers"]
    r["signals"] = s["signals"]
    out["steps"].append(r)

    # Detect step-level environment/dependency failure so EnvirementSetup can trigger an LLM retry
    combined_err = f"{r.get('stdout', '')}\n{r.get('stderr', '')}"
    if r["rc"] != 0 and (
        s["id"].lower() in ("install", "setup", "build", "env")
        or any(tok in combined_err for tok in (
            "ModuleNotFoundError:",
            "ImportError:",
            "pkg_resources.DistributionNotFound",
            "pkg_resources.VersionConflict",
            "Could not find suitable distribution for Requirement.parse",
        ))
    ):
        out["setup_failed"] = True
        out["setup_failure_reason"] = f"step '{s['id']}' failed due to environment/dependency error"
        save()
        print(f"[x] Step '{s['id']}' failed due to environment error; halting for environment repair.")
        sys.exit(2)

    save()

print("\n[✓] Local execution complete. results.json generated.")
'''


DEFAULT_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"


def probe_docker_gpu(image: str = DEFAULT_IMAGE, timeout: int = 180) -> Dict[str, Any]:
    """Probes Docker's NVIDIA runtime and returns the raw outcome (for logging)."""
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
    image: str | None = None,
    *,
    plan_dir: Path | None = None,
    enable_env_retry: bool = True,
    max_env_attempts: int = MAX_SETUP_ATTEMPTS,
    llm_callable: Optional[Callable[[str], Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Runs the RunSpec inside a local Docker container with automatic 5x LLM environment setup
    retry and fallback to the latest version of all required modules.
    """
    gpu_wanted = spec.hardware.gpu in ("required", "helpful")

    def _resolve_img(env: Environment) -> str:
        return image or resolve_docker_image(env, gpu_available=gpu_wanted)

    if not enable_env_retry:
        selected_image = _resolve_img(spec.environment)
        with local_session(image=selected_image, workdir=str(workdir), gpu_policy=spec.hardware.gpu) as sess:
            return _run_locally(spec, workdir, repo_dir, data_dir, selected_image, sess)

    def _attempt_runner(current_spec: RunSpec, is_fallback_latest: bool) -> Optional[Dict[str, Any]]:
        selected_image = _resolve_img(current_spec.environment)
        with local_session(image=selected_image, workdir=str(workdir), gpu_policy=current_spec.hardware.gpu) as sess:
            return _run_locally(
                current_spec,
                workdir,
                repo_dir,
                data_dir,
                selected_image,
                sess,
                is_fallback_latest=is_fallback_latest,
            )

    updated_spec, final_out, _ = run_with_env_retry(
        spec,
        _attempt_runner,
        repo_dir=repo_dir,
        plan_dir=plan_dir,
        mode="local",
        max_attempts=max_env_attempts,
        llm_callable=llm_callable,
        resolve_image_fn=_resolve_img,
    )
    spec.environment = updated_spec.environment
    spec.warnings = list(updated_spec.warnings)
    return final_out


_IGNORE_VCS = shutil.ignore_patterns(".git", ".hg", "__pycache__")


def _stage_data(spec: RunSpec, repo_dir: Path, data_dir: Path, working_data_dir: Path) -> List[Dict[str, Any]]:
    """Copy everything under data_dir into working_data_dir, then alias each spec.data entry."""
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
            entry.update(source=str(repo_dir / d.ref), mode="in_code_repo")
        elif (working_data_dir / name).exists():
            entry.update(source=str(working_data_dir / name), mode="direct")
        else:
            first_seg = d.ref.split("/", 1)[0] if d.kind == "in_repo" else ""
            hit = find_data_folder(working_data_dir, name, d.id, first_seg)
            if hit is not None:
                alias = working_data_dir / name
                try:
                    alias.symlink_to(hit.name, target_is_directory=True)
                    entry.update(source=str(hit), mode="alias")
                except OSError:
                    shutil.copytree(hit, alias)
                    entry.update(source=str(hit), mode="copy")
        mounts.append(entry)
    return mounts


def missing_data_paths(spec: RunSpec, working_data_dir: Path) -> List[tuple]:
    """(step_id, relative path) for every {DATA}/... argument in a step command that does not
    exist in the staged data."""
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
    *,
    is_fallback_latest: bool = False,
) -> Optional[Dict[str, Any]]:
    workdir = Path(workdir).resolve()
    repo_dir = Path(repo_dir).resolve()
    data_dir = Path(data_dir).resolve()

    runner_dir = workdir / "local_run"
    runner_dir.mkdir(parents=True, exist_ok=True)
    out_dir = runner_dir / "out"
    out_dir.mkdir(exist_ok=True)
    shutil.rmtree(out_dir / "raw", ignore_errors=True)
    (out_dir / "results.json").unlink(missing_ok=True)

    # 1. Create a working copy of repo to avoid modifying source downloads
    working_repo_dir = runner_dir / "repo"
    if working_repo_dir.exists():
        shutil.rmtree(working_repo_dir)
    shutil.copytree(repo_dir, working_repo_dir)

    # In fallback mode (after 5 failed attempts), strip hardcoded version pins in the working
    # repo's setup.py / requirements.txt so `setup.py install` does not reject latest packages.
    if is_fallback_latest:
        unpinned_files = unpin_repo_manifests(working_repo_dir)
        if unpinned_files:
            print(f"[*] Unpinned version constraints in working repo manifests: {unpinned_files}")

    # 2. Stage data
    working_data_dir = runner_dir / "data"
    shutil.rmtree(working_data_dir, ignore_errors=True)
    working_data_dir.mkdir(exist_ok=True)
    data_mounts = _stage_data(spec, repo_dir, data_dir, working_data_dir)
    missing = missing_data_paths(spec, working_data_dir)
    for step_id, path in missing:
        print(f"[!] Step '{step_id}' uses {{DATA}}/{path}, which does not exist in the staged data.")
    sess.workspace(runner_dir, repo_dir, working_repo_dir, data_dir, data_mounts)

    # 3. Write runner files
    spec_json_str = spec.model_dump_json(indent=2)
    (runner_dir / "spec.json").write_text(spec_json_str, encoding="utf-8")
    (runner_dir / "runner.py").write_text(DOCKER_RUNNER_SCRIPT, encoding="utf-8")
    sess.save_spec(spec_json_str)
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

    container_output_lines: List[str] = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        sess.container_line(line)
        container_output_lines.append(line)
    process.wait()
    sess.container_exit(process.returncode)

    results_file = out_dir / "results.json"
    sess.ingest_results(results_file if results_file.exists() else None, out_dir / "raw")
    if results_file.exists():
        return json.loads(results_file.read_text(encoding="utf-8"))

    if process.returncode != 0:
        return {
            "setup": [
                {
                    "cmd": shlex.join(docker_cmd),
                    "rc": process.returncode,
                    "sec": 0,
                    "stdout": "",
                    "stderr": "".join(container_output_lines)[-4000:],
                }
            ],
            "steps": [],
            "patches": [],
            "setup_failed": True,
            "setup_failure_count": 1,
        }
    return None
