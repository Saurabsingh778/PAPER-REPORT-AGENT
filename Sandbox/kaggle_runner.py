"""Run a validated RunSpec on Kaggle and fetch results."""
import json
import subprocess
import time
from pathlib import Path
from typing import Sequence

from Schema.RunSpec import RunSpec
from .config import KAGGLE_USER

# Using a standard multiline string with a simple placeholder __SPEC_JSON__
# to avoid str.format() escaping collisions with Python braces and f-strings.
KERNEL_TEMPLATE = r'''
import subprocess, json, time, os, re, sys, shutil

SPEC = json.loads(__SPEC_JSON__)
W = "/kaggle/working"
REPO_DIR = os.path.join(W, "repo")
DATA_DIR = os.path.join(W, "data")
os.makedirs(DATA_DIR, exist_ok=True)

out = {"setup": [], "steps": [], "patches": []}

def run(cmd, cwd, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": cmd, "rc": p.returncode, "sec": round(time.time()-t),
                "stdout": p.stdout[-6000:], "stderr": p.stderr[-3000:]}
    except subprocess.TimeoutExpired as e:
        so = e.stdout if isinstance(e.stdout, str) else ""
        return {"cmd": cmd, "rc": -9, "sec": timeout, "stdout": so[-3000:], "stderr": "TIMEOUT"}

def save():
    with open(os.path.join(W, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

# Check hardware environment
print(run("nvidia-smi -L", W, 60)["stdout"])

# 1. Clone repository
repo_url = SPEC["repo"]["url"]
commit = SPEC["repo"]["commit"]
out["clone"] = run(f"git clone {repo_url} {REPO_DIR}", W, 600)
if commit and commit != "0000000000000000000000000000000000000000":
    run(f"git checkout {commit}", REPO_DIR, 60)

# 2. Setup Data Sources
for d in SPEC.get("data", []):
    kind, ref, dest = d["kind"], d["ref"], d.get("dest")
    target_path = os.path.join(DATA_DIR, dest) if dest else DATA_DIR
    if kind == "github_repo":
        run(f"git clone --depth 1 {ref} {target_path}", W, 600)
    elif kind == "http_download":
        os.makedirs(target_path, exist_ok=True)
        run(f"wget -q -P {target_path} {ref}", W, 600)

# 3. Apply PathRewrites
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

# 4. Install Environment
for pkg in SPEC.get("environment", {}).get("pip", []):
    out["setup"].append(run(f"pip install -q {pkg}", REPO_DIR, 600))
save()

# 5. Execute Steps in Order
done = {}
for s in SPEC["steps"]:
    cmd = s["command"].replace("{REPO}", REPO_DIR).replace("{DATA}", DATA_DIR)
    cwd = s["cwd"].replace("{REPO}", REPO_DIR).replace("{DATA}", DATA_DIR)
    timeout = s["timeout_min"] * 60

    if cmd in done:
        r = dict(done[cmd])
    else:
        r = run(cmd, cwd, timeout)
        done[cmd] = r

    r["step_id"] = s["id"]
    r["covers"] = s["covers"]
    r["signals"] = s["signals"]
    out["steps"].append(r)
    save()
'''


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def write_kernel(spec: RunSpec, workdir: Path, slug: str, datasets: Sequence[str] = ()):
    d = Path(workdir) / "kernel"
    d.mkdir(parents=True, exist_ok=True)

    gpu_enabled = "true" if spec.hardware.gpu in ("required", "helpful") else "false"

    kaggle_datasets = list(datasets) + [
        d_src.ref for d_src in spec.data if d_src.kind == "kaggle_dataset"
    ]

    # Safely inject JSON using repr() and string replacement
    kernel_code = KERNEL_TEMPLATE.replace("__SPEC_JSON__", repr(spec.model_dump_json()))
    (d / "script.py").write_text(kernel_code, encoding="utf-8")

    (d / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{KAGGLE_USER}/{slug}",
                "title": slug,
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": gpu_enabled,
                "enable_internet": "true",
                "dataset_sources": list(set(kaggle_datasets)),
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return d


def run_on_kaggle(
    spec: RunSpec,
    workdir: Path,
    datasets: Sequence[str] = (),
    slug: str = "repro-agent-run",
    poll: int = 30,
    max_wait_min: int = 240,
):
    d = write_kernel(spec, Path(workdir), slug, datasets)
    p = kaggle("kernels", "push", "-p", str(d))
    print(p.stdout, p.stderr)

    ref = f"{KAGGLE_USER}/{slug}"
    t0 = time.time()
    while time.time() - t0 < max_wait_min * 60:
        s = kaggle("kernels", "status", ref).stdout
        print(s.strip())
        if any(state in s for state in ("COMPLETE", "ERROR", "CANCEL")):
            break
        time.sleep(poll)

    out = Path(workdir) / "kaggle_out"
    kaggle("kernels", "output", ref, "-p", str(out))
    f = out / "results.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None