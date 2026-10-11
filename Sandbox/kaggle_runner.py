"""Run a plan on a free Kaggle GPU kernel and fetch results.json."""
import json
import subprocess
import time
from pathlib import Path

from .config import KAGGLE_USER

# The kernel does its own environment setup. It never runs `pip install -r requirements.txt`:
# old exact pins (torch==2.0.1 ...) don't exist for Kaggle's Python and one failure aborts
# the whole install, which is what broke the first run.
KERNEL_TEMPLATE = r'''
import subprocess, json, time, os, re, sys
PLAN = json.loads({plan_json!r})
REPO = {repo_url!r}
W = "/kaggle/working"
out = {{"setup": [], "steps": []}}

def run(cmd, cwd, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {{"cmd": cmd, "rc": p.returncode, "sec": round(time.time()-t),
                "stdout": p.stdout[-6000:], "stderr": p.stderr[-3000:]}}
    except subprocess.TimeoutExpired as e:
        so = e.stdout if isinstance(e.stdout, str) else ""
        return {{"cmd": cmd, "rc": -9, "sec": timeout, "stdout": so[-3000:], "stderr": "TIMEOUT"}}

def save():
    json.dump(out, open(W + "/results.json", "w"), indent=1)

print(run("nvidia-smi -L", W, 60)["stdout"])
out["clone"] = run(f"git clone --depth 1 {{REPO}} repo", W, 600)
RD = W + "/repo"

# ---- environment: install each package separately, unpinned, skipping what Kaggle ships ----
SKIP = {{"torch", "torchvision", "torchaudio", "jax", "jaxlib", "pickle5", "torch-scatter",
        "torch-sparse", "torch-cluster", "numpy", "scipy", "pandas", "scikit-learn",
        "matplotlib", "networkx", "tqdm", "seaborn"}}
req = os.path.join(RD, "requirements.txt")
pkgs = []
if os.path.exists(req):
    for line in open(req):
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[=<>!~\[ ]", line)[0].strip()
        if name and name.lower() not in SKIP:
            pkgs.append(name)
for name in pkgs:                      # e.g. torch-geometric, jax-md
    out["setup"].append(run(f"pip install -q {{name}}", RD, 900))
# torch_geometric's loader import moved between releases; fall back to the paper's pinned version
chk = run("python -c \"import torch_geometric; from torch_geometric.loader import DataLoader\"", RD, 120)
if chk["rc"] != 0:
    out["setup"].append(run("pip install -q torch_geometric==2.3.1", RD, 900))
save()

# ---- steps, in order; same command is never run twice ----
done = {{}}
for s in PLAN["steps"]:
    if s["command"] in done:
        r = dict(done[s["command"]])
    else:
        r = run(s["command"], RD, s["timeout_min"] * 60)
        done[s["command"]] = r
    r["claim_indices"] = s["claim_indices"]
    out["steps"].append(r)
    save()
'''


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def write_kernel(plan, repo_url, workdir, slug, datasets):
    d = Path(workdir) / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    (d / "script.py").write_text(
        KERNEL_TEMPLATE.format(plan_json=plan.model_dump_json(), repo_url=repo_url)
    )
    (d / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{KAGGLE_USER}/{slug}",
                "title": slug,
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                # scripts that hardcode /kaggle/input/datasets/<owner>/<slug>/... need these mounted
                "dataset_sources": list(datasets),
                "competition_sources": [],
                "kernel_sources": [],
            }
        )
    )
    return d


def run_on_kaggle(plan, repo_url, workdir, datasets=(), slug="repro-agent-run",
                  poll=30, max_wait_min=240):
    d = write_kernel(plan, repo_url, workdir, slug, datasets)
    p = kaggle("kernels", "push", "-p", str(d))
    print(p.stdout, p.stderr)

    ref = f"{KAGGLE_USER}/{slug}"
    t0 = time.time()
    while time.time() - t0 < max_wait_min * 60:
        s = kaggle("kernels", "status", ref).stdout
        print(s.strip())
        if "COMPLETE" in s or "ERROR" in s or "CANCEL" in s:
            break
        time.sleep(poll)

    out = Path(workdir) / "kaggle_out"
    kaggle("kernels", "output", ref, "-p", str(out))
    f = out / "results.json"
    return json.loads(f.read_text()) if f.exists() else None