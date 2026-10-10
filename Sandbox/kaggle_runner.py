"""Run a plan on a free Kaggle GPU kernel and fetch results.json."""
import json
import subprocess
import time
from pathlib import Path

from .config import KAGGLE_USER

KERNEL_TEMPLATE = r'''
import subprocess, json, time, os, sys
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

print(run("nvidia-smi -L", W, 60)["stdout"])
out["clone"] = run(f"git clone --depth 1 {{REPO}} repo", W, 600)
RD = W + "/repo"
for c in PLAN["setup_commands"]:
    out["setup"].append(run(c, RD, 1800))
for s in PLAN["steps"]:
    r = run(s["command"], RD, s["timeout_min"] * 60)
    r["claim_index"] = s["claim_index"]
    out["steps"].append(r)
    json.dump(out, open(W + "/results.json", "w"), indent=1)  # save progress
json.dump(out, open(W + "/results.json", "w"), indent=1)
'''


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def write_kernel(plan, repo_url, workdir, slug):
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
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
            }
        )
    )
    return d


def run_on_kaggle(plan, repo_url, workdir, slug="repro-agent-run", poll=30, max_wait_min=240):
    d = write_kernel(plan, repo_url, workdir, slug)
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