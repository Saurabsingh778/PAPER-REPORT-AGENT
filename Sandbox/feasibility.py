"""Local feasibility checker: can THIS machine run the planned steps?

probe_local_resources()  -> what the machine has (GPU/VRAM, RAM, disk, CPUs, Docker)
resources_for_prompt()   -> text block telling the planner the limits (so it plans for them)
check_spec()             -> per-step / per-claim verdict against those resources
apply_gate()             -> spec without blocked steps (their claims marked unreproducible
                            with the concrete resource reason); everything feasible still runs.

Pure rules on hard numbers (VRAM, disk, time): no LLM, so "can run" is never guessed.
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

MAX_LOCAL_RUNTIME_MIN = 480  # a single step estimated longer than this is skipped


def _run(cmd: List[str], timeout: int = 20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def probe_local_resources(path: str = ".") -> Dict[str, Any]:
    gpu = {"available": False, "name": None, "vram_gb": 0.0, "free_vram_gb": 0.0}
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"])
    if out:
        name, total, free = [x.strip() for x in out.splitlines()[0].split(",")]
        gpu = {"available": True, "name": name, "vram_gb": round(float(total) / 1024, 1),
               "free_vram_gb": round(float(free) / 1024, 1)}
    ram_gb = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except Exception:
        pass
    disk_free_gb = round(shutil.disk_usage(path).free / 1024 ** 3, 1)
    return {"gpu": gpu, "ram_gb": ram_gb, "disk_free_gb": disk_free_gb,
            "cpus": os.cpu_count(), "docker": bool(_run(["docker", "--version"]))}


def resources_for_prompt(res: Dict[str, Any]) -> str:
    g = res["gpu"]
    gpu = f"{g['name']} ({g['vram_gb']} GB VRAM)" if g["available"] else "NONE"
    return (f"GPU: {gpu}; RAM: {res['ram_gb']} GB; free disk: {res['disk_free_gb']} GB; "
            f"CPU cores: {res['cpus']}; max single-step runtime accepted: {MAX_LOCAL_RUNTIME_MIN} min")


def check_spec(spec, res: Dict[str, Any]) -> Dict[str, Any]:
    """Return {runnable, global_blockers, steps:{id:[problems]}, claims:{id:'feasible'|reason}}."""
    g = res["gpu"]
    global_blockers = []
    if not res["docker"]:
        global_blockers.append("Docker is not installed or not on PATH")

    need_mb = sum((d.size_mb or 0) for d in spec.data)
    if need_mb and need_mb / 1024 > res["disk_free_gb"] * 0.8:
        global_blockers.append(f"data needs ~{need_mb / 1024:.1f} GB, only {res['disk_free_gb']} GB free")

    problems: Dict[str, List[str]] = {}
    for s in spec.steps:
        p = []
        if s.gpu and not g["available"]:
            p.append("step needs a GPU but no NVIDIA GPU was detected")
        if s.gpu and spec.hardware.min_vram_gb and g["available"] and spec.hardware.min_vram_gb > g["vram_gb"]:
            p.append(f"needs >= {spec.hardware.min_vram_gb} GB VRAM, GPU has {g['vram_gb']} GB")
        est = spec.hardware.est_runtime_min
        if est and len(spec.steps) == 1 and est > MAX_LOCAL_RUNTIME_MIN:
            p.append(f"estimated runtime {est} min exceeds the {MAX_LOCAL_RUNTIME_MIN} min limit")
        if s.timeout_min > MAX_LOCAL_RUNTIME_MIN:
            p.append(f"timeout {s.timeout_min} min exceeds the {MAX_LOCAL_RUNTIME_MIN} min limit")
        problems[s.id] = p
    # a step that needs a blocked step is blocked too
    changed = True
    while changed:
        changed = False
        for s in spec.steps:
            if not problems[s.id] and any(problems.get(n) for n in s.needs):
                problems[s.id] = [f"depends on blocked step(s): {[n for n in s.needs if problems.get(n)]}"]
                changed = True

    claims = {}
    for c in spec.claims:
        if c.status != "mapped":
            claims[c.claim_id] = f"not planned: {c.reason}"
            continue
        bad = [p for sid in c.step_ids for p in problems.get(sid, [])]
        claims[c.claim_id] = "feasible" if not bad else "blocked: " + "; ".join(sorted(set(bad)))
    runnable = not global_blockers and any(not p for p in problems.values())
    return {"runnable": runnable, "resources": res, "global_blockers": global_blockers,
            "steps": problems, "claims": claims}


def apply_gate(spec, report: Dict[str, Any]):
    """Drop blocked steps; their claims become 'unreproducible' with the resource reason."""
    blocked = {sid for sid, p in report["steps"].items() if p}
    if not blocked:
        return spec
    new = spec.model_copy(deep=True)
    new.steps = [s for s in new.steps if s.id not in blocked]
    for c in new.claims:
        if c.status == "mapped" and any(sid in blocked for sid in c.step_ids):
            c.status, c.step_ids = "unreproducible", []
            c.reason = report["claims"][c.claim_id]
    new.warnings.append(f"skipped on this machine: {sorted(blocked)}")
    return new