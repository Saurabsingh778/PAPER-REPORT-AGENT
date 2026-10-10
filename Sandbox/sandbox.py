"""
Phase 2: take Phase 1 extraction.json -> inspect GitHub repo -> plan runs ->
execute on a free Kaggle GPU kernel -> compare against paper numbers.
 
Free-tier pieces used:
  - GitHub REST API (free token, 5000 req/h)         env: GITHUB_TOKEN
  - Gemini API free tier (+ Google Search grounding)  env: API_KEY, MODEL
  - Kaggle API (T4 x2, ~30 GPU h/week, internet on)   ~/.kaggle/kaggle.json
 
Usage:
  python -m Model.phase2_sandbox <phase1_output_dir>
"""

import base64
import json
import os
import re
import subprocess
import time
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

GH_TOKEN = os.getenv("GITHUB_TOKEN")
KAGGLE_USER = os.getenv("KAGGLE_USERNAME", "saurabingh")
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")
client = genai.Client(api_key=os.getenv("API_KEY"))


def gh(path):
    h = {"Accept" : "application/vnd.github+json"}
    if GH_TOKEN:
        h["Authorization"] = f"Bearer {GH_TOKEN}"
    r = requests.get(f"https://api.github.com{path}", headers=h, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_repo(url):
    m = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", url)

    if not m:
        return None

    return m.group(1), m.group(2).removesuffix(".git")

def inspect_repo(url):
    """Return README text + file tree + dependency files (all via API, no clone)."""
    parsed = parse_repo(url)

    if not parse_repo:
        return None

    owner, repo = parsed

    info = gh(f"/repos/{owner}/{repo}")
    tree = gh(f"/repos/{owner}/{repo}/git/trees/{info['default_branch']}?recursive=1")
    paths = [t["path"] for t in tree["tree"] if t["type"] == "blob"]

    def fetch(p):
        j = gh(f"/repos/{owner}/{repo}/contents/{p}")
        return base64.b64decode(j["content"]).decode("utf-8", "replace")
    
    readme = next((fetch(p) for p in paths if p.lower().startswith("readme")), "")

    deps = {p: fetch(p) for p in paths
            if p.split("/")[-1] in ("requirements.txt", "environment.yml", "pyproject.toml")}
    
    return {"owner": owner, "repo": repo, "branch": info["default_branch"],
            "size_kb": info["size"], "files": paths[:300],
            "readme": readme[:12000], "deps": deps}

# ---------------------------------------------------------------- Planner ---

class RunStep(BaseModel):
    claim_index: int          # 1-based index into main_results
    command: str              # shell command run from repo root
    expected_output_hint: str # what to grep for in stdout/files
    timeout_min: int = 30

class Plan(BaseModel):
    setup_commands: list[str]
    steps: list[RunStep]
    unreproducible: list[str]  # claims that need data/compute we can't get

def make_plan(results, repo):
    prompt = f"""You plan reproduction runs on one Kaggle notebook (2x Tesla T4, 12h limit,
internet on, no sudo). Only use scripts that exist in FILE LIST. Do not invent files.
If a claim cannot be reproduced from the repo, list it in unreproducible.
 
CLAIMS:
{json.dumps(results, indent=1)}
 
FILE LIST:
{json.dumps(repo['files'])}
 
README:
{repo['readme']}
 
DEPS:
{json.dumps(repo['deps'])[:4000]}
"""
    # Google Search grounding can't be combined with JSON schema output on all models,
    # so planning uses schema; use grounding separately for lookups (see ground()).
    r = client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Plan))
    return Plan.model_validate_json(r.text)
 
 
def ground(question):
    """Free Google Search grounding, for e.g. 'how do I install X on Kaggle'."""
    r = client.models.generate_content(
        model=MODEL, contents=question,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]))
    return r.text


# ---------------------------------------------------------------- Kaggle ----
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
        return {{"cmd": cmd, "rc": -9, "sec": timeout, "stdout": (e.stdout or "")[-3000:] if isinstance(e.stdout, str) else "", "stderr": "TIMEOUT"}}
 
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


