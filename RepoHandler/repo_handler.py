"""RepoHandler: Analyzes the cloned repo + Phase 1 extraction to produce a validated RunSpec."""

import ast
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types
from pydantic import ValidationError

from Logger import get_phase
from Model.gemini import client, MODEL
from Schema.RunSpec import (
    RunSpec,
    Hardware,
    Evidence,
    check_against_repo,
)

CUDA_RE = re.compile(
    r"(\.cuda\(|to\(['\"]cuda|torch\.device\(['\"]cuda|torch\.cuda|jax\.devices\(['\"]gpu|tf\.device\(['\"]/GPU)",
    re.IGNORECASE,
)


class RepoScanner:
    """Performs static analysis on the local repository folder."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = Path(repo_dir).resolve()

    def get_commit_sha(self) -> str:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception:
            return "0000000000000000000000000000000000000000"

    def scan_files(self) -> List[str]:
        return [
            str(p.relative_to(self.repo_dir))
            for p in self.repo_dir.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.parts)
        ]

    def read_readme(self) -> str:
        for f in self.repo_dir.iterdir():
            if f.is_file() and f.name.lower().startswith("readme"):
                return f.read_text(encoding="utf-8", errors="replace")[:15000]
        return ""

    def scan_python_scripts(self, max_scripts: int = 50) -> Dict[str, Dict[str, Any]]:
        scripts = {}
        py_files = [
            p for p in self.repo_dir.rglob("*.py")
            if not any(part.startswith(".") for part in p.parts)
        ][:max_scripts]

        for p in py_files:
            rel = str(p.relative_to(self.repo_dir))
            content = p.read_text(encoding="utf-8", errors="replace")
            
            # Check for CUDA usage
            cuda_matches = CUDA_RE.findall(content)
            uses_gpu = len(cuda_matches) > 0
            
            # Extract docstrings / top lines
            lines = content.splitlines()
            head = "\n".join(lines[:40])
            
            # Find prints / metric loggers
            metric_prints = [
                line.strip() for line in lines
                if any(k in line.lower() for k in ("print(", "logger.info", "accuracy", "auc", "f1", "loss", "metric"))
            ][:15]

            scripts[rel] = {
                "head": head,
                "uses_gpu_indicators": uses_gpu,
                "gpu_matches": cuda_matches[:5],
                "sample_prints": metric_prints,
            }
        return scripts

    @staticmethod
    def scan_data_dir(data_dir: Path, max_entries: int = 200) -> List[str]:
        """List files in the downloaded data directory (relative paths, capped)."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return []
        entries = []
        for p in sorted(data_dir.rglob("*")):
            if any(part.startswith(".") for part in p.relative_to(data_dir).parts):
                continue
            if p.is_file():
                entries.append(str(p.relative_to(data_dir)))
                if len(entries) >= max_entries:
                    break
        return entries


def build_run_spec_prompt(
    extraction: Dict[str, Any],
    repo_url: str,
    commit_sha: str,
    files: List[str],
    readme: str,
    scripts_info: Dict[str, Any],
    data_files: Optional[List[str]] = None,
) -> str:
    claims = extraction.get("main_results", [])
    data_links = extraction.get("data_links", [])

    return f"""You are a scientific reproducibility planner. Your task is to plan reproduction runs on a Kaggle notebook.
You must construct a complete, strictly valid RunSpec JSON following the provided schema.

TARGET REPOSITORY:
URL: {repo_url}
Commit: {commit_sha}

PAPER EXTRACTED CLAIMS TO REPRODUCE:
{json.dumps(claims, indent=2)}

DATA LINKS MENTIONED IN PAPER:
{json.dumps(data_links, indent=2)}

REPOSITORY FILE LIST:
{json.dumps(files[:250], indent=2)}

DOWNLOADED DATA FILES (paths relative to {{DATA}}; empty means no data was downloaded):
{json.dumps(data_files or [], indent=2)}

README EXCERPT:
{readme}

KEY SCRIPT HEADS & STATIC CODE ANALYSIS:
{json.dumps(scripts_info, indent=2)}

CRITICAL PLANNING GUIDELINES:
1. ONLY reference files that exist in the REPOSITORY FILE LIST. Never invent commands or scripts.
2. HARDWARE / GPU DECISION:
   - Determine whether hardware.gpu should be "required", "helpful", or "none".
   - If training neural networks or scripts explicitly require CUDA (.cuda(), torch.device('cuda')), set hardware.gpu to "required".
   - If evaluation can run on CPU or model is lightweight, set "helpful" or "none".
   - Consistency rule: If hardware.gpu is "none", all steps must have gpu=False. If hardware.gpu is "required", at least one step must have gpu=True.
3. COMMAND FORMAT & TOKENS:
   - Use tokens {{REPO}} and {{DATA}}. The command must NOT contain absolute paths (/home/..., /kaggle/...).
   - cwd must be "{{REPO}}" or "{{REPO}}/<subdir>".
4. SIGNALS & METRIC EXTRACTION:
   - For every claim covered by a step, define a regex line_pattern in signals.
   - Example line_pattern: "(?i)accuracy[:\\s=]+([0-9]+\\.?[0-9]*)"
5. CLAIM MAPPING:
   - Every claim (1-indexed based on the order in CLAIMS) must be listed in claims.
   - If a claim cannot be verified because data/compute is missing, mark its status as "unreproducible" or "unclear" with a detailed reason.
6. PATH REWRITES:
   - If scripts contain hardcoded local paths (e.g., ./data, /path/to/data), provide a PathRewrite patch replacing them with {{DATA}}/<folder>.
"""


def generate_run_spec(
    extraction_json_path: str | Path,
    repo_dir: str | Path,
    repo_url: str,
    data_dir: Optional[str | Path] = None,
    log_dir: Optional[Path] = None,
) -> RunSpec:
    """Analyzes repo, calls Gemini to build RunSpec, validates it, and saves logs."""
    extraction_json_path = Path(extraction_json_path)
    repo_dir = Path(repo_dir)
    data_dir = Path(data_dir) if data_dir is not None else None

    extraction = json.loads(extraction_json_path.read_text(encoding="utf-8"))
    scanner = RepoScanner(repo_dir)

    commit_sha = scanner.get_commit_sha()
    files = scanner.scan_files()
    readme = scanner.read_readme()
    scripts_info = scanner.scan_python_scripts()
    data_files = RepoScanner.scan_data_dir(data_dir) if data_dir else []

    plog = get_phase()
    plog.info("repository scanned", commit=commit_sha, files=len(files), scripts=len(scripts_info),
              data_files=len(data_files), readme_chars=len(readme),
              gpu_script_hits=[p for p, i in scripts_info.items() if i["uses_gpu_indicators"]])
    plog.save_artifact("repo_scan.json", {"files": files, "scripts": scripts_info, "data_files": data_files})

    prompt = build_run_spec_prompt(
        extraction=extraction,
        repo_url=repo_url,
        commit_sha=commit_sha,
        files=files,
        readme=readme,
        scripts_info=scripts_info,
        data_files=data_files,
    )

    plog.save_artifact("planner_prompt.txt", prompt)
    print("[*] Requesting RunSpec generation from Gemini...")
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RunSpec,
            temperature=0.1,
        ),
    )

    # Keep the raw reply before validating: a schema failure below is then still debuggable.
    plog.save_artifact("planner_response_raw.json", response.text or "")
    plog.info("model responded", model=MODEL, usage=getattr(response, "usage_metadata", None))

    spec = RunSpec.model_validate_json(response.text)
    print(f"[✓] RunSpec generated. Target hardware GPU: {spec.hardware.gpu}")
    plog.info("RunSpec validated", gpu=spec.hardware.gpu, steps=len(spec.steps),
              claims=len(spec.claims), confidence=spec.confidence)

    # Local pre-flight check
    problems = check_against_repo(spec, repo_dir)
    if problems:
        print("[!] Pre-flight warnings found:")
        for prob in problems:
            print(f"  - {prob}")
            spec.warnings.append(prob)
            plog.warning(f"pre-flight: {prob}")
    else:
        print("[✓] Pre-flight check passed with 0 errors.")
        plog.info("pre-flight check passed")

    # Save detailed logs
    if log_dir is None:
        log_dir = extraction_json_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_data = {
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "scanned_files_count": len(files),
        "data_files_count": len(data_files),
        "hardware_decision": {
            "gpu": spec.hardware.gpu,
            "reason": spec.hardware.reason,
            "evidence": [e.model_dump() for e in spec.hardware.evidence],
        },
        "preflight_problems": problems,
        "raw_model_response": response.text,
        "run_spec": spec.model_dump(),
    }
    
    log_path = log_dir / f"repo_handler_{Path(extraction_json_path).stem}.json"
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    print(f"[✓] RepoHandler log saved to: {log_path}")
    plog.save_artifact("repo_handler_log.json", log_path)

    return spec