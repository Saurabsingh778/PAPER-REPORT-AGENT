"""RepoHandler: Analyzes the cloned repo + Phase 1 extraction to produce a validated RunSpec."""

import ast
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from google.genai import types
except ImportError:
    types = None
from pydantic import ValidationError

from EnvirementSetup import MAX_SETUP_ATTEMPTS, build_latest_fallback_environment
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

ENVIRONMENT_FILE_RE = re.compile(
    r"(?:^|/)(?:requirements(?:[-_.][\w-]+)?\.(?:txt|in)|environment(?:[-_.][\w-]+)?\.ya?ml|"
    r"pyproject\.toml|setup\.py|setup\.cfg|Pipfile|poetry\.lock|conda\.ya?ml|Dockerfile)$",
    re.IGNORECASE,
)
PYTHON_VERSION_RE = re.compile(
    r"(?i)\bpython\s*(?:version\s*)?(?:==|>=|<=|~=|>|<)?\s*(2\.7|3\.\d{1,2})(?:\.\d+)?"
)
FRAMEWORK_VERSION_RE = re.compile(
    r"(?i)\b(pytorch|torch|tensorflow|jax)\s*(?:==|>=|<=|~=|>|<|version\s*)?\s*(\d+\.\d+(?:\.\d+)?)"
)
IMPORT_PACKAGE_NAMES = {
    "PIL": "Pillow", "cv2": "opencv-python", "sklearn": "scikit-learn",
    "yaml": "PyYAML", "bs4": "beautifulsoup4", "torch": "torch",
    "tensorflow": "tensorflow", "jax": "jax", "networkx": "networkx",
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
    "matplotlib": "matplotlib", "transformers": "transformers",
}


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
            ][:40]

            scripts[rel] = {
                "head": head,
                "uses_gpu_indicators": uses_gpu,
                "gpu_matches": cuda_matches[:5],
                "sample_prints": metric_prints,
            }
        return scripts

    def scan_environment(self, max_manifest_chars: int = 12000, max_scripts: int = 200) -> Dict[str, Any]:
        """Collect dependency evidence for the planner without inventing versions.

        Dependency manifests and README text remain authoritative. Imports merely establish
        a fallback package list when a repository does not declare its environment.
        """
        manifests: Dict[str, str] = {}
        imports: set[str] = set()
        for path in self.repo_dir.rglob("*"):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(self.repo_dir).parts):
                continue
            rel = str(path.relative_to(self.repo_dir))
            if ENVIRONMENT_FILE_RE.search(rel):
                manifests[rel] = path.read_text(encoding="utf-8", errors="replace")[:max_manifest_chars]

        for path in list(self.repo_dir.rglob("*.py"))[:max_scripts]:
            if any(part.startswith(".") for part in path.relative_to(self.repo_dir).parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    package = IMPORT_PACKAGE_NAMES.get(module.split(".", 1)[0])
                    if package:
                        imports.add(package)

        evidence = "\n".join(manifests.values())
        return {
            "manifests": manifests,
            "detected_import_packages": sorted(imports),
            "python_version_hints": sorted(set(PYTHON_VERSION_RE.findall(evidence))),
            "framework_version_hints": [
                {"name": "pytorch" if name.lower() in ("pytorch", "torch") else name.lower(), "version": version}
                for name, version in FRAMEWORK_VERSION_RE.findall(evidence)
            ],
        }

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
    environment_info: Dict[str, Any],
    data_files: Optional[List[str]] = None,
    resources: str = "",
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

LOCAL MACHINE THE RUN WILL USE (plan within these limits):
{resources or "unknown"}

README EXCERPT:
{readme}

KEY SCRIPT HEADS & STATIC CODE ANALYSIS:
{json.dumps(scripts_info, indent=2)}

ENVIRONMENT DISCOVERY (manifest text is authoritative; imports are fallback-only):
{json.dumps(environment_info, indent=2)}

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
   - HARD RULE: every claim index listed in a step's `covers` MUST have a matching entry in that
     same step's `signals` (claim_id + metric + line_pattern). If a step cannot print a verifiable
     number for a claim, do NOT put that claim in `covers`; mark it "unclear" or "unreproducible"
     in claims with a reason instead.
   - Example line_pattern: "(?i)accuracy[:\\s=]+([0-9]+\\.?[0-9]*)"
5. CLAIM MAPPING:
   - Every claim (1-indexed based on the order in CLAIMS) must be listed in claims.
   - If a claim cannot be verified because data/compute is missing, mark its status as "unreproducible" or "unclear" with a detailed reason.
   - TRY TO REPRODUCE EVERY CLAIM. "Not part of the default pipeline", "extended ablation" or "value is
     computed internally but not printed" are NOT valid reasons to give up. First look for: another
     script in the file list, an argparse flag/config for the variant (feature set, normalisation,
     ablation, test), or a function that computes the value. Then add a separate step for it (command
     with the right flags, or `python -c "..."` that imports the repo's function and PRINTS the value)
     with its own signal, confidence "low" if unsure.
   - Mark a claim unreproducible only if (a) no code in the repo can produce it, (b) its data is
     unavailable, or (c) it clearly exceeds the LOCAL MACHINE limits above (VRAM, runtime). State which.
6. PATH REWRITES:
   - If scripts contain hardcoded local paths (e.g., ./data, /path/to/data), provide a PathRewrite patch replacing them with {{DATA}}/<folder>.
7. DATA PATHS:
   - {{DATA}} is the root of the DOWNLOADED DATA FILES listing above, with that exact layout. Write data
     paths as {{DATA}}/<path exactly as listed there>, e.g. {{DATA}}/<repo_folder>/<sub>/<folder>.
   - Pick the script whose INPUT FORMAT matches the files actually downloaded. A script that loads a
     file which is neither in the repo nor downloaded (e.g. a dataset.csv made by a helper script that
     is not in the FILE LIST) cannot run; do not plan it.
8. ARGUMENT VALUES:
   - Any value passed to a flag (--experiments, --config, --model, ...) must be a name that literally
     appears in that script's source. Copy it from the code; never invent or abbreviate one.
9. ENVIRONMENT (REQUIRED):
   - Fill environment.language, environment.language_version, environment.frameworks,
     environment.packages, environment.requirements_files, and environment.requirements_policy.
   - Honour versions stated in README, setup.py, pyproject.toml, requirements*, and environment*.yml.
     A README's bounded compatibility statement overrides a loose lower bound in a manifest. Choose
     one concrete compatible Python version, never the host's default Python version.
   - requirements_files may contain only installable requirements*.txt/.in or constraints*.txt/.in
     files. Never put setup.py, pyproject.toml, or environment.yml there: they are evidence, not
     inputs to `pip install -r`. Use policy "resolved" when packages.pip contains a concrete
     compatible resolution; use "install_as_is" only when the requirements file is safe to install
     unchanged; use "ignore" only when it is not installed.
   - If no repository documentation or dependency manifest specifies a runtime, infer it from the
     detected imports/code, choose a supported Python and packages, set policy "resolved", and use
     low-confidence llm_judgment evidence for that inference.
   - Never combine an old framework with a modern default Python. For example, if the repository
     supports Python 2.7 or 3.6 and PyTorch 0.4 or 0.5, choose Python 3.6 and one PyTorch 0.x
     release, rather than trying to install it on Python 3.10+.
   - Resolve contradictory old pins before execution. In particular, TensorFlow 1.x requires
     Python 3.5-3.7, numpy>=1.16,<1.19, and protobuf==3.20.3; do not retain a conflicting
     numpy==1.15.* pin merely because it appears in an old requirements file.
   - docker_image is optional. Leave it null unless the repository explicitly needs a custom base;
     the executor will choose the version-matched Docker image from this environment contract.
"""


MAX_SPEC_ATTEMPTS = MAX_SETUP_ATTEMPTS


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors())


def _request_spec(prompt: str, plog, max_attempts: int = MAX_SPEC_ATTEMPTS, repo_dir: Optional[Path] = None):
    """Ask Gemini for a RunSpec. If the reply fails schema validation, send the validator's
    errors back and ask for a corrected spec (up to 5 attempts). If all 5 attempts fail due to
    environment validation errors, fall back to the latest versions of all required modules."""
    contents = prompt
    last_error: Optional[ValidationError] = None
    last_raw: str = ""
    last_response = None
    for attempt in range(1, max_attempts + 1):
        print(f"[*] Requesting RunSpec generation from Gemini (attempt {attempt}/{max_attempts})...")
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RunSpec,
                temperature=0.1,
            ),
        )
        raw = response.text or ""
        last_raw = raw
        last_response = response
        # Stored before validating, so a rejected reply is always inspectable.
        plog.save_artifact(f"planner_response_attempt{attempt}.json", raw)
        plog.info("model responded", attempt=attempt, model=MODEL,
                  usage=getattr(response, "usage_metadata", None))
        try:
            return RunSpec.model_validate_json(raw), response
        except ValidationError as exc:
            last_error = exc
            problems = _format_validation_error(exc)
            plog.warning(f"RunSpec rejected by validator (attempt {attempt}/{max_attempts}): {problems}")
            print(f"[!] RunSpec rejected by validator (attempt {attempt}/{max_attempts}): {problems}")
            if attempt == max_attempts:
                break
            contents = (
                prompt
                + "\n\nYOUR PREVIOUS RESPONSE WAS REJECTED BY THE SCHEMA VALIDATOR.\n"
                + f"VALIDATION ERRORS:\n{problems}\n\n"
                + f"PREVIOUS RESPONSE:\n{raw}\n\n"
                + "Return a corrected, complete RunSpec JSON that fixes every listed error. "
                + "Remember: every claim in a step's `covers` needs a matching signal in that step; "
                + "if you cannot define a signal, remove the claim from `covers` and mark it "
                + "'unclear' or 'unreproducible' with a reason (only 'mapped' claims list step_ids)."
            )

    # If all 5 attempts failed because the model produced an invalid `environment` block,
    # replace `environment` with the latest-version fallback environment and re-validate.
    if last_raw:
        try:
            parsed = json.loads(last_raw)
            if isinstance(parsed, dict):
                fallback_env = build_latest_fallback_environment(parsed, repo_dir=repo_dir)
                parsed["environment"] = fallback_env.model_dump()
                spec = RunSpec.model_validate_json(json.dumps(parsed))
                spec.warnings.append(
                    f"Model environment configuration failed validation after {max_attempts} attempts; "
                    "using fallback environment with latest versions of all modules."
                )
                print("[✓] Recovered RunSpec using fallback environment with latest module versions.")
                plog.warning("recovered RunSpec using latest-version fallback environment")
                return spec, last_response
        except Exception:
            pass
    raise last_error


def _unmapped(spec: RunSpec):
    return [c for c in spec.claims if c.status != "mapped"]


def _fill_gaps(prompt: str, spec: RunSpec, response, plog, rounds: int = 1, repo_dir: Optional[Path] = None):
    """Second planning pass for claims the first plan gave up on: ask for extra steps that
    reproduce them. The new plan is kept only if it validates and maps strictly more claims."""
    for rnd in range(1, rounds + 1):
        gaps = _unmapped(spec)
        if not gaps:
            break
        listing = "\n".join(f"- claim {c.claim_id} ({c.status}): {c.reason}" for c in gaps)
        print(f"[*] {len(gaps)} claim(s) unmapped; asking planner to find a way to reproduce them (pass {rnd})...")
        extra = (
            "\n\nYOUR PREVIOUS PLAN LEFT THESE CLAIMS UNMAPPED:\n" + listing +
            "\n\nPREVIOUS PLAN:\n" + spec.model_dump_json() +
            "\n\nReturn a COMPLETE RunSpec that keeps the working steps and ADDS steps (other scripts, "
            "flags/ablation variants, or `python -c` calls that print the value) so these claims are "
            "mapped. Give up on a claim only for reasons (a)-(c) in guideline 5, and say which."
        )
        try:
            new_spec, new_resp = _request_spec(prompt + extra, plog, repo_dir=repo_dir)
        except Exception as exc:  # keep the first plan
            plog.warning(f"gap-fill pass failed, keeping first plan: {exc}")
            break
        before, after = len(spec.mapped_claim_ids()), len(new_spec.mapped_claim_ids())
        plog.info("gap-fill pass", round=rnd, mapped_before=before, mapped_after=after)
        if after > before:
            spec, response = new_spec, new_resp
        else:
            break
    return spec, response


def _repair_preflight(prompt: str, spec: RunSpec, problems: List[str], response, plog,
                      repo_dir: Path, data_dir: Optional[Path]):
    """One extra planning pass when the local pre-flight finds concrete problems (invented
    flag values, missing data paths, ...). The corrected plan replaces the first one only if it
    validates and has strictly fewer problems, so this can never make things worse."""
    listing = "\n".join(f"- {p}" for p in problems)
    print(f"[*] Pre-flight found {len(problems)} problem(s); asking planner to fix the plan...")
    extra = (
        "\n\nA LOCAL PRE-FLIGHT CHECK OF YOUR PLAN FOUND THESE PROBLEMS:\n" + listing +
        "\n\nPREVIOUS PLAN:\n" + spec.model_dump_json() +
        "\n\nReturn a COMPLETE corrected RunSpec. Use only option values that appear in the script "
        "source, only data paths that exist under {DATA} as listed, and only scripts whose input "
        "format matches the downloaded data. If another script already runs the variant a claim needs "
        "(check its feature sets / experiment names), use that script instead. Mark a claim "
        "unreproducible only if no script in the repo can produce it."
    )
    try:
        new_spec, new_resp = _request_spec(prompt + extra, plog, repo_dir=repo_dir)
    except Exception as exc:
        plog.warning(f"pre-flight repair failed, keeping first plan: {exc}")
        return spec, response, problems
    new_problems = check_against_repo(new_spec, repo_dir, data_dir)
    before = (len(problems), -len(spec.mapped_claim_ids()))
    after = (len(new_problems), -len(new_spec.mapped_claim_ids()))
    plog.info("pre-flight repair pass", problems_before=len(problems), problems_after=len(new_problems),
              mapped_before=len(spec.mapped_claim_ids()), mapped_after=len(new_spec.mapped_claim_ids()))
    if after < before:
        return new_spec, new_resp, new_problems
    return spec, response, problems


def generate_run_spec(
    extraction_json_path: str | Path,
    repo_dir: str | Path,
    repo_url: str,
    data_dir: Optional[str | Path] = None,
    log_dir: Optional[Path] = None,
    resources: str = "",
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
    environment_info = scanner.scan_environment()
    data_files = RepoScanner.scan_data_dir(data_dir) if data_dir else []

    plog = get_phase()
    plog.info("repository scanned", commit=commit_sha, files=len(files), scripts=len(scripts_info),
              data_files=len(data_files), readme_chars=len(readme),
              gpu_script_hits=[p for p, i in scripts_info.items() if i["uses_gpu_indicators"]])
    plog.save_artifact("repo_scan.json", {"files": files, "scripts": scripts_info,
                                            "environment": environment_info, "data_files": data_files})

    prompt = build_run_spec_prompt(
        extraction=extraction,
        repo_url=repo_url,
        commit_sha=commit_sha,
        files=files,
        readme=readme,
        scripts_info=scripts_info,
        environment_info=environment_info,
        data_files=data_files,
        resources=resources,
    )

    plog.save_artifact("planner_prompt.txt", prompt)
    spec, response = _request_spec(prompt, plog, repo_dir=repo_dir)
    spec, response = _fill_gaps(prompt, spec, response, plog, repo_dir=repo_dir)
    print(f"[✓] RunSpec generated. Target hardware GPU: {spec.hardware.gpu}")
    plog.info("RunSpec validated", gpu=spec.hardware.gpu, steps=len(spec.steps),
              claims=len(spec.claims), confidence=spec.confidence)

    # Local pre-flight check (data_dir lets in_repo / {DATA} paths be verified against what was
    # actually downloaded; without it every data-repo path was reported as "not in the repo").
    problems = check_against_repo(spec, repo_dir, data_dir)
    if problems:
        spec, response, problems = _repair_preflight(prompt, spec, problems, response, plog,
                                                     repo_dir, data_dir)
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
