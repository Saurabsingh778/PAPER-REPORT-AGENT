"""EnvirementSetup: Resilient environment setup with 5x LLM retry and latest-version fallback.

Workflow:
  1. Attempt to set up the execution environment using the planned `Environment` specification.
  2. If the environment setup fails (e.g., broken version pins, missing wheels, incompatible
     Python version, failed `pip`/`conda`/`apt` command, `setup.py install` failure, or
     `ModuleNotFoundError`/`ImportError`), call the Gemini model with the exact error logs
     (`cmd`, `rc`, `stdout`, `stderr`), the current `Environment` JSON, the repository's
     dependency evidence, and the history of previous failed attempts.
  3. Repeat this model-guided repair loop up to `MAX_SETUP_ATTEMPTS` (5) times.
  4. If all 5 attempts fail, automatically fall back to starting the environment with the
     **latest version of all required modules** (stripping all version specifiers from pip,
     conda, frameworks, and repository manifest files so every module installs at its latest
     available version).
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from Logger import get_phase
from Schema.RunSpec import (
    Environment,
    EnvironmentMetadata,
    EnvironmentVariable,
    FrameworkSpec,
    PackageManagers,
    RunSpec,
)

MAX_SETUP_ATTEMPTS = 5

# Mapping from common framework names or top-level import names to canonical PyPI package names
FRAMEWORK_TO_PIP: Dict[str, str] = {
    "pytorch": "torch",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "keras": "keras",
    "jax": "jax",
    "flax": "flax",
    "transformers": "transformers",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "mxnet": "mxnet",
    "paddlepaddle": "paddlepaddle",
    "paddle": "paddlepaddle",
    "dgl": "dgl",
    "torch_geometric": "torch-geometric",
    "pyg": "torch-geometric",
}

IMPORT_TO_PIP: Dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "tensorflow": "tensorflow",
    "jax": "jax",
    "flax": "flax",
    "networkx": "networkx",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "tqdm": "tqdm",
    "requests": "requests",
    "h5py": "h5py",
    "tables": "tables",
    "sympy": "sympy",
    "statsmodels": "statsmodels",
    "nltk": "nltk",
    "spacy": "spacy",
    "gensim": "gensim",
    "einops": "einops",
    "timm": "timm",
    "wandb": "wandb",
    "tensorboard": "tensorboard",
    "tensorboardX": "tensorboardX",
    "optuna": "optuna",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
    "lxml": "lxml",
    "joblib": "joblib",
    "numba": "numba",
    "cython": "Cython",
    "pytest": "pytest",
    "absl": "absl-py",
    "six": "six",
    "packaging": "packaging",
    "filelock": "filelock",
    "regex": "regex",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "sentencepiece": "sentencepiece",
    "protobuf": "protobuf",
    "grpc": "grpcio",
    "dateutil": "python-dateutil",
    "pytz": "pytz",
    "psutil": "psutil",
    "rich": "rich",
    "click": "click",
    "typer": "typer",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "flask": "Flask",
    "aiohttp": "aiohttp",
    "httpx": "httpx",
    "plotly": "plotly",
    "bokeh": "bokeh",
    "imageio": "imageio",
    "skimage": "scikit-image",
}

_IGNORED_PACKAGES = {
    "python",
    "pip",
    "setuptools",
    "wheel",
    "distribute",
    "pkg_resources",
    "distutils",
    "__future__",
    "sys",
    "os",
    "re",
    "json",
    "time",
    "math",
    "random",
    "pickle",
    "cPickle",
    "collections",
    "itertools",
    "functools",
    "pathlib",
    "typing",
    "subprocess",
    "shutil",
    "glob",
    "argparse",
    "logging",
    "unittest",
    "copy",
    "io",
    "csv",
    "sqlite3",
    "urllib",
    "http",
    "socket",
    "threading",
    "multiprocessing",
    "concurrent",
    "asyncio",
    "dataclasses",
    "enum",
    "abc",
    "contextlib",
    "warnings",
    "traceback",
    "inspect",
    "ast",
    "hashlib",
    "base64",
    "gzip",
    "zipfile",
    "tarfile",
    "tempfile",
    "datetime",
    "string",
    "struct",
    "operator",
    "heapq",
    "bisect",
    "queue",
    "pprint",
    "textwrap",
    "shlex",
    "signal",
    "ctypes",
    "platform",
}

_VERSION_SPEC_RE = re.compile(r"\s*(?:===|==|~=|!=|<=|>=|<|>|@)\s*.*$")
_CONDA_SPEC_RE = re.compile(r"\s*(?:==|=|<=|>=|<|>).*$")
_MISSING_MOD_RE = re.compile(
    r"(?:ModuleNotFoundError:\s*No module named\s+['\"]([^'\"]+)['\"]"
    r"|ImportError:\s*No module named\s+([A-Za-z0-9_.\-]+)"
    r"|Requirement\.parse\(['\"]([^'\"]+)['\"]\)"
    r"|No matching distribution found for\s+([A-Za-z0-9_.\-]+))",
    re.IGNORECASE,
)
_SETUP_ERR_PATTERNS = (
    "ModuleNotFoundError:",
    "ImportError:",
    "pkg_resources.DistributionNotFound",
    "pkg_resources.VersionConflict",
    "Could not find suitable distribution for Requirement.parse",
    "Could not find a version that satisfies the requirement",
    "No matching distribution found for",
    "ResolutionImpossible",
    "ERROR: Cannot install",
    "SubprocessError: Command '['pip'",
)


@dataclass
class AttemptRecord:
    """Record of one environment setup attempt."""

    attempt: int
    mode: str  # "planned" | "model_retry" | "latest_fallback"
    environment: Dict[str, Any]
    docker_image: Optional[str] = None
    succeeded: bool = False
    error_summary: str = ""
    setup_results: List[Dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "mode": self.mode,
            "environment": self.environment,
            "docker_image": self.docker_image,
            "succeeded": self.succeeded,
            "error_summary": self.error_summary,
            "setup_results": self.setup_results,
            "duration_s": self.duration_s,
        }


@dataclass
class EnvironmentSetupResult:
    """Final outcome of `setup_environment_with_retry` / `run_with_env_retry`."""

    succeeded: bool
    attempts_used: int
    max_attempts: int
    fallback_latest_used: bool
    final_environment: Environment
    run_out: Optional[Dict[str, Any]] = None
    history: List[AttemptRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": (
                "fallback_latest_succeeded"
                if (self.succeeded and self.fallback_latest_used)
                else "succeeded"
                if self.succeeded
                else "fallback_latest_failed"
                if self.fallback_latest_used
                else "failed"
            ),
            "succeeded": self.succeeded,
            "attempts_used": self.attempts_used,
            "max_attempts": self.max_attempts,
            "fallback_latest_used": self.fallback_latest_used,
            "final_environment": _dump_model(self.final_environment),
            "history": [h.to_dict() for h in self.history],
        }


def _dump_model(model_obj: Any) -> Dict[str, Any]:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    if hasattr(model_obj, "dict"):
        return model_obj.dict()
    if isinstance(model_obj, dict):
        return dict(model_obj)
    return {}


def _dump_model_json(model_obj: Any, indent: int = 2) -> str:
    if hasattr(model_obj, "model_dump_json"):
        return model_obj.model_dump_json(indent=indent)
    return json.dumps(_dump_model(model_obj), indent=indent)


def _coerce_environment(data: Dict[str, Any]) -> Environment:
    """Validate `data` into an `Environment`. If the model returned an environment that trips
    strict cross-field compatibility checks (e.g., `tensorflow==1.15.x` on Python 3.10 without
    `protobuf==3.20.3`), preserve the model's requested fields via `model_construct`/`construct`
    so the attempt can still be evaluated and counted rather than prematurely triggering fallback."""
    try:
        if hasattr(Environment, "model_validate"):
            return Environment.model_validate(data)
        return Environment.parse_obj(data)
    except Exception:
        pkgs_raw = data.get("packages") or {}
        if isinstance(pkgs_raw, PackageManagers):
            pkgs = pkgs_raw
        elif isinstance(pkgs_raw, dict):
            pkgs = PackageManagers(
                pip=list(pkgs_raw.get("pip", data.get("pip", [])) or []),
                conda=list(pkgs_raw.get("conda", []) or []),
                apt=list(pkgs_raw.get("apt", data.get("system_packages", [])) or []),
                custom=[],
            )
        else:
            pkgs = PackageManagers(pip=list(data.get("pip", []) or []))

        construct_fn = getattr(Environment, "model_construct", None) or getattr(Environment, "construct")
        return construct_fn(
            language=str(data.get("language", "python")),
            language_version=str(data.get("language_version", "3.10")),
            cuda_version=data.get("cuda_version"),
            frameworks=[],
            docker_image=data.get("docker_image"),
            packages=pkgs,
            requirements_files=list(data.get("requirements_files", []) or []),
            requirements_policy=str(data.get("requirements_policy", "resolved")),
            env_vars=[],
            metadata=[],
            notes=str(data.get("notes", "")),
        )


def strip_version_specifiers(requirement: str, is_conda: bool = False) -> str:
    """Strip version pins, markers, and URLs from a package specifier to get the bare module name."""
    if not requirement or not isinstance(requirement, str):
        return ""
    line = requirement.strip()
    if not line or line.startswith(("#", "-", "--", "git+", "http://", "https://", "/")):
        return ""
    line = line.split("#", 1)[0].split(";", 1)[0].strip()
    if not line:
        return ""
    if is_conda:
        base = _CONDA_SPEC_RE.sub("", line).strip()
    else:
        base = _VERSION_SPEC_RE.sub("", line).strip()
    base = base.strip(" ,")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._\-\[\]]*$", base):
        return ""
    return base


def _parse_requirements_text(text: str) -> List[str]:
    pkgs: List[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-r", "--", "-e", "-f", "--find-links", "--index-url", "--extra-index-url")):
            continue
        name = strip_version_specifiers(line)
        if name and name.lower() not in _IGNORED_PACKAGES:
            pkgs.append(name)
    return pkgs


def _parse_setup_py_requires(text: str) -> List[str]:
    pkgs: List[str] = []
    for m in re.finditer(r"install_requires\s*=\s*\[([^\]]*)\]", text or "", re.DOTALL):
        body = m.group(1)
        for q in re.finditer(r"['\"]([^'\"]+)['\"]", body):
            name = strip_version_specifiers(q.group(1))
            if name and name.lower() not in _IGNORED_PACKAGES:
                pkgs.append(name)
    return pkgs


def _parse_pyproject_requires(text: str) -> List[str]:
    pkgs: List[str] = []
    for m in re.finditer(r"dependencies\s*=\s*\[([^\]]*)\]", text or "", re.DOTALL):
        body = m.group(1)
        for q in re.finditer(r"['\"]([^'\"]+)['\"]", body):
            name = strip_version_specifiers(q.group(1))
            if name and name.lower() not in _IGNORED_PACKAGES:
                pkgs.append(name)
    return pkgs


def _parse_environment_yml(text: str) -> Tuple[List[str], List[str]]:
    conda_pkgs: List[str] = []
    pip_pkgs: List[str] = []
    in_deps = False
    in_pip = False
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("dependencies:"):
            in_deps = True
            in_pip = False
            continue
        if not in_deps:
            continue
        if re.match(r"^[A-Za-z0-9_]+:", stripped) and not stripped.startswith("-"):
            if stripped.startswith("pip:"):
                in_pip = True
            else:
                in_deps = False
                in_pip = False
            continue
        if stripped.startswith("-"):
            item = stripped.lstrip("-").strip()
            if item == "pip:":
                in_pip = True
                continue
            if in_pip:
                name = strip_version_specifiers(item, is_conda=False)
                if name and name.lower() not in _IGNORED_PACKAGES:
                    pip_pkgs.append(name)
            else:
                name = strip_version_specifiers(item, is_conda=True)
                if name and name.lower() not in _IGNORED_PACKAGES:
                    conda_pkgs.append(name)
    return conda_pkgs, pip_pkgs


def _scan_repo_imports_and_manifests(repo_dir: Optional[Path]) -> Dict[str, Any]:
    """Collect all dependency evidence and bare package names from `repo_dir`."""
    if not repo_dir or not Path(repo_dir).is_dir():
        return {"manifests": {}, "pip_packages": [], "conda_packages": [], "local_packages": set()}

    repo_dir = Path(repo_dir).resolve()
    manifests: Dict[str, str] = {}
    pip_pkgs: List[str] = []
    conda_pkgs: List[str] = []

    local_packages: set[str] = set()
    for p in repo_dir.rglob("*.py"):
        rel_parts = p.relative_to(repo_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        local_packages.add(p.stem)
        if len(rel_parts) > 1:
            local_packages.add(rel_parts[0])

    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(repo_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        rel = str(path.relative_to(repo_dir))
        fname = path.name.lower()
        if re.match(r"^requirements(?:[-_.][\w-]+)?\.(?:txt|in)$", fname):
            txt = path.read_text(encoding="utf-8", errors="replace")[:15000]
            manifests[rel] = txt
            pip_pkgs.extend(_parse_requirements_text(txt))
        elif fname == "setup.py":
            txt = path.read_text(encoding="utf-8", errors="replace")[:15000]
            manifests[rel] = txt
            pip_pkgs.extend(_parse_setup_py_requires(txt))
        elif fname == "pyproject.toml":
            txt = path.read_text(encoding="utf-8", errors="replace")[:15000]
            manifests[rel] = txt
            pip_pkgs.extend(_parse_pyproject_requires(txt))
        elif re.match(r"^(?:environment|conda)(?:[-_.][\w-]+)?\.ya?ml$", fname):
            txt = path.read_text(encoding="utf-8", errors="replace")[:15000]
            manifests[rel] = txt
            c_pkgs, p_pkgs = _parse_environment_yml(txt)
            conda_pkgs.extend(c_pkgs)
            pip_pkgs.extend(p_pkgs)

    for path in list(repo_dir.rglob("*.py"))[:200]:
        rel_parts = path.relative_to(repo_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mod_names: List[str] = []
            if isinstance(node, ast.Import):
                mod_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and (node.level or 0) == 0:
                mod_names = [node.module]
            for mod in mod_names:
                top = mod.split(".", 1)[0]
                if top in local_packages or top in _IGNORED_PACKAGES:
                    continue
                mapped = IMPORT_TO_PIP.get(top)
                if mapped:
                    pip_pkgs.append(mapped)

    return {
        "manifests": manifests,
        "pip_packages": pip_pkgs,
        "conda_packages": conda_pkgs,
        "local_packages": local_packages,
    }


def collect_all_required_modules(
    spec_or_env: RunSpec | Environment | Dict[str, Any],
    repo_dir: Optional[Path] = None,
    failure_logs: Optional[List[str]] = None,
) -> List[str]:
    """Gather all required Python modules (unpinned) from the spec, repository manifests,
    AST imports, and any missing-module errors observed during failed attempts."""
    if isinstance(spec_or_env, RunSpec):
        env_dict = _dump_model(spec_or_env.environment)
    elif isinstance(spec_or_env, Environment):
        env_dict = _dump_model(spec_or_env)
    elif isinstance(spec_or_env, dict):
        env_dict = spec_or_env.get("environment", spec_or_env)
    else:
        env_dict = {}

    raw_candidates: List[str] = []

    for fw in env_dict.get("frameworks", []) or []:
        fw_name = (fw.get("name") if isinstance(fw, dict) else getattr(fw, "name", "")) or ""
        fw_clean = strip_version_specifiers(fw_name)
        if fw_clean:
            mapped = FRAMEWORK_TO_PIP.get(fw_clean.lower(), fw_clean)
            raw_candidates.append(mapped)

    packages_block = env_dict.get("packages") or {}
    if isinstance(packages_block, dict):
        pip_list = packages_block.get("pip", []) or []
        conda_list = packages_block.get("conda", []) or []
    else:
        pip_list = getattr(packages_block, "pip", []) or []
        conda_list = getattr(packages_block, "conda", []) or []

    for pkg in [*pip_list, *(env_dict.get("pip", []) or [])]:
        name = strip_version_specifiers(str(pkg))
        if name:
            raw_candidates.append(FRAMEWORK_TO_PIP.get(name.lower(), name))

    for pkg in conda_list:
        name = strip_version_specifiers(str(pkg), is_conda=True)
        if name and name.lower() not in ("python", "cudatoolkit", "cuda"):
            raw_candidates.append(FRAMEWORK_TO_PIP.get(name.lower(), name))

    repo_scan = _scan_repo_imports_and_manifests(repo_dir)
    local_packages = repo_scan["local_packages"]
    raw_candidates.extend(repo_scan["pip_packages"])

    if repo_dir and Path(repo_dir).is_dir():
        for req_rel in env_dict.get("requirements_files", []) or []:
            req_path = Path(repo_dir) / req_rel
            if req_path.is_file():
                raw_candidates.extend(
                    _parse_requirements_text(req_path.read_text(encoding="utf-8", errors="replace"))
                )

    for log_text in failure_logs or []:
        for m in _MISSING_MOD_RE.finditer(log_text or ""):
            hit = next((g for g in m.groups() if g), "")
            if not hit:
                continue
            top = strip_version_specifiers(hit).split(".", 1)[0]
            if not top or top in local_packages or top.lower() in _IGNORED_PACKAGES:
                continue
            mapped = IMPORT_TO_PIP.get(top, FRAMEWORK_TO_PIP.get(top.lower(), top))
            raw_candidates.append(mapped)

    seen_lower: set[str] = set()
    final_modules: List[str] = []
    for cand in raw_candidates:
        clean = strip_version_specifiers(cand)
        if not clean:
            continue
        low = clean.lower()
        if low in _IGNORED_PACKAGES or clean in local_packages or low in seen_lower:
            continue
        seen_lower.add(low)
        final_modules.append(clean)

    return final_modules


def build_latest_fallback_environment(
    spec_or_env: RunSpec | Environment | Dict[str, Any],
    repo_dir: Optional[Path] = None,
    failure_logs: Optional[List[str] | str] = None,
    python_version: str = "3.10",
) -> Environment:
    """Build a fallback `Environment` that installs the **latest version of all required modules**
    with all version pins removed.

    Used automatically when all 5 model-guided environment setup attempts fail.
    """
    if isinstance(failure_logs, str):
        failure_logs_list = [failure_logs]
    else:
        failure_logs_list = list(failure_logs or [])

    if isinstance(spec_or_env, RunSpec):
        base_env = _dump_model(spec_or_env.environment)
    elif isinstance(spec_or_env, Environment):
        base_env = _dump_model(spec_or_env)
    elif isinstance(spec_or_env, dict):
        base_env = spec_or_env.get("environment", spec_or_env)
    else:
        base_env = {}

    latest_pip_modules = collect_all_required_modules(
        base_env, repo_dir=repo_dir, failure_logs=failure_logs_list
    )

    unpinned_frameworks: List[FrameworkSpec] = []
    seen_fw: set[str] = set()
    for fw in base_env.get("frameworks", []) or []:
        fw_dict = fw if isinstance(fw, dict) else _dump_model(fw)
        name = (fw_dict.get("name") or "").strip()
        if name and name.lower() not in seen_fw:
            seen_fw.add(name.lower())
            unpinned_frameworks.append(
                FrameworkSpec(name=name, version=None, backend=fw_dict.get("backend"))
            )

    packages_block = base_env.get("packages") or {}
    if isinstance(packages_block, dict):
        conda_raw = packages_block.get("conda", []) or []
        apt_raw = packages_block.get("apt", base_env.get("system_packages", [])) or []
    else:
        conda_raw = getattr(packages_block, "conda", []) or []
        apt_raw = getattr(packages_block, "apt", []) or []

    unpinned_conda = []
    for c_pkg in conda_raw:
        c_name = strip_version_specifiers(str(c_pkg), is_conda=True)
        if c_name and c_name.lower() not in ("python",):
            unpinned_conda.append(c_name)
    unpinned_conda = list(dict.fromkeys(unpinned_conda))

    env_vars: List[EnvironmentVariable] = []
    for ev in base_env.get("env_vars", []) or []:
        ev_dict = ev if isinstance(ev, dict) else _dump_model(ev)
        if ev_dict.get("name"):
            env_vars.append(
                EnvironmentVariable(name=str(ev_dict["name"]), value=str(ev_dict.get("value", "")))
            )

    metadata: List[EnvironmentMetadata] = [
        EnvironmentMetadata(key="setup_mode", value="latest_fallback"),
        EnvironmentMetadata(key="fallback_after_attempts", value=str(MAX_SETUP_ATTEMPTS)),
        EnvironmentMetadata(key="unpinned_all_modules", value="true"),
    ]

    valid_req_files: List[str] = []
    for rf in base_env.get("requirements_files", []) or []:
        base_name = Path(rf).name.lower()
        if re.fullmatch(r"(?:requirements|constraints)(?:[-_.][\w-]+)?\.(?:txt|in)", base_name):
            valid_req_files.append(rf)

    return Environment(
        language="python",
        language_version=python_version,
        cuda_version=None,
        frameworks=unpinned_frameworks,
        docker_image=None,
        packages=PackageManagers(
            pip=latest_pip_modules,
            conda=unpinned_conda,
            apt=list(dict.fromkeys(str(a) for a in apt_raw if a)),
            custom=[],
        ),
        requirements_files=valid_req_files,
        requirements_policy="resolved",
        env_vars=env_vars,
        metadata=metadata,
        notes=(
            f"Fallback environment after {MAX_SETUP_ATTEMPTS} failed setup attempts: "
            "starting environment with the latest version of all required modules."
        ),
    )


def unpin_repo_manifests(repo_dir: Path) -> List[str]:
    """Rewrite `requirements*.txt` and `setup.py` inside a working copy of the repository
    so hardcoded version constraints do not cause `python setup.py install` or `pip install -r`
    to fail when running with latest modules."""
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        return []

    modified: List[str] = []
    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(repo_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        fname = path.name.lower()

        if re.match(r"^(?:requirements|constraints)(?:[-_.][\w-]+)?\.(?:txt|in)$", fname):
            orig = path.read_text(encoding="utf-8", errors="replace")
            new_lines = []
            for line in orig.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "-")):
                    new_lines.append(line)
                else:
                    bare = strip_version_specifiers(stripped)
                    if bare and bare.lower() not in ("setuptools", "pip", "wheel"):
                        new_lines.append(bare)
            updated = "\n".join(new_lines) + "\n"
            if updated != orig:
                path.write_text(updated, encoding="utf-8")
                modified.append(str(path.relative_to(repo_dir)))

        elif fname == "setup.py":
            orig = path.read_text(encoding="utf-8", errors="replace")

            def _unpin_list(match: re.Match) -> str:
                prefix, body, suffix = match.group(1), match.group(2), match.group(3)

                def _unpin_str(qmatch: re.Match) -> str:
                    quote_char = qmatch.group(1)
                    req_str = qmatch.group(2)
                    bare = strip_version_specifiers(req_str)
                    return f"{quote_char}{bare or req_str}{quote_char}"

                new_body = re.sub(r"(['\"])([^'\"]+)\1", _unpin_str, body)
                return f"{prefix}{new_body}{suffix}"

            updated = re.sub(
                r"(install_requires\s*=\s*\[)([^\]]*)(\])",
                _unpin_list,
                orig,
                flags=re.DOTALL,
            )
            if updated != orig:
                path.write_text(updated, encoding="utf-8")
                modified.append(str(path.relative_to(repo_dir)))

    return modified


def is_setup_failure(run_out: Optional[Dict[str, Any]]) -> bool:
    """Return True if `run_out` indicates that the environment setup (or package installation /
    import resolution) failed."""
    if not run_out or not isinstance(run_out, dict):
        return True

    if run_out.get("setup_failed"):
        return True

    for item in run_out.get("setup", []) or []:
        if item.get("rc", 0) != 0:
            return True

    steps = run_out.get("steps", []) or []
    for step in steps:
        rc = step.get("rc", 0)
        if rc == 0:
            continue
        step_id = str(step.get("step_id", "")).lower()
        cmd = str(step.get("cmd", "")).lower()
        combined = f"{step.get('stdout', '')}\n{step.get('stderr', '')}"

        if step_id in ("install", "setup", "build", "env", "dependencies") or any(
            k in cmd for k in ("setup.py install", "setup.py develop", "pip install", "conda install")
        ):
            return True

        if any(pat in combined for pat in _SETUP_ERR_PATTERNS):
            return True

    return False


def extract_setup_error_summary(run_out: Optional[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Extract a detailed diagnostic summary of why the environment setup failed, suitable for
    feeding directly to the LLM so it can repair the environment."""
    if not run_out or not isinstance(run_out, dict):
        return (
            "Container or environment runner failed to start or produced no results.json "
            "(the Docker image may be invalid/unavailable or the base Python interpreter crashed)."
        )

    chunks: List[str] = []
    env_info = run_out.get("environment")
    if env_info:
        chunks.append(f"Runner environment metadata: {json.dumps(env_info)}")

    for idx, item in enumerate(run_out.get("setup", []) or [], 1):
        rc = item.get("rc", 0)
        if rc != 0:
            cmd = item.get("cmd", "")
            so = (item.get("stdout") or "").strip()[-1500:]
            se = (item.get("stderr") or "").strip()[-2500:]
            chunks.append(
                f"[Setup Command #{idx} FAILED (rc={rc})]\n"
                f"Command: {cmd}\n"
                + (f"STDOUT:\n{so}\n" if so else "")
                + (f"STDERR:\n{se}\n" if se else "")
            )

    for step in run_out.get("steps", []) or []:
        rc = step.get("rc", 0)
        if rc == 0:
            continue
        step_id = str(step.get("step_id", ""))
        cmd = str(step.get("cmd", ""))
        combined = f"{step.get('stdout', '')}\n{step.get('stderr', '')}"
        if (
            step_id.lower() in ("install", "setup", "build", "env", "dependencies")
            or any(k in cmd.lower() for k in ("setup.py install", "setup.py develop", "pip install", "conda install"))
            or any(pat in combined for pat in _SETUP_ERR_PATTERNS)
        ):
            so = (step.get("stdout") or "").strip()[-1000:]
            se = (step.get("stderr") or "").strip()[-2000:]
            chunks.append(
                f"[Step '{step_id}' FAILED with Environment/Dependency Error (rc={rc})]\n"
                f"Command: {cmd}\n"
                + (f"STDOUT:\n{so}\n" if so else "")
                + (f"STDERR:\n{se}\n" if se else "")
            )

    if not chunks:
        chunks.append(f"Environment setup marked failed: {json.dumps(run_out)[:2000]}")

    summary = "\n\n".join(chunks)
    return summary[:max_chars]


def build_env_repair_prompt(
    current_env: Environment,
    error_summary: str,
    attempt: int,
    max_attempts: int = MAX_SETUP_ATTEMPTS,
    repo_dir: Optional[Path] = None,
    hardware_gpu: str = "none",
    mode: str = "local",
    history: Optional[List[AttemptRecord]] = None,
) -> str:
    """Construct the prompt sent to Gemini when environment setup attempt `attempt` fails."""
    repo_scan = _scan_repo_imports_and_manifests(repo_dir)
    manifests_json = json.dumps(repo_scan["manifests"], indent=2)[:8000]
    imports_json = json.dumps(sorted(set(repo_scan["pip_packages"])), indent=2)

    history_lines: List[str] = []
    for h in history or []:
        history_lines.append(
            f"- Attempt {h.attempt} ({h.mode}, image={h.docker_image}): FAILED\n"
            f"  Environment: {json.dumps(h.environment)}\n"
            f"  Error: {h.error_summary[:800]}"
        )
    history_block = "\n".join(history_lines) if history_lines else "(none)"

    return f"""You are an expert Python & ML environment troubleshooting agent.
The environment setup for reproducing a research paper repository FAILED on attempt {attempt} of {max_attempts}.
Your job is to diagnose the exact setup failure below and return a corrected, working `Environment` JSON specification.

EXECUTION MODE: {mode} (GPU requirement: {hardware_gpu})

CURRENT ENVIRONMENT SPECIFICATION THAT FAILED:
{_dump_model_json(current_env)}

EXACT SETUP FAILURE OUTPUT (Attempt {attempt}/{max_attempts}):
{error_summary}

PREVIOUS FAILED ATTEMPTS HISTORY (DO NOT repeat these broken configurations):
{history_block}

REPOSITORY DEPENDENCY MANIFESTS (requirements.txt / setup.py / pyproject.toml / environment.yml):
{manifests_json}

DETECTED IMPORT PACKAGES IN REPOSITORY CODE:
{imports_json}

CRITICAL INSTRUCTIONS FOR FIXING THE ENVIRONMENT:
1. Analyze the STDERR output carefully:
   - If `pip install` reported `Could not find a version that satisfies the requirement <pkg>==<ver> (from versions: ...)`, either:
     (a) Pick a compatible version that actually exists in the `(from versions: ...)` list for the active Python interpreter, OR
     (b) In `local` Docker mode, change `language_version` to a Python version that supports that package (e.g. `"3.7"` for `tensorflow==1.15.*` with `numpy==1.16.6` and `protobuf==3.20.3`, or `"3.10"` with a compatible newer version).
     (c) In `kaggle` mode, the base Python interpreter CANNOT be changed, so you MUST match `language_version` to Kaggle's Python version shown in the error log and pick package versions available for that Python!
   - If `requirements_policy` was `"install_as_is"` and a requirements file failed due to outdated or conflicting pins, change `requirements_policy` to `"resolved"` and put the resolved, compatible packages in `packages.pip`.
   - If a step failed with `ModuleNotFoundError: No module named '<mod>'` or `ImportError`, add the corresponding PyPI package to `packages.pip`.
   - If `python setup.py install` failed trying to fetch an unavailable dependency pin via `easy_install`, ensure `packages.pip` has compatible versions of all dependencies from `setup.py`.
   - If an old Debian container failed `apt-get update` or `docker pull`, switch `language_version` to a supported modern Python (`"3.9"`, `"3.10"`, or `"3.11"`), set `docker_image` to `null`, and use compatible package versions.
2. Never return the exact same `language_version` + `packages.pip` combination that already failed in the attempt history.
3. Return ONLY a valid `Environment` JSON object matching the schema.
"""


def _default_llm_generate(prompt: str) -> str:
    """Call the repository's configured Gemini client (`Model.gemini.client` and `MODEL`)
    to generate a repaired `Environment` JSON."""
    from google.genai import types
    from Model.gemini import MODEL, client

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Environment,
                temperature=0.2,
            ),
        )
        return response.text or ""
    except Exception:
        schema_str = (
            json.dumps(Environment.model_json_schema())
            if hasattr(Environment, "model_json_schema")
            else json.dumps(Environment.schema())
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt + f"\n\nReturn ONLY valid JSON matching this Environment schema:\n{schema_str}",
        )
        raw = (response.text or "").strip()
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()


def repair_environment_with_model(
    current_env: Environment,
    error_summary: str,
    attempt: int,
    max_attempts: int = MAX_SETUP_ATTEMPTS,
    repo_dir: Optional[Path] = None,
    hardware_gpu: str = "none",
    mode: str = "local",
    history: Optional[List[AttemptRecord]] = None,
    llm_callable: Optional[Callable[[str], str | Environment | Dict[str, Any]]] = None,
) -> Environment:
    """Call the model once to notify it that environment setup attempt `attempt` failed and
    return its proposed `Environment` specification."""
    plog = get_phase()
    prompt = build_env_repair_prompt(
        current_env=current_env,
        error_summary=error_summary,
        attempt=attempt,
        max_attempts=max_attempts,
        repo_dir=repo_dir,
        hardware_gpu=hardware_gpu,
        mode=mode,
        history=history,
    )
    plog.save_artifact(f"env_repair_prompt_attempt_{attempt}.txt", prompt)
    print(
        f"[*] Environment setup attempt {attempt}/{max_attempts} failed. "
        f"Calling model to repair environment configuration..."
    )

    generator = llm_callable or _default_llm_generate
    try:
        reply = generator(prompt)
        if isinstance(reply, Environment):
            repaired = reply
            raw_text = _dump_model_json(repaired)
        elif isinstance(reply, dict):
            raw_text = json.dumps(reply, indent=2)
            repaired = _coerce_environment(reply)
        else:
            raw_text = str(reply or "").strip()
            raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text).strip()
            parsed = json.loads(raw_text)
            repaired = _coerce_environment(parsed)

        plog.save_artifact(f"env_repair_response_attempt_{attempt}.json", raw_text)
        plog.info(
            "model repaired environment",
            attempt=attempt,
            language_version=repaired.language_version,
            pip_count=len(repaired.packages.pip),
            policy=repaired.requirements_policy,
        )
        return repaired
    except Exception as exc:
        plog.warning(f"model environment repair call failed on attempt {attempt}: {exc}")
        print(f"[!] Model repair call failed ({exc}); keeping current environment for next retry.")
        return current_env


def run_with_env_retry(
    spec: RunSpec,
    executor_fn: Callable[[RunSpec, bool], Optional[Dict[str, Any]]],
    *,
    repo_dir: Optional[Path] = None,
    plan_dir: Optional[Path] = None,
    mode: str = "local",
    max_attempts: int = MAX_SETUP_ATTEMPTS,
    llm_callable: Optional[Callable[[str], str | Environment | Dict[str, Any]]] = None,
    resolve_image_fn: Optional[Callable[[Environment], str]] = None,
    fallback_python_version: str = "3.10",
) -> Tuple[RunSpec, Optional[Dict[str, Any]], EnvironmentSetupResult]:
    """Execute `spec` via `executor_fn(current_spec, is_fallback_latest)` with up to
    `max_attempts` (5) model-guided environment setup attempts.

    If an attempt fails environment setup:
      - Calls the model (`repair_environment_with_model`) with the failure details.
      - Updates `spec.environment` and retries (up to `max_attempts` = 5 attempts).
    If all 5 attempts fail:
      - Automatically falls back to starting the environment with the **latest version of all
        required modules** (`build_latest_fallback_environment` + `is_fallback_latest=True`).
    """
    plog = get_phase()
    working_spec = (
        spec.model_copy(deep=True)
        if hasattr(spec, "model_copy")
        else copy.deepcopy(spec)
    )
    history: List[AttemptRecord] = []
    failure_logs: List[str] = []
    last_run_out: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        attempt_mode = "planned" if attempt == 1 else "model_retry"
        img = resolve_image_fn(working_spec.environment) if resolve_image_fn else working_spec.environment.docker_image
        print(
            f"\n[EnvirementSetup] Attempt {attempt}/{max_attempts} ({attempt_mode}): "
            f"Python {working_spec.environment.language_version}, "
            f"pip={working_spec.environment.packages.pip}"
        )
        plog.info(
            "environment setup attempt started",
            attempt=attempt,
            max_attempts=max_attempts,
            mode=attempt_mode,
            language_version=working_spec.environment.language_version,
            pip=working_spec.environment.packages.pip,
            image=img,
        )

        t0 = time.time()
        run_out = executor_fn(working_spec, False)
        duration = round(time.time() - t0, 2)
        last_run_out = run_out

        if not is_setup_failure(run_out):
            rec = AttemptRecord(
                attempt=attempt,
                mode=attempt_mode,
                environment=_dump_model(working_spec.environment),
                docker_image=img,
                succeeded=True,
                error_summary="",
                setup_results=(run_out or {}).get("setup", []),
                duration_s=duration,
            )
            history.append(rec)
            print(f"[✓] [EnvirementSetup] Environment setup succeeded on attempt {attempt}/{max_attempts}.")
            plog.info("environment setup succeeded", attempt=attempt, duration_s=duration)

            env_result = EnvironmentSetupResult(
                succeeded=True,
                attempts_used=attempt,
                max_attempts=max_attempts,
                fallback_latest_used=False,
                final_environment=working_spec.environment,
                run_out=run_out,
                history=history,
            )
            if isinstance(run_out, dict):
                run_out["env_setup"] = env_result.to_dict()
            if plan_dir and Path(plan_dir).is_dir():
                (Path(plan_dir) / "run_spec.json").write_text(
                    _dump_model_json(working_spec, indent=2), encoding="utf-8"
                )
                (Path(plan_dir) / "env_setup_history.json").write_text(
                    json.dumps(env_result.to_dict(), indent=2), encoding="utf-8"
                )
            return working_spec, run_out, env_result

        # Setup failed on this attempt
        err_summary = extract_setup_error_summary(run_out)
        failure_logs.append(err_summary)
        rec = AttemptRecord(
            attempt=attempt,
            mode=attempt_mode,
            environment=_dump_model(working_spec.environment),
            docker_image=img,
            succeeded=False,
            error_summary=err_summary,
            setup_results=(run_out or {}).get("setup", []),
            duration_s=duration,
        )
        history.append(rec)
        print(f"[!] [EnvirementSetup] Attempt {attempt}/{max_attempts} failed:\n{err_summary[:600]}")
        plog.warning(
            "environment setup attempt failed",
            attempt=attempt,
            max_attempts=max_attempts,
            error=err_summary[:1000],
        )

        # Make a call to the model reporting that this setup failed.
        # On attempts 1..(max_attempts-1), the model's repaired environment is used for the next attempt.
        # On attempt == max_attempts, if a custom llm_callable is supplied (or for logging the 5th model
        # notification), we notify the model and then switch to the latest-versions fallback.
        if attempt < max_attempts:
            repaired_env = repair_environment_with_model(
                current_env=working_spec.environment,
                error_summary=err_summary,
                attempt=attempt,
                max_attempts=max_attempts,
                repo_dir=repo_dir,
                hardware_gpu=working_spec.hardware.gpu,
                mode=mode,
                history=history,
                llm_callable=llm_callable,
            )
            working_spec.environment = repaired_env
            if plan_dir and Path(plan_dir).is_dir():
                (Path(plan_dir) / f"run_spec_attempt_{attempt + 1}.json").write_text(
                    _dump_model_json(working_spec, indent=2), encoding="utf-8"
                )
        elif llm_callable is not None:
            repair_environment_with_model(
                current_env=working_spec.environment,
                error_summary=err_summary,
                attempt=attempt,
                max_attempts=max_attempts,
                repo_dir=repo_dir,
                hardware_gpu=working_spec.hardware.gpu,
                mode=mode,
                history=history,
                llm_callable=llm_callable,
            )

    # All 5 attempts failed -> Fallback: start the environment with the latest version of all modules
    print(
        f"\n[!] [EnvirementSetup] All {max_attempts} environment setup attempts failed. "
        "Starting fallback environment with the LATEST version of all required modules..."
    )
    plog.warning(
        "all environment setup attempts failed; falling back to latest versions of all modules",
        attempts=max_attempts,
    )

    fallback_env = build_latest_fallback_environment(
        working_spec,
        repo_dir=repo_dir,
        failure_logs=failure_logs,
        python_version=fallback_python_version,
    )
    working_spec.environment = fallback_env
    fallback_warning = (
        f"Environment setup failed after {max_attempts} model-guided attempts; "
        f"started environment with latest versions of all modules: {fallback_env.packages.pip}"
    )
    if fallback_warning not in working_spec.warnings:
        working_spec.warnings.append(fallback_warning)

    img = resolve_image_fn(working_spec.environment) if resolve_image_fn else working_spec.environment.docker_image
    t0 = time.time()
    fallback_run_out = executor_fn(working_spec, True)
    duration = round(time.time() - t0, 2)
    fallback_ok = not is_setup_failure(fallback_run_out)

    history.append(
        AttemptRecord(
            attempt=max_attempts + 1,
            mode="latest_fallback",
            environment=_dump_model(working_spec.environment),
            docker_image=img,
            succeeded=fallback_ok,
            error_summary="" if fallback_ok else extract_setup_error_summary(fallback_run_out),
            setup_results=(fallback_run_out or {}).get("setup", []),
            duration_s=duration,
        )
    )

    if fallback_ok:
        print("[✓] [EnvirementSetup] Fallback environment with latest module versions succeeded.")
        plog.info("fallback latest-version environment succeeded", pip=fallback_env.packages.pip)
    else:
        print("[x] [EnvirementSetup] Fallback environment with latest module versions also encountered errors.")
        plog.error("fallback latest-version environment failed")

    final_out = fallback_run_out if fallback_run_out is not None else last_run_out
    env_result = EnvironmentSetupResult(
        succeeded=fallback_ok,
        attempts_used=max_attempts,
        max_attempts=max_attempts,
        fallback_latest_used=True,
        final_environment=working_spec.environment,
        run_out=final_out,
        history=history,
    )
    if isinstance(final_out, dict):
        final_out["env_setup"] = env_result.to_dict()
    if plan_dir and Path(plan_dir).is_dir():
        (Path(plan_dir) / "run_spec.json").write_text(
            _dump_model_json(working_spec, indent=2), encoding="utf-8"
        )
        (Path(plan_dir) / "env_setup_history.json").write_text(
            json.dumps(env_result.to_dict(), indent=2), encoding="utf-8"
        )
    return working_spec, final_out, env_result


def _run_host_env_setup(
    env: Environment,
    repo_dir: Path,
    python_executable: str = sys.executable,
    is_fallback_latest: bool = False,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    """Directly execute environment setup commands (`pip install`, `conda install`, etc.)
    in a local host/virtualenv Python interpreter and return a `run_out`-compatible dict."""
    setup_logs: List[Dict[str, Any]] = []
    repo_dir = Path(repo_dir).resolve()

    def _exec(cmd: str, timeout: int = timeout_sec) -> Dict[str, Any]:
        t0 = time.time()
        try:
            p = subprocess.run(
                cmd,
                shell=True,
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "cmd": cmd,
                "rc": p.returncode,
                "sec": round(time.time() - t0, 2),
                "stdout": p.stdout[-8000:],
                "stderr": p.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "cmd": cmd,
                "rc": -9,
                "sec": timeout,
                "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr": "TIMEOUT",
            }

    if is_fallback_latest:
        unpin_repo_manifests(repo_dir)

    if env.requirements_policy == "install_as_is" and not is_fallback_latest:
        for rel in env.requirements_files:
            req_path = repo_dir / rel
            if req_path.is_file():
                cmd = f"{shlex.quote(python_executable)} -m pip install -q --retries 5 --timeout 300 -r {shlex.quote(str(req_path))}"
                setup_logs.append(_exec(cmd))

    if env.packages.pip:
        upgrade_flag = "--upgrade " if is_fallback_latest else ""
        pkgs = " ".join(shlex.quote(str(p)) for p in env.packages.pip)
        cmd = f"{shlex.quote(python_executable)} -m pip install -q {upgrade_flag}--retries 5 --timeout 300 {pkgs}"
        setup_logs.append(_exec(cmd))

    if env.packages.conda:
        pkgs = " ".join(shlex.quote(str(p)) for p in env.packages.conda)
        setup_logs.append(_exec(f"conda install -y {pkgs}"))

    failed = any(r.get("rc", 0) != 0 for r in setup_logs)
    return {
        "setup": setup_logs,
        "steps": [],
        "setup_failed": failed,
        "setup_failure_count": sum(1 for r in setup_logs if r.get("rc", 0) != 0),
    }


def setup_environment_with_retry(
    spec_or_env: RunSpec | Environment,
    repo_dir: Path,
    *,
    max_attempts: int = MAX_SETUP_ATTEMPTS,
    python_executable: str = sys.executable,
    executor_fn: Optional[Callable[[Environment, bool], Dict[str, Any]]] = None,
    llm_callable: Optional[Callable[[str], str | Environment | Dict[str, Any]]] = None,
    plan_dir: Optional[Path] = None,
) -> EnvironmentSetupResult:
    """Standalone entry point to set up an `Environment` (or `RunSpec.environment`) with up to
    `max_attempts` (5) model-guided retries on failure, followed by a fallback to installing
    the latest versions of all required modules.
    """
    repo_dir = Path(repo_dir).resolve()
    if isinstance(spec_or_env, RunSpec):
        dummy_spec = (
            spec_or_env.model_copy(deep=True)
            if hasattr(spec_or_env, "model_copy")
            else copy.deepcopy(spec_or_env)
        )
    else:
        from Schema.RunSpec import Hardware, RepoRef

        dummy_spec = RunSpec(
            repo=RepoRef(url="https://github.com/local/repo", commit="0" * 40),
            decided_by="readme",
            environment=spec_or_env,
            hardware=Hardware(gpu="none", reason="standalone environment setup"),
            steps=[],
            claims=[],
            confidence="medium",
        )

    def _wrapped_executor(s: RunSpec, is_fallback: bool) -> Dict[str, Any]:
        if executor_fn is not None:
            return executor_fn(s.environment, is_fallback)
        return _run_host_env_setup(
            s.environment,
            repo_dir=repo_dir,
            python_executable=python_executable,
            is_fallback_latest=is_fallback,
        )

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    _, _, env_result = run_with_env_retry(
        dummy_spec,
        _wrapped_executor,
        repo_dir=repo_dir,
        plan_dir=plan_dir,
        mode="local",
        max_attempts=max_attempts,
        llm_callable=llm_callable,
        fallback_python_version=py_ver,
    )
    return env_result


class EnvironmentSetupManager:
    """Object-oriented interface for managing environment setup with 5-attempt LLM retry and
    latest-version module fallback."""

    def __init__(
        self,
        max_attempts: int = MAX_SETUP_ATTEMPTS,
        llm_callable: Optional[Callable[[str], str | Environment | Dict[str, Any]]] = None,
        fallback_python_version: str = "3.10",
    ):
        self.max_attempts = max_attempts
        self.llm_callable = llm_callable
        self.fallback_python_version = fallback_python_version

    def setup(
        self,
        spec_or_env: RunSpec | Environment,
        repo_dir: Path,
        executor_fn: Optional[Callable[[Environment, bool], Dict[str, Any]]] = None,
        plan_dir: Optional[Path] = None,
    ) -> EnvironmentSetupResult:
        return setup_environment_with_retry(
            spec_or_env,
            repo_dir=repo_dir,
            max_attempts=self.max_attempts,
            executor_fn=executor_fn,
            llm_callable=self.llm_callable,
            plan_dir=plan_dir,
        )

    def run_spec_with_retry(
        self,
        spec: RunSpec,
        executor_fn: Callable[[RunSpec, bool], Optional[Dict[str, Any]]],
        *,
        repo_dir: Optional[Path] = None,
        plan_dir: Optional[Path] = None,
        mode: str = "local",
        resolve_image_fn: Optional[Callable[[Environment], str]] = None,
    ) -> Tuple[RunSpec, Optional[Dict[str, Any]], EnvironmentSetupResult]:
        return run_with_env_retry(
            spec,
            executor_fn,
            repo_dir=repo_dir,
            plan_dir=plan_dir,
            mode=mode,
            max_attempts=self.max_attempts,
            llm_callable=self.llm_callable,
            resolve_image_fn=resolve_image_fn,
            fallback_python_version=self.fallback_python_version,
        )
