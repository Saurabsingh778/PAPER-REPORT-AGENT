"""Cheap checks that run BEFORE any slow stage, so a missing key or tool fails in a second
instead of after ten minutes of PDF conversion."""
from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .paths import PROJECT_ROOT


class PreflightError(RuntimeError):
    def __init__(self, checks: list["Check"]):
        self.checks = checks
        lines = [f"  - {c.name}: {c.detail}" for c in checks if c.fatal and not c.ok]
        super().__init__("preflight failed:\n" + "\n".join(lines))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True  # a failed non-fatal check is only a warning


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _kaggle_credentials() -> bool:
    if os.getenv("KAGGLE_KEY") and os.getenv("KAGGLE_USERNAME"):
        return True
    cfg_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (cfg_dir / "kaggle.json").is_file()


def run_preflight(pdf: Path, *, mode: str, reader: str, plan_only: bool) -> list[Check]:
    load_dotenv(PROJECT_ROOT / ".env")
    checks: list[Check] = []

    # the PDF
    if not pdf.is_file():
        checks.append(Check("pdf", False, f"file not found: {pdf}"))
    else:
        with open(pdf, "rb") as fh:
            head = fh.read(5)
        ok = head == b"%PDF-"
        checks.append(Check("pdf", ok, str(pdf) if ok else f"{pdf.name} does not look like a PDF"))

    # Gemini (extraction, planning, claim matching)
    checks.append(Check("gemini key", bool(os.getenv("API_KEY")),
                        "API_KEY set" if os.getenv("API_KEY") else "API_KEY is not set (put it in .env)"))
    checks.append(Check("google-genai", _has_module("google.genai"),
                        "installed" if _has_module("google.genai") else "pip install google-genai"))

    # PDF reader
    ok = _has_module(reader)
    checks.append(Check(f"{reader} reader", ok,
                        "installed" if ok else f"pip install {'marker-pdf' if reader == 'marker' else 'docling'}"
                                               f" (or use the other reader with --reader)"))

    # execution backend (not needed when we stop after planning)
    if not plan_only:
        if mode == "kaggle":
            have_cli = shutil.which("kaggle") is not None
            checks.append(Check("kaggle cli", have_cli, "found" if have_cli else "pip install kaggle"))
            have_cred = _kaggle_credentials()
            checks.append(Check("kaggle credentials", have_cred,
                                "found" if have_cred else
                                "set KAGGLE_USERNAME + KAGGLE_KEY or create ~/.kaggle/kaggle.json"))
        else:
            have_docker = shutil.which("docker") is not None
            checks.append(Check("docker", have_docker, "found" if have_docker else "docker is not on PATH"))

    # optional
    has_gh = bool(os.getenv("GITHUB_TOKEN"))
    checks.append(Check("github token", has_gh,
                        "GITHUB_TOKEN set" if has_gh else
                        "GITHUB_TOKEN not set: GitHub API is limited to 60 requests/hour", fatal=False))
    return checks


def assert_ready(checks: list[Check]) -> None:
    if any(c.fatal and not c.ok for c in checks):
        raise PreflightError(checks)
