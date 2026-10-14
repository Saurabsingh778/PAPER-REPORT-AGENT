"""Self-test for EnvirementSetup: verifies 5-attempt LLM retry and latest-version fallback.

    python -m EnvirementSetup.selftest
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from EnvirementSetup import (
    EnvironmentSetupManager,
    MAX_SETUP_ATTEMPTS,
    build_latest_fallback_environment,
    collect_all_required_modules,
    extract_setup_error_summary,
    is_setup_failure,
    run_with_env_retry,
    setup_environment_with_retry,
    strip_version_specifiers,
    unpin_repo_manifests,
)
from Schema.RunSpec import (
    ClaimMap,
    Environment,
    Evidence,
    FrameworkSpec,
    Hardware,
    PackageManagers,
    RepoRef,
    RunSpec,
    Signal,
    Step,
)

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _failures.append(msg)


def _make_gcn_repo(repo_dir: Path) -> Path:
    """Create a fixture matching the uploaded 1609-02907v4 (tkipf/gcn) repository."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "requirements.txt").write_text(
        "networkx==2.2\nscipy==1.1.0\nsetuptools==40.6.3\nnumpy==1.15.4\ntensorflow==1.15.4\n",
        encoding="utf-8",
    )
    (repo_dir / "setup.py").write_text(
        "from setuptools import setup, find_packages\n"
        "setup(name='gcn', version='1.0',\n"
        "      install_requires=['numpy>=1.15.4', 'tensorflow>=1.15.2,<2.0', 'networkx>=2.2', 'scipy>=1.1.0'],\n"
        "      packages=find_packages())\n",
        encoding="utf-8",
    )
    gcn_pkg = repo_dir / "gcn"
    gcn_pkg.mkdir(exist_ok=True)
    (gcn_pkg / "__init__.py").write_text("", encoding="utf-8")
    (gcn_pkg / "train.py").write_text(
        "import tensorflow as tf\nimport numpy as np\nimport networkx as nx\nimport scipy.sparse as sp\n"
        "print('accuracy= 0.81500')\n",
        encoding="utf-8",
    )
    return repo_dir


def _make_gcn_spec() -> RunSpec:
    return RunSpec(
        repo=RepoRef(url="https://github.com/tkipf/gcn", commit="7b05e72790807bd7a69fd7fd18cbf064a542a6f2"),
        decided_by="readme",
        environment=Environment(
            language="python",
            language_version="3.7",
            frameworks=[FrameworkSpec(name="tensorflow", version="1.15.2", backend="cpu")],
            packages=PackageManagers(
                pip=[
                    "tensorflow==1.15.2",
                    "protobuf==3.20.3",
                    "networkx==2.2",
                    "scipy==1.1.0",
                    "numpy==1.16.6",
                ]
            ),
            requirements_files=["requirements.txt"],
            requirements_policy="resolved",
        ),
        hardware=Hardware(gpu="helpful", reason="CPU or GPU"),
        steps=[
            Step(
                id="install",
                command="python setup.py install",
                cwd="{REPO}",
                gpu=False,
                timeout_min=5,
                confidence="high",
                evidence=[Evidence(kind="readme", text="python setup.py install")],
            ),
            Step(
                id="train_cora",
                command="python train.py --dataset cora",
                cwd="{REPO}/gcn",
                needs=["install"],
                covers=[1],
                signals=[Signal(claim_id=1, metric="accuracy", line_pattern=r"accuracy=\s*([0-9.]+)")],
                gpu=False,
                timeout_min=15,
                confidence="high",
                evidence=[Evidence(kind="readme", text="python train.py --dataset cora")],
            ),
        ],
        claims=[ClaimMap(claim_id=1, status="mapped", step_ids=["train_cora"])],
        confidence="high",
    )


def test_strip_and_collect(tmp: Path) -> None:
    print("1. strip_version_specifiers & collect_all_required_modules")
    check(strip_version_specifiers("tensorflow==1.15.2") == "tensorflow", "strips == pin")
    check(strip_version_specifiers("tensorflow>=1.15.2,<2.0") == "tensorflow", "strips compound >=,< pin")
    check(strip_version_specifiers("numpy~=1.19.0; python_version<'3.9'") == "numpy", "strips marker and ~=")

    repo_dir = _make_gcn_repo(tmp / "gcn_repo")
    spec = _make_gcn_spec()
    modules = collect_all_required_modules(
        spec,
        repo_dir=repo_dir,
        failure_logs=["ModuleNotFoundError: No module named 'tqdm'"],
    )
    check(
        all(m in modules for m in ("tensorflow", "networkx", "scipy", "numpy", "tqdm")),
        f"collected all unpinned modules including missing import from logs: {modules}",
    )
    check("gcn" not in modules and "setuptools" not in modules, "excludes local repo package and setuptools")


def test_failure_detection_and_unpin(tmp: Path) -> None:
    print("2. is_setup_failure, extract_setup_error_summary & unpin_repo_manifests")
    archived_failure = {
        "setup": [
            {
                "cmd": "pip install -q tensorflow==1.15.2",
                "rc": 1,
                "sec": 1,
                "stdout": "",
                "stderr": "ERROR: Could not find a version that satisfies the requirement tensorflow==1.15.2\n"
                          "ERROR: No matching distribution found for tensorflow==1.15.2\n",
            }
        ],
        "steps": [],
    }
    check(is_setup_failure(archived_failure) is True, "detects failed pip command in setup")
    summary = extract_setup_error_summary(archived_failure)
    check("tensorflow==1.15.2" in summary and "No matching distribution" in summary, "extracts stderr for model prompt")

    step_env_failure = {
        "setup": [{"cmd": "pip install -q networkx", "rc": 0, "sec": 1, "stdout": "", "stderr": ""}],
        "steps": [
            {
                "step_id": "train_citeseer",
                "cmd": "python train.py --dataset citeseer",
                "rc": 1,
                "sec": 0,
                "stdout": "",
                "stderr": "Traceback (most recent call last):\n  File 'train.py', line 5\n    import tensorflow as tf\nModuleNotFoundError: No module named 'tensorflow'\n",
            }
        ],
    }
    check(is_setup_failure(step_env_failure) is True, "detects ModuleNotFoundError in step execution as setup failure")

    repo_copy = _make_gcn_repo(tmp / "gcn_copy")
    modified = unpin_repo_manifests(repo_copy)
    check("requirements.txt" in modified and "setup.py" in modified, f"unpinned repo manifests: {modified}")
    setup_txt = (repo_copy / "setup.py").read_text(encoding="utf-8")
    req_txt = (repo_copy / "requirements.txt").read_text(encoding="utf-8")
    check("<2.0" not in setup_txt and "'tensorflow'" in setup_txt, "setup.py install_requires unpinned")
    check("==" not in req_txt and "tensorflow" in req_txt, "requirements.txt unpinned")


def test_retry_succeeds_on_third_attempt(tmp: Path) -> None:
    print("3. model retry loop succeeds when model fixes environment on attempt 3/5")
    repo_dir = _make_gcn_repo(tmp / "gcn_retry_ok")
    spec = _make_gcn_spec()
    model_prompts: list[str] = []

    def fake_llm(prompt: str) -> dict:
        model_prompts.append(prompt)
        call_num = len(model_prompts)
        if call_num == 1:
            return {
                "language": "python",
                "language_version": "3.10",
                "packages": {"pip": ["tensorflow==2.8.0", "networkx==2.2", "scipy==1.1.0"]},
                "requirements_policy": "resolved",
            }
        return {
            "language": "python",
            "language_version": "3.10",
            "packages": {"pip": ["tensorflow==2.15.0", "networkx>=2.8", "scipy>=1.10", "numpy>=1.23"]},
            "requirements_policy": "resolved",
        }

    run_calls = 0

    def fake_executor(current_spec: RunSpec, is_fallback: bool) -> dict:
        nonlocal run_calls
        run_calls += 1
        pip_pkgs = current_spec.environment.packages.pip
        if "tensorflow==2.15.0" in pip_pkgs:
            return {
                "setup": [{"cmd": "pip install tensorflow==2.15.0 ...", "rc": 0, "sec": 5, "stdout": "", "stderr": ""}],
                "steps": [{"step_id": "train_cora", "cmd": "python train.py", "rc": 0, "sec": 4, "stdout": "accuracy= 0.81500", "stderr": ""}],
            }
        return {
            "setup": [{"cmd": f"pip install {' '.join(pip_pkgs)}", "rc": 1, "sec": 1, "stdout": "", "stderr": f"ERROR: failed to install {pip_pkgs}"}],
            "steps": [],
            "setup_failed": True,
        }

    updated_spec, run_out, env_res = run_with_env_retry(
        spec,
        fake_executor,
        repo_dir=repo_dir,
        plan_dir=tmp / "plan_ok",
        llm_callable=fake_llm,
    )
    check(env_res.succeeded is True, "setup succeeded")
    check(env_res.attempts_used == 3 and run_calls == 3, f"succeeded on attempt 3/5 (attempts_used={env_res.attempts_used})")
    check(env_res.fallback_latest_used is False, "did not need latest-version fallback")
    check(len(model_prompts) == 2, "called model after attempt 1 and attempt 2 failures")
    check("Attempt 1/5" in model_prompts[0] and "Attempt 2/5" in model_prompts[1], "prompts include attempt number and failure logs")


def test_five_failures_then_latest_fallback(tmp: Path) -> None:
    print("4. 5 failed attempts -> calls model on failures -> starts environment with latest versions of all modules")
    repo_dir = _make_gcn_repo(tmp / "gcn_fallback")
    spec = _make_gcn_spec()
    plan_dir = tmp / "plan_fallback"
    plan_dir.mkdir(parents=True, exist_ok=True)

    model_calls = 0

    def fake_llm_always_bad(prompt: str) -> dict:
        nonlocal model_calls
        model_calls += 1
        return {
            "language": "python",
            "language_version": "3.10",
            "packages": {"pip": [f"tensorflow==1.15.{model_calls}", "networkx==2.2", "scipy==1.1.0", "numpy==1.15.4"]},
            "requirements_policy": "resolved",
        }

    attempt_modes: list[tuple[bool, list[str]]] = []

    def fake_executor(current_spec: RunSpec, is_fallback: bool) -> dict:
        pip_pkgs = list(current_spec.environment.packages.pip)
        attempt_modes.append((is_fallback, pip_pkgs))
        if not is_fallback:
            return {
                "setup": [
                    {
                        "cmd": f"pip install {' '.join(pip_pkgs)}",
                        "rc": 1,
                        "sec": 1,
                        "stdout": "",
                        "stderr": f"ERROR: No matching distribution found for {pip_pkgs[0]}",
                    }
                ],
                "steps": [],
                "setup_failed": True,
            }
        return {
            "setup": [
                {
                    "cmd": f"pip install --upgrade {' '.join(pip_pkgs)}",
                    "rc": 0,
                    "sec": 8,
                    "stdout": "Successfully installed latest modules",
                    "stderr": "",
                }
            ],
            "steps": [
                {
                    "step_id": "train_cora",
                    "cmd": "python train.py --dataset cora",
                    "rc": 0,
                    "sec": 3,
                    "stdout": "Test set results: cost= 1.01234 accuracy= 0.81500 time= 0.01200",
                    "stderr": "",
                }
            ],
        }

    updated_spec, run_out, env_res = run_with_env_retry(
        spec,
        fake_executor,
        repo_dir=repo_dir,
        plan_dir=plan_dir,
        max_attempts=MAX_SETUP_ATTEMPTS,
        llm_callable=fake_llm_always_bad,
    )

    check(MAX_SETUP_ATTEMPTS == 5, "MAX_SETUP_ATTEMPTS is 5")
    check(env_res.attempts_used == 5, f"tried 5 times before fallback (attempts_used={env_res.attempts_used})")
    check(model_calls == 5, f"made 5 calls to model reporting setup failure (model_calls={model_calls})")
    check(env_res.fallback_latest_used is True and env_res.succeeded is True, "fallback to latest versions ran and succeeded")
    check(len(attempt_modes) == 6, "5 model-guided attempts + 1 latest-version fallback run")

    fallback_flag, fallback_pkgs = attempt_modes[-1]
    check(fallback_flag is True, "final execution had is_fallback=True")
    check(
        all("==" not in p and ">=" not in p and "<" not in p for p in fallback_pkgs),
        f"all fallback modules are unpinned (latest version): {fallback_pkgs}",
    )
    check(
        all(m in fallback_pkgs for m in ("tensorflow", "networkx", "scipy", "numpy")),
        f"all required modules present in fallback: {fallback_pkgs}",
    )
    check((plan_dir / "env_setup_history.json").is_file(), "saved env_setup_history.json")


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_strip_and_collect(tmp)
        test_failure_detection_and_unpin(tmp)
        test_retry_succeeds_on_third_attempt(tmp)
        test_five_failures_then_latest_fallback(tmp)
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED")
        return 1
    print("all EnvirementSetup checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
