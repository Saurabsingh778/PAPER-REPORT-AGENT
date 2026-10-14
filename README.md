# PAPER-REPORT-AGENT

An agent that checks whether a paper's headline results can actually be reproduced.

Give it a paper PDF. It converts the paper to markdown, extracts the headline numbers and the
code/data links, plans how to run the authors' code, sets up the environment with automatic
model-guided error recovery (up to 5 attempts before falling back to the latest versions of all
modules), executes it (on a free Kaggle GPU or in local Docker), compares the numbers it gets
against the paper, and writes a reproducibility report.

**Status: working end-to-end demo.** PDF in, `report.md` out, with every intermediate artifact kept
in one folder per run. The execution environment is built per paper from the repo itself and
automatically repaired by `EnvirementSetup` when legacy dependency pins fail, with a web UI sitting
on top of the pipeline.

```
python -m ui.server                           # web UI at http://127.0.0.1:8000
python main.py run paper.pdf                  # execute on Kaggle (default)
python main.py run paper.pdf --local          # execute locally in Docker
python main.py run paper.pdf --plan-only      # stop after the RunSpec; nothing is executed
python main.py run --resume Outputs/<paper>/<run_id>   # reuse the finished pdf/extract stages
python main.py list                           # every run, newest first
python main.py latest [paper-slug]            # path of the newest report
python -m Pipeline.selftest                   # offline check of the integration
python -m EnvirementSetup.selftest            # offline check of 5x env retry + latest fallback
```

## How it works

```
 paper.pdf
    |
    v
 [1 pdf]       Marker (default) or Docling  ->  paper.md + figures
    |
    v
 [2 extract]   Gemini  ->  main results (claim, metric, value), code repos, data links
    |                       each value/URL is flagged "verified" only if it appears verbatim in the paper
    v
 [3 pipeline]  fetch repo + data  ->  RepoHandler scans the repo (code + environment files)
    |          ->  plans a RunSpec (steps + environment)  ->  (feasibility gate, local mode)
    |          ->  EnvirementSetup builds & verifies the environment (up to 5 model-guided attempts,
    |              then fallback to the latest version of all required modules)
    |          ->  execute steps (Kaggle or Docker)  ->  match output to claims  ->  compare
    v
 [4 report]    report.md + report.json   (always written, even if a stage failed)
```

**RepoHandler and the RunSpec.** The planner does not run anything itself. It reads the repo's file
list, README, script heads (plus a static scan for CUDA use) and its dependency files, and produces
a `RunSpec`: one schema-validated plan used for every paper. The plan lists the data to fetch, the
execution environment, the hardware decision (GPU required / helpful / none, with evidence), the
ordered steps, and, for each paper claim, which step produces it and the regex that finds its number
in the output. A claim that cannot be reproduced must say why; none is dropped silently. The runners
execute the spec and make no decisions of their own.

Guardrails around the model's plan:

- The schema rejects absolute paths, README placeholders (`/path/to/...`), unknown tokens, claims
  with no signal, and inconsistent GPU flags. A rejected reply is sent back to the model with the
  validator's errors (up to 5 attempts, with automatic fallback to the latest module versions if
  environment validation still fails).
- A second pass asks the planner to find a way to reproduce any claim it gave up on.
- The environment section is validated too, including known-incompatible stacks; the model is told
  exactly why a plan was rejected.
- During execution, `EnvirementSetup` monitors environment setup and dependency installation. If
  setup fails, it sends the exact failure output (`cmd`, `rc`, `stdout`, `stderr`) back to the model
  to repair the environment (up to 5 attempts), and if all 5 attempts fail, it starts the
  environment with the latest version of all required modules.
- A local pre-flight reads the cloned repo and checks the plan against it (scripts exist, data paths
  exist, flag values appear in the script source, patch strings are present). If it finds problems,
  one repair pass runs, and the corrected plan is kept only if it is strictly better.
- Code is never edited except for explicit, reviewable `PathRewrite` patches that swap a hardcoded
  path for `{REPO}` / `{DATA}` (and manifest unpinning in the working copy when the latest-version
  fallback is triggered).

**How the verdict is reached.** Gemini only *locates* the output line that reports each claimed
quantity. Code then verifies it: the quoted line must exist verbatim in the real output, the value
must appear in that line, and it must be within 3% (relative) of the paper's value (a percent vs.
fraction difference such as 95.1 vs 0.951 is tolerated). If the LLM matching call fails, a plain
number-in-output heuristic is used instead. The verdict itself is never generated by a model.

Failures are classified, so an environment problem is not reported as a paper that failed to
reproduce:

| Status | Meaning |
|---|---|
| `reproduced` | a matching number was found within tolerance |
| `ran_but_mismatch` | the code ran, but the reported value differs or was not found |
| `not_attempted` | no step covered the claim (see the planner's reason) |
| `timeout` / `env_failed` / `data_missing` / `run_failed` | the step failed; the reason is classified from its stderr |

## Dynamic environments and `EnvirementSetup`

Old research code often needs an old stack (Python 3.6 with PyTorch 0.4, TensorFlow 1.x), and
installing it into a current default Python fails. Conversely, old pinned wheels in `requirements.txt`
or `setup.py` may no longer exist on PyPI for the target container architecture. So the environment
is worked out per paper and backed by an automated 5-attempt repair and fallback loop in
`EnvirementSetup/`.

**1. Discovery (`RepoScanner.scan_environment`).** Before planning, the scanner collects evidence
without guessing versions:

- dependency files, up to 12,000 characters each: `requirements*.txt/.in`, `environment*.yml`,
  `pyproject.toml`, `setup.py`, `setup.cfg`, `Pipfile`, `poetry.lock`, `conda.yml`, `Dockerfile`;
- Python and framework version hints (PyTorch, TensorFlow, JAX) found in that text;
- the packages the code imports, from an AST scan of up to 200 scripts, mapped from import names
  to package names (for example `PIL` to `Pillow`, `sklearn` to `scikit-learn`). This list is a
  fallback only.

**2. Planning.** The planner receives this evidence and must fill in the environment: language and
exact `language_version`, frameworks (name, version, backend), CUDA version, packages grouped by
installer (`pip`, `conda`, `apt`, and custom managers), requirements files and a policy. Rules it is
given: manifest text is authoritative and imports are fallback-only; a bounded compatibility
statement in the README overrides a loose lower bound in a manifest; pick one concrete Python
version, never the host's default; never pair an old framework with a modern Python. If the repo
documents nothing, the planner infers the stack from the code, resolves it to concrete packages, and
records that with low-confidence `llm_judgment` evidence. `docker_image` is an optional override;
leave it empty and the executor chooses.

| `requirements_policy` | Effect |
|---|---|
| `resolved` (default) | install the planner's concrete `packages`; manifests are evidence only |
| `install_as_is` | run `pip install -r` on the listed requirements files, then the `packages` |
| `ignore` | record the manifests but do not install them |

**3. Validation (`Schema/RunSpec.py`).** The schema rejects a plan, and the planner must correct it,
when: `language_version` is not concrete (for example `3.6` or `3.11.4`); `requirements_files` lists
anything other than pip requirements/constraints files (`setup.py`, `pyproject.toml` and
`environment.yml` are evidence, not `pip install -r` inputs); a framework is listed twice;
TensorFlow 1.x is planned on a Python outside 3.5 to 3.7, with a `numpy` pin below 1.16, or without
a `protobuf` pin; or PyTorch 0.x is planned on a Python other than 2.7, 3.5 or 3.6. If the model
still fails environment schema validation after 5 attempts, `RepoHandler` automatically substitutes
the unpinned latest-version fallback environment so planning can proceed. Specs saved before this
schema (flat `pip`, `requirements_file`, `system_packages`) still load and are normalised into the
new contract.

**4. Execution (`Sandbox/local_runner.py` and `Sandbox/kaggle_runner.py`).**

- **Local Docker** picks the base image from the environment (`Sandbox/docker_resolver.py`):
  an explicit `docker_image` wins; PyTorch 0.x with Python 3.6 on a GPU run uses
  `pytorch/pytorch:0.4.1-cuda9-cudnn7-devel`; PyTorch 2.2 or 2.3 with Python 3.10 on a GPU run uses
  the matching `pytorch/pytorch` CUDA runtime image; everything else uses `python:<major.minor>-slim`
  at exactly the requested version, rather than silently upgrading the interpreter.
  Inside the container the runner checks that the interpreter matches the request, sets `env_vars`,
  then installs `apt` packages, requirements files (policy `install_as_is`), the resolved `pip`
  packages in one command, `conda` packages and custom managers, in that order. If a step running
  `python setup.py install` or `python setup.py develop` fails because of deprecated `easy_install`
  dependency resolution after packages are already installed, the runner automatically retries it
  with `pip install --no-build-isolation --no-deps .`.
- **Kaggle** cannot replace its Python, so it checks the interpreter first and stops with a clear
  message ("Use --local") when the plan needs a different version. Otherwise it applies `env_vars`,
  installs requirements files and resolved `pip` packages. It does not install `apt`, `conda` or
  custom-manager packages.

**5. Automatic 5-Attempt Model Repair & Latest-Version Fallback (`EnvirementSetup/`).**

Even when a planned environment passes static schema checks, real package installation can fail at
run time (missing wheels for a specific version, conflicting sub-dependencies, failed `setup.py`
installs, or missing imports). `EnvirementSetup` wraps execution with a self-healing loop:

1. **Failure detection (`is_setup_failure` & `extract_setup_error_summary`).** An attempt is flagged
   as an environment setup failure if the container fails to start, the Python version check fails,
   any `apt`/`pip`/`conda`/custom setup command exits non-zero, a prerequisite install step
   (`setup.py install`, `pip install`) fails, or a step crashes with `ModuleNotFoundError`,
   `ImportError`, `DistributionNotFound`, or `VersionConflict`.
2. **Up to 5 model-guided attempts (`run_with_env_retry` & `repair_environment_with_model`).** When
   an attempt fails, `EnvirementSetup` sends the failed `Environment` JSON, the exact `stdout` and
   `stderr` tails (including available version lists printed by `pip`), the repository's manifests
   and AST-detected imports, and the history of all prior failed attempts to Gemini. The model
   returns a repaired `Environment`, the Docker image is re-resolved, and setup is retried—up to
   `MAX_SETUP_ATTEMPTS = 5` attempts in total.
3. **Fallback to latest versions of all modules (`build_latest_fallback_environment` &
   `unpin_repo_manifests`).** If all 5 attempts fail, `EnvirementSetup` collects every required
   module across the `RunSpec`, repository manifests (`requirements*.txt`, `setup.py`,
   `pyproject.toml`, `environment*.yml`), AST imports, and `ModuleNotFoundError` logs, strips all
   version specifiers (`==`, `>=`, `<=`, `<`, `>`, `~=`), unpins `setup.py` and `requirements*.txt`
   in the working repository copy, injects a lightweight `sitecustomize.py` compatibility shim for
   legacy NumPy/SciPy/NetworkX aliases, and starts the environment installing the **latest version**
   of all modules (`pip install --upgrade ...`).

## Setup

Requires Python 3.10+.

```
pip install google-genai pydantic python-dotenv requests pillow
pip install marker-pdf          # default PDF reader (or: pip install docling, and use --reader docling)
pip install kaggle              # Kaggle mode only
```

Docker is needed for `--local` (and the NVIDIA container toolkit if you want the GPU). There is no
requirements file for the pipeline yet; the lines above are the packages the code imports. The UI
has its own (`pip install -r ui/requirements.txt`).

Create a `.env` file in the project root (never commit it) with:

| Variable | Used for |
|---|---|
| `API_KEY` | Gemini: extraction, RunSpec planning, environment repair, claim matching (required) |
| `MODEL` | Gemini model name (default `gemini-3.5-flash-lite`) |
| `GITHUB_TOKEN` | optional; raises the GitHub API limit from 60 to 5000 requests/hour |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle mode (or use `~/.kaggle/kaggle.json`) |
| `AGENT_OUTPUT_ROOT` | where run folders are written (default `./Outputs`) |

A missing key or tool is reported in about a second, before any slow work starts. The whole stack
is built to run at no cost: Gemini free tier, the GitHub API, and Kaggle's free notebook GPU.

## Execution modes

| | Kaggle (default) | Local (`--local`) |
|---|---|---|
| Where | a private Kaggle script kernel, GPU on when the plan needs it, internet on | a Docker container on this machine |
| Code and data | cloned inside the kernel | downloaded by the agent into a per-paper cache, then copied into the container |
| Environment | Kaggle's own Python (a plan that needs another version stops with "Use --local"), plus the plan's pip packages | an image chosen from the plan's Python/framework versions, plus apt, pip, conda and custom packages, with 5x model-guided setup retry and latest-version fallback |
| Before running | pushes the kernel, polls its status (up to 4 h) | probes GPU, VRAM, RAM, disk and Docker, then gates the plan |

In local mode the **feasibility gate** is rule-based (no LLM): steps that need more VRAM or time
than the machine has are dropped, and their claims are marked unreproducible with the concrete
reason. Everything that can run still runs. If the GPU runtime is not available to Docker, the run
falls back to CPU with a warning.

## Where the output goes

Everything for a run is in one folder. Start with `report.md`.

```
Outputs/                              (override with --out or AGENT_OUTPUT_ROOT)
  index.json                          every run, newest first
  <paper-slug>/
    latest -> <run_id>                newest run of this paper
    _cache/Repo, _cache/Data          downloaded code/data, reused by later runs of this paper
    <run_id>/                         e.g. 20261012-074500
      report.md                       the final report
      report.json                     the same report, machine-readable
      manifest.json                   run status + per-stage status, timings, outputs
      00_input/<paper>.pdf            the PDF as received
      01_pdf/                         paper.md, figures, block JSON
      02_extraction/                  extraction.json (claims, repos, data links), extraction.md
      03_plan/                        run_spec.json, run_spec_attempt_<N>.json, env_setup_history.json + planner logs
      04_execution/                   runner workspace, execution_results.json, comparison_rows.json, phase2.md
      logs/                           full Logger output for the run
```

The report lists every claim with the paper's value, the reproduced value, the exact output line it
came from, the matcher's one-line reasoning, and the planner's status for it, followed by the steps
that ran (command, exit code, seconds), anything not reproducible, and warnings.

A run that fails part-way still ends with a `report.md` that says which stage failed and why;
`manifest.json` shows which stages finished. `--resume` reuses the slow PDF and extraction stages
and redoes the rest.

## Web UI

A live dashboard over `Pipeline.run_paper()`. It adds no pipeline logic: UI runs write the same
`Outputs/<paper>/<run_id>/` folders as the CLI, and CLI runs appear in the UI's history.

```
pip install -r ui/requirements.txt
python -m ui.server                  # http://127.0.0.1:8000
python -m ui.server --demo           # simulated stages: no keys, Docker, Kaggle or GPU needed
python -m ui.server --out <dir> --port 9000
```

- **New run**: drop a PDF, pick Kaggle or Local Docker, Marker or Docling, and plan-only. The
  environment card runs the same preflight checks as the CLI and blocks *Start* until the fatal ones
  pass.
- **Live run**: stage rail with timings, a streaming console, and the extracted claims and planned
  steps as soon as those files exist. Refreshing the page replays the run.
- **Runs** and **Report**: history with status and score, then per-claim detail (paper vs reproduced
  value, matched output line, planner reason), the plan, the paper's figures, `report.md` and a file
  browser. *Re-run* uses `--resume`.
- A JSON API with Server-Sent Events lives under `/api` (docs at `/api/docs`).

Limits: one run at a time, no cancel button (`run_paper()` has no cancellation hook), and no
authentication, so it binds to `127.0.0.1` and should not be exposed on a network. Demo-mode numbers
are simulated and the generated RunSpec says so. Marker's GPU memory is released after each PDF
conversion, so the long-lived server does not hold it.

## From code

```python
from pathlib import Path
from Pipeline import run_paper, list_runs, latest_run
from EnvirementSetup import setup_environment_with_retry, EnvironmentSetupManager

result = run_paper("paper.pdf", mode="local", on_event=print)    # events: {stage, status, message, ts}
result.report_md, result.status, (result.reproduced, result.total)

# Standalone environment setup with 5x model retry + latest-version fallback:
env_res = setup_environment_with_retry(spec, repo_dir=Path("./Downloads/Repo/tkipf_gcn"))
env_res.succeeded, env_res.attempts_used, env_res.fallback_latest_used
```

`manifest.json` is updated at every stage boundary, so progress can also be polled from disk.

## Layout

| Folder | Role |
|---|---|
| `main.py` | command-line entry point (`run`, `list`, `latest`) |
| `Pipeline/` | the glue: run folder layout, manifest, preflight, one module per stage, `run_paper()` |
| `Pdf-Reader/` | PDF to markdown (Marker, Docling) |
| `Model/`, `Schema/` | Gemini extraction; `phase_1_schema` and the `RunSpec` contract |
| `RepoHandler/` | fetch the repo + data, scan code and environment files, plan the run (`RunSpec`) |
| `EnvirementSetup/` | 5-attempt model-guided environment setup retry loop, error diagnosis, manifest unpinning, and fallback to the latest version of all modules |
| `Sandbox/` | feasibility gate, Docker image resolver, Kaggle and Docker runners, claim matching, comparison, sandbox report |
| `ui/` | FastAPI server and single-page dashboard over the pipeline |
| `Logger/` | hierarchical run logging (see `Logger/README.md`) |
| `kaggle_jobs/gpu_test/` | a tiny Kaggle kernel for checking GPU access |

Individual stages can still be run on their own, for debugging:

```
python -m Model.gemini <dir containing paper.md>      # extraction only
python -m Sandbox.main <extraction.json> [--local]    # fetch, plan, execute, compare
python -m Sandbox.collect [extraction.json]           # fetch a Kaggle run you stopped waiting for
python -m EnvirementSetup.selftest                    # verify 5x env retry and latest-version fallback
python -m Logger.cli list                             # browse logs; python -m Logger.selftest to verify
```

## Tests

- `python -m Pipeline.selftest` runs offline, with no API keys, Docker, Kaggle or network. PDF
  conversion, Gemini and Kaggle are replaced by fakes; the folder layout, manifest, Logger, report,
  index, resume and the Sandbox orchestration are real. It covers the completed, plan-only, no-repo,
  no-claims and failed-stage paths.
- `python -m EnvirementSetup.selftest` runs offline and verifies version-specifier stripping,
  dependency collection across manifests and AST imports, setup failure detection, recovery when the
  model repairs the environment on an intermediate retry, and automatic fallback to the latest
  versions of all required modules after 5 failed attempts.
- `python -m Logger.selftest` verifies structured run/phase/session logging and secret redaction.

## Known limitations

- **GitHub only.** Code repos and data links must be GitHub URLs to be downloaded. Other data links
  are skipped with a warning (in Kaggle mode the plan can still fetch `http_download` or
  `kaggle_dataset` sources inside the kernel).
- **One repo per paper.** The first code repo that downloads is used.
- **Not pinned to a commit.** The default branch is downloaded as a zip, so the report cannot state
  the exact commit that was run.
- **Old Python stacks need `--local`.** Kaggle cannot change its interpreter, so a paper that needs,
  say, Python 3.6 stops at environment setup in Kaggle mode unless the latest-version fallback can
  run on Kaggle's current Python.
- **Few image shortcuts.** Only two PyTorch combinations map to a ready-made CUDA image (0.x on
  Python 3.6, 2.2/2.3 on Python 3.10); anything else starts from `python:<version>-slim` and installs
  its packages, which is slower. Import-based package detection maps a curated list of common
  libraries, so a repo with no dependency files relies on AST import mapping and the planner's
  judgment.
- **Link extraction is lenient.** Links must come from the paper's text. A link goes to `data_links`
  when the extractor's note mentions data and to `code_repos` otherwise, and duplicates are removed.
- **Free-tier limits.** One Kaggle session is capped at 12 h, local steps over 8 h are skipped, and
  Gemini free-tier quotas apply. Heavy GPU training will often be marked unreproducible.
- **Matching is only as good as the output.** A claim is reproduced only if the run prints a number
  that matches it; a repo that never prints the metric yields `ran_but_mismatch` or `not_attempted`.

## Roadmap

- Cancelling a run from the UI (needs a cancellation hook in `run_paper()`).
- Running more than one paper at a time in the UI.
