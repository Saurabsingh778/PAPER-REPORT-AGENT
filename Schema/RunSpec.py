"""RunSpec: the contract between RepoHandler (decides what to run) and the Kaggle runner
(executes it, no decisions of its own).

Design rules
  - Everything the runner needs is in the spec. Nothing is inferred at run time.
  - Paths are relative, or use the two tokens {REPO} and {DATA}. Absolute paths and README
    placeholders (/path/to/x, D:\\x) are rejected, so the first Kaggle run never dies on one.
  - Every step cites evidence for why it was chosen, and says how confident the handler is.
  - Every claim is accounted for: mapped to steps, or explicitly unreproducible/unclear
    with a reason. Nothing silently disappears.
  - Every code change is an explicit, reviewable PathRewrite. Model and training logic are
    never edited.

`validate_spec` rules live in the model validators (self-consistency, no repo needed).
`check_against_repo` is the local pre-flight (needs the cloned repo, still no execution).
"""
from __future__ import annotations

import posixpath
import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1"
MAX_SESSION_MIN = 720  # one Kaggle session
CONF_RANK = {"low": 0, "medium": 1, "high": 2}
Confidence = Literal["low", "medium", "high"]

ALLOWED_TOKENS = {"{REPO}", "{DATA}"}
TOKEN_RE = re.compile(r"\{[A-Z_]+\}")
PLACEHOLDER_RE = re.compile(r"/path/to|<[A-Za-z_ -]+>|(?<![\w])[A-Za-z]:\\|your[_-]|\bX{3,}\b", re.I)
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# ----------------------------------------------------------------------------- evidence
class Evidence(BaseModel):
    """Why the handler believes something. Short quote, never a paragraph."""

    kind: Literal[
        "readme",        # README documents this command / number
        "docstring",     # script docstring or header says so
        "number_match",  # the paper's number literally appears in this file
        "print_line",    # the script prints this metric on this line
        "argparse",      # flag / default found in the code
        "ast",           # found by static analysis (reads/writes, device use, imports)
        "prior_run",     # observed in an earlier execution
        "llm_judgment",  # model's call, no hard evidence: keep confidence low
    ]
    file: str | None = None
    line: int | None = None
    text: str = Field(max_length=300)


# ----------------------------------------------------------------------------- repo + data
class RepoRef(BaseModel):
    url: str
    commit: str = Field(min_length=7, description="full sha of the code that was analysed")
    subdir: str = Field("", description="repo-relative folder the code lives in, if not the root")


class DataSource(BaseModel):
    """Data a step needs that is not code. Anything not in the repo must be fetchable."""

    id: str
    kind: Literal["in_repo", "github_repo", "kaggle_dataset", "http_download", "unavailable"]
    ref: str = Field(description="in_repo: repo path | github_repo/http_download: URL | "
                                 "kaggle_dataset: owner/slug | unavailable: what it is")
    dest: str | None = Field(None, description="folder name under {DATA}; not used for in_repo")
    size_mb: float | None = None
    note: str = ""

    @field_validator("id")
    @classmethod
    def _id(cls, v):
        if not ID_RE.match(v):
            raise ValueError(f"data id '{v}' must match {ID_RE.pattern}")
        return v

    @model_validator(mode="after")
    def _kind_rules(self):
        k, r = self.kind, self.ref
        if k == "in_repo" and (r.startswith("/") or ".." in r.split("/")):
            raise ValueError(f"data '{self.id}': in_repo ref must be a relative path inside the repo")
        if k == "github_repo" and not r.startswith("https://github.com/"):
            raise ValueError(f"data '{self.id}': github_repo ref must be a https://github.com/ URL")
        if k == "http_download" and not r.startswith("https://"):
            raise ValueError(f"data '{self.id}': http_download ref must be an https URL")
        if k == "kaggle_dataset" and not re.match(r"^[\w.-]+/[\w.-]+$", r):
            raise ValueError(f"data '{self.id}': kaggle_dataset ref must be owner/slug")
        if k in ("github_repo", "http_download", "kaggle_dataset") and not self.dest:
            raise ValueError(f"data '{self.id}': {k} needs a dest folder name")
        return self


# ----------------------------------------------------------------------------- environment / hardware
class Environment(BaseModel):
    """Python packages on top of what Kaggle ships. Exact pins from old requirements files
    usually don't install on a current Python, so the default is to ignore them."""

    pip: list[str] = Field(default_factory=list, description="names/specs installed one by one")
    requirements_file: str | None = None
    requirements_policy: Literal["ignore", "unpinned", "as_is"] = "ignore"
    system_packages: list[str] = Field(default_factory=list)
    notes: str = ""


class Hardware(BaseModel):
    gpu: Literal["required", "helpful", "none"]
    reason: str
    est_runtime_min: int | None = Field(None, description="None = unknown; never a guess")
    min_vram_gb: float | None = None
    evidence: list[Evidence] = Field(default_factory=list)


# ----------------------------------------------------------------------------- steps
class PathRewrite(BaseModel):
    """The only kind of code change the runner may apply: swap a hardcoded path string."""

    file: str = Field(description="repo-relative file")
    old: str
    new: str
    reason: str

    @model_validator(mode="after")
    def _tokens(self):
        for t in TOKEN_RE.findall(self.new):
            if t not in ALLOWED_TOKENS:
                raise ValueError(f"rewrite for {self.file}: unknown token {t}")
        if self.old == self.new:
            raise ValueError(f"rewrite for {self.file}: old and new are identical")
        return self


class Signal(BaseModel):
    """Where in a step's output a claim's number should appear. The comparer only looks at
    lines matching line_pattern, so another metric's number can't be mistaken for it."""

    claim_id: int
    metric: str
    line_pattern: str = Field(description="regex matched against single output lines")
    unit: Literal["percent", "fraction", "raw"] | None = None

    @field_validator("line_pattern")
    @classmethod
    def _regex(cls, v):
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"line_pattern is not a valid regex: {e}")
        return v


class Step(BaseModel):
    id: str
    command: str
    cwd: str = "{REPO}"
    needs: list[str] = Field(default_factory=list, description="step ids that must succeed first")
    reads: list[str] = Field(default_factory=list,
                             description="'data:<id>' or a path relative to cwd")
    writes: list[str] = Field(default_factory=list, description="paths relative to cwd")
    covers: list[int] = Field(default_factory=list, description="claim ids; [] = prerequisite step")
    signals: list[Signal] = Field(default_factory=list)
    gpu: bool
    timeout_min: int = Field(ge=1, le=MAX_SESSION_MIN)
    level: Literal["full_rerun", "from_cached_artifacts"] = "full_rerun"
    cached_inputs: list[str] = Field(
        default_factory=list,
        description="committed intermediate files used instead of regenerating them")
    patches: list[PathRewrite] = Field(default_factory=list)
    confidence: Confidence
    evidence: list[Evidence] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _id(cls, v):
        if not ID_RE.match(v):
            raise ValueError(f"step id '{v}' must match {ID_RE.pattern}")
        return v

    @field_validator("cwd")
    @classmethod
    def _cwd(cls, v):
        if not (v == "{REPO}" or v.startswith("{REPO}/")):
            raise ValueError(f"cwd '{v}' must be {{REPO}} or {{REPO}}/<subfolder>")
        return v

    @field_validator("command")
    @classmethod
    def _command(cls, v):
        if not v.strip():
            raise ValueError("empty command")
        if PLACEHOLDER_RE.search(v):
            raise ValueError(f"placeholder path in command (resolve it to {{REPO}}/{{DATA}}): {v!r}")
        for t in TOKEN_RE.findall(v):
            if t not in ALLOWED_TOKENS:
                raise ValueError(f"unknown token {t} in command: {v!r}")
        try:
            toks = shlex.split(v)
        except ValueError as e:
            raise ValueError(f"command does not parse: {e}")
        for t in toks:
            bare = t.split("=", 1)[-1]
            if bare.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", bare):
                raise ValueError(f"absolute path in command, use {{REPO}}/{{DATA}}: {t!r}")
        return v

    @model_validator(mode="after")
    def _consistency(self):
        if self.cached_inputs and self.level != "from_cached_artifacts":
            raise ValueError(f"step '{self.id}': cached_inputs set but level is {self.level}")
        if self.level == "from_cached_artifacts" and not self.cached_inputs:
            raise ValueError(f"step '{self.id}': level is from_cached_artifacts but cached_inputs is empty")
        sig_ids = {s.claim_id for s in self.signals}
        if not sig_ids <= set(self.covers):
            raise ValueError(f"step '{self.id}': signals for claims {sorted(sig_ids - set(self.covers))} "
                             f"that the step does not cover")
        missing = set(self.covers) - sig_ids
        if missing:
            raise ValueError(f"step '{self.id}': covers claims {sorted(missing)} but has no signal "
                             f"saying where they appear in the output")
        return self


# ----------------------------------------------------------------------------- claims
class ClaimMap(BaseModel):
    claim_id: int
    status: Literal["mapped", "unreproducible", "unclear"]
    step_ids: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def _rules(self):
        if self.status == "mapped" and not self.step_ids:
            raise ValueError(f"claim {self.claim_id}: mapped but no steps")
        if self.status != "mapped":
            if self.step_ids:
                raise ValueError(f"claim {self.claim_id}: {self.status} must not list steps")
            if not self.reason.strip():
                raise ValueError(f"claim {self.claim_id}: {self.status} needs a reason")
        return self


# ----------------------------------------------------------------------------- the spec
class RunSpec(BaseModel):
    schema_version: Literal["1"] = SCHEMA_VERSION
    repo: RepoRef
    decided_by: Literal["manifest", "readme", "triage"] = Field(
        description="manifest: repo ships reproduce.yaml | readme: documented commands, checked "
                    "against the code | triage: model chose entry points from the code")
    data: list[DataSource] = Field(default_factory=list)
    environment: Environment = Field(default_factory=Environment)
    hardware: Hardware
    steps: list[Step]
    claims: list[ClaimMap]
    confidence: Confidence
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _whole_spec(self):
        errs: list[str] = []

        step_ids = [s.id for s in self.steps]
        data_ids = [d.id for d in self.data]
        claim_ids = [c.claim_id for c in self.claims]
        for what, ids in (("step", step_ids), ("data", data_ids), ("claim", claim_ids)):
            dup = {i for i in ids if ids.count(i) > 1}
            if dup:
                errs.append(f"duplicate {what} ids: {sorted(dup)}")
        if errs:
            raise ValueError("; ".join(errs))

        # needs: exist, and point backwards (list order is execution order -> no cycles)
        pos = {sid: i for i, sid in enumerate(step_ids)}
        for s in self.steps:
            for n in s.needs:
                if n not in pos:
                    errs.append(f"step '{s.id}' needs unknown step '{n}'")
                elif pos[n] >= pos[s.id]:
                    errs.append(f"step '{s.id}' needs '{n}' which does not run before it")

        # data references
        data = {d.id: d for d in self.data}
        for s in self.steps:
            for r in s.reads:
                if r.startswith("data:"):
                    d = data.get(r[5:])
                    if d is None:
                        errs.append(f"step '{s.id}' reads unknown data '{r[5:]}'")
                    elif d.kind == "unavailable":
                        errs.append(f"step '{s.id}' reads data '{d.id}' which is unavailable; "
                                    f"its claims must be marked unreproducible instead")

        # claims <-> steps, both directions
        steps = {s.id: s for s in self.steps}
        for c in self.claims:
            for sid in c.step_ids:
                if sid not in steps:
                    errs.append(f"claim {c.claim_id} points at unknown step '{sid}'")
                elif c.claim_id not in steps[sid].covers:
                    errs.append(f"claim {c.claim_id} lists step '{sid}' but that step does not cover it")
        by_claim = {c.claim_id: c for c in self.claims}
        for s in self.steps:
            for cid in s.covers:
                c = by_claim.get(cid)
                if c is None:
                    errs.append(f"step '{s.id}' covers unknown claim {cid}")
                elif s.id not in c.step_ids:
                    errs.append(f"step '{s.id}' covers claim {cid} but the claim does not list the step")

        # hardware
        any_gpu = any(s.gpu for s in self.steps)
        if self.hardware.gpu == "none" and any_gpu:
            errs.append("hardware.gpu is 'none' but a step is marked gpu")
        if self.hardware.gpu == "required" and not any_gpu:
            errs.append("hardware.gpu is 'required' but no step is marked gpu")

        # one session must fit everything
        total = sum(s.timeout_min for s in self.steps)
        if total > MAX_SESSION_MIN:
            errs.append(f"timeouts add up to {total} min, more than one {MAX_SESSION_MIN}-min session; "
                        f"split into several specs or drop steps")

        # honest confidence: the spec is never more sure than its least sure step.
        # Derived, not a reason to reject: the planner kept tripping on this and each
        # rejection cost a full LLM retry, so the value is simply lowered to the floor.
        if self.steps:
            floor_conf = min((s.confidence for s in self.steps), key=CONF_RANK.__getitem__)
            if CONF_RANK[self.confidence] > CONF_RANK[floor_conf]:
                self.confidence = floor_conf
        if self.decided_by == "triage" and any(
            all(e.kind == "llm_judgment" for e in s.evidence) and s.confidence == "high"
            for s in self.steps
        ):
            errs.append("a step backed only by llm_judgment cannot be high confidence")

        if errs:
            raise ValueError("; ".join(errs))
        return self

    # convenience for the executor / report
    def mapped_claim_ids(self) -> list[int]:
        return [c.claim_id for c in self.claims if c.status == "mapped"]


# ----------------------------------------------------------------------------- data layout
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_data_folder(data_root, *names: str):
    """Locate, under a downloaded-data root, the top-level folder a data entry refers to.

    Downloaded repos land in <data_root>/<owner>_<repo>, while a spec names them by `dest`,
    by `id`, or by the first segment of an in_repo `ref`. Match exact first, then by
    normalised substring (either direction). Returns a Path or None.
    """
    root = Path(data_root)
    if not root.is_dir():
        return None
    folders = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    wanted = [n for n in names if n]
    for n in wanted:                      # exact
        for f in folders:
            if f.name == n:
                return f
    for n in wanted:                      # fuzzy
        nn_ = _norm(n)
        if len(nn_) < 3:
            continue
        for f in folders:
            fn = _norm(f.name)
            if nn_ in fn or fn in nn_:
                return f
    return None


# ----------------------------------------------------------------------------- pre-flight
def _check_flag_values(step_id: str, toks: list[str], script: str, script_path: Path) -> list[str]:
    """`--experiments B_ext8d`: a name-like value (contains '_') that never appears in the script
    is almost certainly invented by the planner. Paths, numbers and plain words are ignored."""
    try:
        src = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[str] = []
    for i, t in enumerate(toks[:-1]):
        if not (t.startswith("--") and "=" not in t):
            continue
        val = toks[i + 1]
        if val.startswith("-") or "/" in val or "\\" in val:
            continue
        for part in val.split(","):
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", part) and part not in src:
                out.append(f"step '{step_id}': value '{part}' for {t} does not appear in {script}; "
                           f"the script has no such option/name")
    return out


def check_against_repo(spec: RunSpec, repo_root, data_root=None) -> list[str]:
    """Check the spec against the cloned repo. Reads files, never executes anything.
    Returns a list of problems; empty means the spec is runnable as far as can be told locally.
    `data_root`: folder where github_repo/http_download data has been fetched, if available."""
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root).resolve() if data_root else None
    problems: list[str] = []

    def sub(s: str) -> str:
        s = s.replace("{REPO}", str(repo_root))
        return s.replace("{DATA}", str(data_root)) if data_root else s

    data = {d.id: d for d in spec.data}
    for d in spec.data:
        if d.kind == "in_repo":
            # in_repo data may live in the code repo OR in a data repo downloaded under {DATA}
            # (the planner is shown both trees), so it only counts as missing if neither has it.
            in_code = (repo_root / d.ref).exists()
            in_data = bool(data_root and (data_root / d.ref).exists())
            if not in_code and not in_data:
                where = "the repo or the data folder" if data_root else "the repo"
                problems.append(f"data '{d.id}': {d.ref} is not in {where}")
        elif data_root and d.kind in ("github_repo", "http_download"):
            if not (d.dest and (data_root / d.dest).exists()) and not find_data_folder(data_root, d.dest, d.id):
                problems.append(f"data '{d.id}': {d.dest} not found under the data folder")

    produced: set[str] = set()
    for s in spec.steps:
        cwd = Path(sub(s.cwd))
        if not cwd.is_dir():
            problems.append(f"step '{s.id}': cwd {s.cwd} does not exist in the repo")
            continue
        cwd_rel = s.cwd.removeprefix("{REPO}").lstrip("/")

        toks = shlex.split(sub(s.command))
        script = next((t for t in toks if t.endswith(".py")), None)
        if script:
            p = Path(script)
            script_path = p if p.is_absolute() else cwd / p
            if not script_path.exists():
                problems.append(f"step '{s.id}': script {script} not found (cwd {s.cwd})")
            else:
                problems += _check_flag_values(s.id, toks, script, script_path)
        if data_root:
            for t in toks:
                bare = t.split("=", 1)[-1]
                if bare.startswith(str(data_root)) and not Path(bare).exists():
                    problems.append(f"step '{s.id}': path {bare.replace(str(data_root), '{DATA}')} does not exist")

        for r in s.reads:
            if r.startswith("data:"):
                continue  # existence of data ids is checked above
            rel = posixpath.normpath(posixpath.join(cwd_rel, r))
            if not (repo_root / rel).exists() and rel not in produced:
                problems.append(f"step '{s.id}': reads {r} which is neither in the repo nor written by an earlier step")
        for c in s.cached_inputs:
            rel = posixpath.normpath(posixpath.join(cwd_rel, c))
            if not (repo_root / rel).exists():
                problems.append(f"step '{s.id}': cached input {c} is not committed in the repo")

        for pr in s.patches:
            f = repo_root / pr.file
            if not f.is_file():
                problems.append(f"step '{s.id}': patch target {pr.file} not found")
            elif pr.old not in f.read_text(encoding="utf-8", errors="replace"):
                problems.append(f"step '{s.id}': patch string not found in {pr.file}: {pr.old[:70]!r}")

        for w in s.writes:
            produced.add(posixpath.normpath(posixpath.join(cwd_rel, w)))
    return problems