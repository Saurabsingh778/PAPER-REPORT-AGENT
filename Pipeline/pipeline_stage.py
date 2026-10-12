"""Stage 3: extraction.json -> fetch repo, plan (RunSpec), execute, compare.

Thin wrapper over Sandbox.main.run_pipeline: it points the sandbox at this run's folders and
turns what comes back into plain JSON-safe data the report stage can use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import RunLayout


@dataclass
class ReproductionResult:
    status: str                      # completed | plan_only | no_claims | no_repo | not_runnable
    mode: str                        # kaggle | local_docker
    repo_url: str | None = None
    rows: list[dict] = field(default_factory=list)     # one per paper claim (Sandbox.compare rows)
    n_ok: int = 0
    spec: dict | None = None         # the RunSpec as a plain dict
    steps: list[dict] = field(default_factory=list)    # what actually ran: id, command, rc, seconds
    unreproducible: list[str] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ReproductionResult":
        spec = raw.get("spec")
        run_out = raw.get("run_out") or {}
        steps = [{"id": s.get("step_id"), "command": s.get("cmd"), "rc": s.get("rc"),
                  "seconds": s.get("sec")} for s in run_out.get("steps", [])]
        return cls(
            status=raw["status"], mode=raw["mode"], repo_url=raw.get("repo_url"),
            rows=raw.get("rows") or [], n_ok=raw.get("n_ok") or 0,
            spec=spec.model_dump() if spec is not None else None,
            steps=steps, unreproducible=raw.get("unreproducible") or [], error=raw.get("error"),
        )


def run_reproduction(extraction_json: Path, layout: RunLayout, *, use_local: bool,
                     plan_only: bool = False) -> ReproductionResult:
    from Sandbox.main import run_pipeline  # lazy: pulls in the Gemini client, Kaggle/Docker runners

    raw = run_pipeline(
        str(extraction_json),
        use_local=use_local,
        out_dir=layout.execution_dir,
        plan_dir=layout.plan_dir,
        repo_root=layout.repo_cache,
        data_root=layout.data_cache,
        plan_only=plan_only,
    )
    return ReproductionResult.from_raw(raw)
