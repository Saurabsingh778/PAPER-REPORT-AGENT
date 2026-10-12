"""Stage 4: build the final report (report.md + report.json) for one run.

Always runs, even when an earlier stage failed: the report then says where and why the run
stopped, so every run ends with something to open. Verdicts come from Sandbox.compare (rule
based, never an LLM); this module only assembles and formats them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .manifest import Manifest
from .paths import RunLayout
from .pipeline_stage import ReproductionResult
from .utils import now_iso, read_json, write_json_atomic, write_text_atomic

REPORT_SCHEMA_VERSION = "1"

STATUS_LABELS = {
    "reproduced": "Reproduced",
    "ran_but_mismatch": "Ran, value differs",
    "not_attempted": "Not attempted",
    "timeout": "Timed out",
    "env_failed": "Environment error",
    "data_missing": "Data missing",
    "run_failed": "Run failed",
}


def guess_title(paper_md: Path, fallback: str) -> str:
    """First markdown heading of the converted paper (a display hint, not a verified title)."""
    try:
        for line in paper_md.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^#{1,3}\s+(.+)", line.strip())
            if m:
                title = re.sub(r"[*_`]", "", m.group(1)).strip()
                if len(title) > 8:
                    return title[:200]
    except OSError:
        pass
    return fallback


def _claim_entries(extraction: dict | None, result: ReproductionResult | None,
                   spec: dict | None) -> list[dict[str, Any]]:
    claims = (extraction or {}).get("main_results", [])
    rows = {i: r for i, r in enumerate((result.rows if result else []), 1)}
    planner = {c["claim_id"]: c for c in (spec or {}).get("claims", [])}
    entries = []
    for i, c in enumerate(claims, 1):
        row = rows.get(i, {})
        plan = planner.get(i, {})
        entries.append({
            "index": i,
            "claim": c.get("claim"),
            "metric": c.get("metric"),
            "paper_value": c.get("value"),
            "system_or_dataset": c.get("system_or_dataset"),
            "source": c.get("source"),
            "value_verified_in_paper_text": c.get("verified"),
            "status": row.get("status", "not_attempted"),
            "reproduced_value": row.get("found"),
            "matched_output_line": row.get("matched_line"),
            "match_reasoning": row.get("match_reasoning") or None,
            "planner_status": plan.get("status"),
            "planner_reason": plan.get("reason") or None,
            "planner_steps": plan.get("step_ids", []),
        })
    return entries


def _cell(text: Any, limit: int | None = None) -> str:
    s = "" if text is None else str(text)
    s = " ".join(s.split()).replace("|", "\\|")
    return s if limit is None or len(s) <= limit else s[: limit - 1] + "…"


def _headline(outcome: str, run_status: str, n_ok: int, total: int, failed_stage: str | None,
              error: str | None) -> str:
    if run_status == "failed":
        return f"Run failed at stage '{failed_stage}': {error}"
    if outcome == "plan_only":
        return "Plan only: the reproduction was planned but nothing was executed."
    if outcome == "no_repo":
        return "Not reproducible: no usable code repository."
    if outcome == "no_claims":
        return "Nothing to reproduce: no quantitative results were extracted."
    if outcome == "not_runnable":
        return "Not run: nothing in the plan can run on this machine."
    pct = f" ({round(100 * n_ok / total)}%)" if total else ""
    return f"{n_ok} of {total} claims reproduced{pct}"


def build_report(layout: RunLayout, manifest: Manifest, *, extraction: dict | None,
                 result: ReproductionResult | None, run_status: str,
                 failed_stage: str | None, error: str | None) -> dict[str, Any]:
    """Write report.md and report.json. Returns the report dict (what report.json holds)."""
    spec = result.spec if result and result.spec else read_json(layout.run_spec_json)
    claims = _claim_entries(extraction, result, spec)
    outcome = result.status if result else "failed"
    n_ok = sum(c["status"] == "reproduced" for c in claims)
    by_status: dict[str, int] = {}
    for c in claims:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1

    data = manifest.data
    pdf_name = data["input"]["name"]
    title = data.get("paper_title") or Path(pdf_name).stem
    headline = _headline(outcome, run_status, n_ok, len(claims), failed_stage, error)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated": now_iso(),
        "paper": {"title": title, "pdf": data["input"]["path"], "sha256": data["input"]["sha256"]},
        "run": {"id": layout.run_id, "slug": layout.slug, "status": run_status,
                "reproduction_outcome": outcome, "mode": (result.mode if result else data["options"].get("mode")),
                "reader": data["options"].get("reader"), "created": data["created"],
                "failed_stage": failed_stage, "error": error or (result.error if result else None)},
        "headline": headline,
        "summary": {"claims_total": len(claims), "reproduced": n_ok,
                    "percent_reproduced": round(100 * n_ok / len(claims), 1) if claims else None,
                    "by_status": by_status},
        "repo": ({"url": result.repo_url, "commit": (spec or {}).get("repo", {}).get("commit")}
                 if result and result.repo_url else None),
        "hardware": (spec or {}).get("hardware"),
        "claims": claims,
        "execution_steps": result.steps if result else [],
        "warnings": list((spec or {}).get("warnings", [])),
        "files": {
            "report": "report.md", "manifest": "manifest.json",
            "pdf": layout.rel(layout.input_dir / pdf_name),
            "markdown": layout.rel(layout.paper_md) if layout.paper_md.exists() else None,
            "extraction": layout.rel(layout.extraction_json) if layout.extraction_json.exists() else None,
            "run_spec": layout.rel(layout.run_spec_json) if layout.run_spec_json.exists() else None,
            "comparison_rows": (layout.rel(layout.execution_dir / "comparison_rows.json")
                                if (layout.execution_dir / "comparison_rows.json").exists() else None),
            "phase2_report": (layout.rel(layout.execution_dir / "phase2.md")
                              if (layout.execution_dir / "phase2.md").exists() else None),
            "logs": layout.rel(layout.logs_dir),
        },
    }
    write_json_atomic(layout.report_json, report)
    write_text_atomic(layout.report_md, _render_markdown(report, claims, result))
    return report


def _render_markdown(report: dict[str, Any], claims: list[dict], result: ReproductionResult | None) -> str:
    run, paper = report["run"], report["paper"]
    L: list[str] = [f"# Reproducibility report: {paper['title']}", "", f"**{report['headline']}**", ""]

    L += ["| | |", "|---|---|",
          f"| Paper | `{Path(paper['pdf']).name}` |",
          f"| Run | `{run['id']}` ({run['created']}) |",
          f"| Mode | {run['mode'] or '-'} (PDF reader: {run['reader'] or '-'}) |"]
    if report["repo"]:
        commit = (report["repo"]["commit"] or "")[:10]
        L.append(f"| Code | {report['repo']['url']}" + (f" @ `{commit}`" if commit else "") + " |")
    hw = report["hardware"]
    if hw:
        L.append(f"| Hardware | GPU {hw.get('gpu')}: {_cell(hw.get('reason'), 160)} |")
    L.append("")

    if claims:
        L += ["## Claims", "", "| # | Claim | Metric | Paper | Reproduced | Status |", "|---|---|---|---|---|---|"]
        for c in claims:
            found = "-" if c["reproduced_value"] is None else c["reproduced_value"]
            L.append(f"| {c['index']} | {_cell(c['claim'], 140)} | {_cell(c['metric'])} | "
                     f"{_cell(c['paper_value'])} | {_cell(found)} | {STATUS_LABELS.get(c['status'], c['status'])} |")
        L.append("")

        L.append("## Details")
        for c in claims:
            L += ["", f"### {c['index']}. {_cell(c['claim'])}",
                  f"- Paper value: {_cell(c['paper_value'])} ({_cell(c['metric'])}, {_cell(c['source'])})"
                  + ("" if c["value_verified_in_paper_text"] is not False
                     else " - **value not found verbatim in the paper text, check it**"),
                  f"- Dataset/system: {_cell(c['system_or_dataset'])}",
                  f"- Result: **{STATUS_LABELS.get(c['status'], c['status'])}**"]
            if c["matched_output_line"]:
                L.append(f"- Output line: `{_cell(c['matched_output_line'], 200)}`")
            if c["match_reasoning"]:
                L.append(f"- Matching note: {_cell(c['match_reasoning'])}")
            if c["planner_status"] and c["planner_status"] != "mapped":
                L.append(f"- Planner: {c['planner_status']}: {_cell(c['planner_reason'])}")
            elif c["planner_steps"]:
                L.append(f"- Planned steps: {', '.join(c['planner_steps'])}")
        L.append("")

    if result and result.steps:
        L += ["## Execution steps", "", "| Step | Command | Exit code | Seconds |", "|---|---|---|---|"]
        for s in result.steps:
            L.append(f"| {_cell(s['id'])} | `{_cell(s['command'], 120)}` | {s['rc']} | {s['seconds']} |")
        L.append("")

    notes = [r for r in (result.unreproducible if result else []) if r]
    if notes:
        L += ["## Not reproducible / not attempted", ""] + [f"- {_cell(n)}" for n in notes] + [""]

    err = run["error"]
    if err and run["status"] != "failed":
        L += ["## Notes", "", f"- {_cell(err)}", ""]
    if report["warnings"]:
        L += ["## Warnings", ""] + [f"- {_cell(w)}" for w in report["warnings"]] + [""]

    L += ["## Files", ""]
    for label, key in (("PDF (as uploaded)", "pdf"), ("Paper markdown", "markdown"),
                       ("Extraction", "extraction"), ("Run plan", "run_spec"),
                       ("Comparison rows", "comparison_rows"), ("Sandbox report", "phase2_report"),
                       ("Run status", "manifest"), ("Logs", "logs")):
        if report["files"].get(key):
            path = report["files"][key]
            L.append(f"- {label}: [{path}]({quote(path)})")  # quoted: file names may contain spaces
    L.append("- Machine-readable version of this report: [report.json](report.json)")
    return "\n".join(L) + "\n"
