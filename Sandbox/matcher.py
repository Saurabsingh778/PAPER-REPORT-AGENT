"""LLM matching call: maps each paper claim to the line of run output that measures it."""
import json
from pathlib import Path
from typing import List, Optional

from Schema.phase_1_schema import BaseModel
try:
    from google.genai import types
except ImportError:
    types = None

from Model.gemini import client, MODEL
from Sandbox.compare import NUM, _match, first_number

MAX_OUTPUT_CHARS = 60000


class ClaimMatch(BaseModel):
    claim_index: int          # 1-based, as numbered in the prompt
    found: bool
    line: str = ""            # verbatim output line holding the measured value
    value: Optional[float] = None
    reasoning: str = ""


class MatchResponse(BaseModel):
    matches: List[ClaimMatch]


def collect_output(run_out, raw_dir: Optional[Path] = None) -> str:
    """All step output, preferring the untruncated raw logs over results.json tails."""
    parts = []
    for s in (run_out or {}).get("steps", []):
        text = None
        if raw_dir:
            f = Path(raw_dir) / f"step_{s.get('step_id')}.log"
            if f.exists():
                text = f.read_text(encoding="utf-8", errors="replace")
        if text is None:
            text = f"--- STDOUT ---\n{s.get('stdout', '')}\n--- STDERR ---\n{s.get('stderr', '')}"
        parts.append(f"===== step {s.get('step_id')} (rc={s.get('rc')}) =====\n{text}")
    out = "\n".join(parts)
    if len(out) > MAX_OUTPUT_CHARS:
        out = "[...head truncated...]\n" + out[-MAX_OUTPUT_CHARS:]
    return out


def _prompt(results, output):
    claims = "\n".join(
        f"{i}. {r['claim']} | metric: {r['metric']} | paper value: {r['value']}"
        for i, r in enumerate(results, 1)
    )
    return f"""You match claims from a research paper to the output of a reproduction run.

CLAIMS:
{claims}

RUN OUTPUT:
{output}

For EVERY claim, find the output line that reports the SAME quantity (same metric, same model/
condition/dataset, same aggregation, e.g. a cross-validated mean in a summary table, not one
fold or one epoch). Rules:
- "line" must be copied VERBATIM from the output, and "value" must be the number in that line
  that corresponds to the claim's central value (for "a ± b" claims, the mean).
- Do not accept a number just because it is numerically close; the quantity must match.
- If the output does not report that quantity, set found=false (empty line, null value).
- Give a one-sentence reasoning. Return one entry per claim, using the claim numbers above."""


def llm_match(results, output) -> List[ClaimMatch]:
    resp = client.models.generate_content(
        model=MODEL,
        contents=_prompt(results, output),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MatchResponse,
            temperature=0.0,
        ),
    )
    return MatchResponse.model_validate_json(resp.text or "").matches


def verify(m: ClaimMatch, paper, output: str, tol: float):
    """Return (status, value, line) after checking the LLM's evidence against the real output."""
    if not m.found or m.value is None or not m.line.strip():
        return "ran_but_mismatch", None, None
    line = m.line.strip()
    if line not in output:
        norm = " ".join(line.split())
        real = next((l.strip() for l in output.splitlines() if " ".join(l.split()) == norm), None)
        if real is None:
            return "ran_but_mismatch", None, None
        line = real
    if not any(abs(float(x) - m.value) < 1e-9 for x in NUM.findall(line)):
        return "ran_but_mismatch", None, None
    if paper is not None and _match(m.value, paper, tol):
        return "reproduced", m.value, line
    return "ran_but_mismatch", m.value, line