"""Gemini-based planning: turn claims + repo contents into a run plan."""
import json

from google.genai import types
from pydantic import BaseModel

from .config import MODEL, client


class RunStep(BaseModel):
    command: str                  # shell command run from repo root
    claim_indices: list[int]      # 1-based ids of the claims this run can confirm; [] = prerequisite
    expected_output_hint: str     # text/number the script prints for these claims
    timeout_min: int = 30


class Plan(BaseModel):
    steps: list[RunStep]          # in execution order (producers before consumers)
    unreproducible: list[str]     # claims that cannot be reproduced, with the reason


def normalize(plan: Plan, n_claims: int) -> Plan:
    """LLMs sometimes number from 0; the claim list is 1-based. Fix that, drop junk ids."""
    all_ids = [i for s in plan.steps for i in s.claim_indices]
    if all_ids and min(all_ids) == 0 and max(all_ids) <= n_claims - 1:
        for s in plan.steps:
            s.claim_indices = [i + 1 for i in s.claim_indices]
    for s in plan.steps:
        s.claim_indices = sorted({i for i in s.claim_indices if 1 <= i <= n_claims})
    return plan


def make_plan(results, repo) -> Plan:
    claims = [
        {"id": i, "claim": r["claim"], "metric": r["metric"], "value": r["value"],
         "system": r["system_or_dataset"]}
        for i, r in enumerate(results, 1)
    ]
    prompt = f"""You plan reproduction runs on one Kaggle notebook (2x Tesla T4, 12h limit,
internet on, no sudo; torch and CUDA are preinstalled). Each claim has a 1-based "id".

Rules:
- Only use scripts that exist in FILE LIST. Do not invent files.
- Pick the script whose HEAD (docstring / printed output) actually prints that claim's metric
  and dataset. A script that merely trains a related model is the WRONG choice.
- If a script reads files produced by another script (e.g. loads a .pkl written earlier),
  put the producing script first as a step with claim_indices [].
- One step may confirm many claims: list all their ids in claim_indices instead of repeating
  the same command.
- If a claim cannot be reproduced from the repo, add it to unreproducible with the reason.

CLAIMS:
{json.dumps(claims, indent=1)}

FILE LIST:
{json.dumps(repo['files'])}

SCRIPT HEADS (first lines of each .py):
{json.dumps(repo['script_heads'], indent=1)[:60000]}

README:
{repo['readme']}
"""
    r = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Plan
        ),
    )
    return normalize(Plan.model_validate_json(r.text), len(results))