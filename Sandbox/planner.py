"""Gemini-based planning: turn claims + repo contents into a run plan."""
import json

from google.genai import types
from pydantic import BaseModel

from .config import MODEL, client


class RunStep(BaseModel):
    claim_index: int           # 1-based index into main_results
    command: str               # shell command run from repo root
    expected_output_hint: str  # what to look for in stdout/files
    timeout_min: int = 30


class Plan(BaseModel):
    setup_commands: list[str]
    steps: list[RunStep]
    unreproducible: list[str]  # claims needing data/compute we can't get


def make_plan(results, repo) -> Plan:
    prompt = f"""You plan reproduction runs on one Kaggle notebook (2x Tesla T4, 12h limit,
internet on, no sudo). Only use scripts that exist in FILE LIST. Do not invent files.
If a claim cannot be reproduced from the repo, list it in unreproducible.

CLAIMS:
{json.dumps(results, indent=1)}

FILE LIST:
{json.dumps(repo['files'])}

README:
{repo['readme']}

DEPS:
{json.dumps(repo['deps'])[:4000]}
"""
    r = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Plan
        ),
    )
    return Plan.model_validate_json(r.text)