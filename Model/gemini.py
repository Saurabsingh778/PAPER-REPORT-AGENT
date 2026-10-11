from google import genai
import os
from dotenv import load_dotenv
from Schema.phase_1_schema import Result, Extraction, Link
import re
import sys
import json
from pathlib import Path
from datetime import datetime

from Logger import get_phase, link_artifact, logged, phase, start_run

load_dotenv()

OUT_DIR = Path("/home/saurab/research-paper-agent/Model/Model_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")

client = genai.Client(api_key=API_KEY)

PROMPT = """Extract facts from the scientific paper below.
Use only information present in the text. If something is absent, return an empty list or null.
Only list a URL if it appears in the candidate links. Copy numbers exactly as written; never compute or estimate.

CANDIDATE LINKS (found by regex, with context):
{candidates}

PAPER:
{paper}
"""

URL_RE = re.compile(r'https?://[^\s)\]>"\'<]+')

def candidates(text):
    seen, out = set(), []
    for m in URL_RE.finditer(text):
        url = m.group(0).rstrip('.,;:')
        if url in seen:
            continue
        seen.add(url)
        ctx = text[max(0, m.start() - 120): m.end() + 60].replace("\n", " ")
        out.append(f"- {url}\n  context: ...{ctx}...")
    return out

@logged(
    "gemini_extract",
    summarize=lambda e: f"{len(e.main_results)} results, {len(e.code_repos)} repos, {len(e.data_links)} data links",
)
def extract(prompt) -> Extraction:
    try:
        it = client.interactions.create(
            model=MODEL,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": Extraction.model_json_schema(),
            },
        )
        # Raw model reply is stored before validation so a schema failure stays debuggable.
        get_phase().save_artifact("extract_response_structured.json", it.output_text)
        return Extraction.model_validate_json(it.output_text)
    except Exception as e:
        # Fallback for models that don't support response_format (e.g. Gemma): ask for JSON in the prompt
        print(f"structured output failed ({type(e).__name__}: {str(e)[:120]}), falling back to plain JSON")
        get_phase().warning(f"structured output failed, falling back to plain JSON: {type(e).__name__}: {e}")
        schema = json.dumps(Extraction.model_json_schema())
        txt = client.interactions.create(
            model=MODEL,
            input=prompt + f"\n\nReturn ONLY JSON matching this schema, no commentary:\n{schema}",
        ).output_text.strip()
        get_phase().save_artifact("extract_response_fallback.txt", txt)
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
        return Extraction.model_validate_json(txt)

def compact(s):
    return re.sub(r"\s+", "", s)

def main(out_dir):
    out = Path(out_dir)
    text = (out / "paper.md").read_text(encoding="utf-8")
    cands = candidates(text)
    print(f"{len(cands)} candidate links found by regex")
    get_phase().info("paper loaded", paper_chars=len(text), candidate_links=len(cands), model=MODEL)
    get_phase().save_artifact("candidate_links.txt", "\n".join(cands) or "(none)")

    data = extract(PROMPT.format(candidates="\n".join(cands) or "(none)", paper=text))

    flat = compact(text)
    results = []
    for r in data.main_results:
        d = r.model_dump(); d["verified"] = compact(r.value) in flat; results.append(d)
    repos = [{**l.model_dump(), "verified": compact(l.url) in flat} for l in data.code_repos]
    datas = [{**l.model_dump(), "verified": compact(l.url) in flat} for l in data.data_links]

    final = {"main_results": results, "code_repos": repos, "data_links": datas,
             "availability_statement": data.availability_statement}
    (out / "extraction.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    print("\nMAIN RESULTS")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{'ok' if r['verified'] else 'CHECK'}] {r['claim']} | {r['metric']} = {r['value']} ({r['source']})")
    print("\nCODE REPOS")
    for r in repos:
        print(f"- [{'ok' if r['verified'] else 'CHECK'}] {r['url']}  {r['note']}")
    print("\nDATA LINKS")
    for r in datas:
        print(f"- [{'ok' if r['verified'] else 'CHECK'}] {r['url']}  {r['note']}")
    print("\nAVAILABILITY STATEMENT:", final["availability_statement"])

    lines = ["# Phase 1: Extraction results\n", "## Main results\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{'ok' if r['verified'] else 'CHECK'}] {r['claim']} | {r['metric']} = {r['value']} ({r['source']})")
    lines += ["\n## Code repositories\n"] + [f"- [{'ok' if r['verified'] else 'CHECK'}] {r['url']} ({r['note']})" for r in repos]
    lines += ["\n## Data links\n"] + [f"- [{'ok' if r['verified'] else 'CHECK'}] {r['url']} ({r['note']})" for r in datas]
    lines += ["\n## Availability statement\n", final["availability_statement"] or "none"]

    (OUT_DIR / f"phase1_{stamp}.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / f"phase1_{stamp}.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Saved to {OUT_DIR}/phase1_{stamp}.md")

    unverified = [r["claim"] for r in results if not r["verified"]]
    get_phase().info("extraction written", results=len(results), results_unverified=len(unverified),
                     repos=len(repos), data_links=len(datas), output=str(OUT_DIR / f"phase1_{stamp}.json"))
    for claim in unverified:
        get_phase().warning(f"result value not found in paper text: {claim}")
    get_phase().save_artifact("extraction_final.json", final)
    # Lets the Phase 2 run find this run as its upstream.
    link_artifact(OUT_DIR / f"phase1_{stamp}.json", "phase1_json")
    link_artifact(OUT_DIR / f"phase1_{stamp}.md", "phase1_markdown")

if __name__ == "__main__":
    out_dir = sys.argv[1]
    with start_run("phase1_extraction", upstream_artifact=Path(out_dir) / "paper.md",
                   meta={"out_dir": out_dir, "model": MODEL}):
        with phase("phase1_extraction", "LLM extraction of results, code repos and data links"):
            main(out_dir)