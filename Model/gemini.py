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

# Default location for the standalone CLI. Relative to this file (not a machine-specific path)
# and created only when something is written. The integrated pipeline passes its own results_dir.
OUT_DIR = Path(__file__).resolve().parent / "Model_outputs"
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")

client = genai.Client(api_key=API_KEY)

PROMPT = """Extract facts from the scientific paper below.
Use only information present in the text. If something is absent, return an empty list or null.
Only list a URL if it appears in the candidate links. Copy numbers exactly as written; never compute or estimate.

CRITICAL INSTRUCTION: You MUST extract relevant URLs from the CANDIDATE LINKS and explicitly place them into the `code_repos` and `data_links` arrays. Use "code" as the note for repositories and "dataset" for data links.

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

def main(out_dir, results_dir=None, stem=None):
    """Extract claims, code repos and data links from <out_dir>/paper.md.

    Standalone (results_dir=None): extraction.json goes next to paper.md and the stamped
    phase1_<stamp>.json/.md go to Model/Model_outputs, as before.
    Integrated (results_dir given): everything is written into results_dir, named <stem>.*
    (default stem "extraction"), so one run's outputs stay together.
    Returns {"extraction_json", "phase1_json", "phase1_md", "data"}.
    """
    out = Path(out_dir)
    extraction_dir = Path(results_dir) if results_dir else out
    phase1_dir = Path(results_dir) if results_dir else OUT_DIR
    stem = stem or ("extraction" if results_dir else f"phase1_{stamp}")
    extraction_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)

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

    all_links = data.code_repos + data.data_links

    # Keep a set of seen URLs to remove duplicates
    seen_urls = set()
    unique_links = []
    for link in all_links:
        if link.url not in seen_urls:
            unique_links.append(link)
            seen_urls.add(link.url)
        
    # Forgiving filter: route to datas if "data" is in the note, otherwise route to repos
    repos = []
    datas = []
    for l in unique_links:
        item = {**l.model_dump(), "verified": compact(l.url) in flat}
        note_text = (l.note or "").lower()
        
        if "data" in note_text:
            datas.append(item)
        else:
            repos.append(item)

    final = {"main_results": results, "code_repos": repos, "data_links": datas,
             "availability_statement": data.availability_statement}
    extraction_json = extraction_dir / "extraction.json"
    extraction_json.write_text(json.dumps(final, indent=2), encoding="utf-8")

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

    phase1_md = phase1_dir / f"{stem}.md"
    phase1_json = phase1_dir / f"{stem}.json"
    phase1_md.write_text("\n".join(lines), encoding="utf-8")
    if phase1_json.resolve() != extraction_json.resolve():  # integrated runs: same file, write once
        phase1_json.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Saved to {phase1_md}")

    unverified = [r["claim"] for r in results if not r["verified"]]
    get_phase().info("extraction written", results=len(results), results_unverified=len(unverified),
                     repos=len(repos), data_links=len(datas), output=str(phase1_json))
    for claim in unverified:
        get_phase().warning(f"result value not found in paper text: {claim}")
    get_phase().save_artifact("extraction_final.json", final)
    # Lets the Phase 2 run find this run as its upstream.
    link_artifact(phase1_json, "phase1_json")
    link_artifact(phase1_md, "phase1_markdown")
    return {"extraction_json": extraction_json, "phase1_json": phase1_json,
            "phase1_md": phase1_md, "data": final}

if __name__ == "__main__":
    out_dir = sys.argv[1]
    with start_run("phase1_extraction", upstream_artifact=Path(out_dir) / "paper.md",
                   meta={"out_dir": out_dir, "model": MODEL}):
        with phase("phase1_extraction", "LLM extraction of results, code repos and data links"):
            main(out_dir)