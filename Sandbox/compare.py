"""Rule-based comparison of paper values vs. Kaggle run output.

Deliberately no LLM here, so the reproducibility verdict can't be hallucinated.
Failures are classified so an environment/data problem in the agent is never reported
as if the paper's number had failed to reproduce.
"""
import re

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# LaTeX like "8.6 \times 10^{-4}" -> "8.6e-4"
SCI = re.compile(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?\s*(-?\d+)\s*\}?")


def first_number(s):
    """First number in a paper value string, handling LaTeX scientific notation."""
    s = SCI.sub(lambda m: f"{m.group(1)}e{m.group(2)}", s or "")
    m = NUM.search(s)
    return float(m.group(0)) if m else None


def _match(n, paper, tol):
    # sign-insensitive (a run may print |rho|); percent vs fraction tolerated (95.1 vs 0.951)
    a, p = abs(n), abs(paper)
    return abs(a - p) <= tol * p or abs(a * 100 - p) <= tol * p


def classify_failure(step):
    """Why a step with rc != 0 failed: agent-side (env/data) vs. the code itself."""
    err = step.get("stderr", "")
    if "TIMEOUT" in err:
        return "timeout"
    if "ModuleNotFoundError" in err or "ImportError" in err:
        return "env_failed"
    if "FileNotFoundError" in err and "/kaggle/input" in err:
        return "data_missing"
    return "run_failed"


DEFAULT_TOL = 0.03  # a result within 3% (relative) of the paper's value counts as reproduced


def _find_match(stdout, paper, tol):
    """First number in the output within tol of the paper value, with the line it came from."""
    for line in (stdout or "").splitlines():
        for x in NUM.findall(line):
            if _match(float(x), paper, tol):
                return float(x), line.strip()
    return None


def compare(results, run_out, tol=DEFAULT_TOL, matches=None, output=""):
    """Return (rows, n_reproduced). Each row records the output line that matched.

    `matches` (optional, {claim_index: ClaimMatch} from Sandbox.matcher) switches claim
    matching from the number-in-stdout heuristic to the LLM-located, code-verified evidence.
    The heuristic stays as the fallback when no LLM matches are supplied."""
    if matches is not None:
        from Sandbox.matcher import verify

    by_claim = {}
    for s in (run_out or {}).get("steps", []):
        for i in s.get("claim_indices", []):
            by_claim.setdefault(i, []).append(s)

    rows = []
    for i, r in enumerate(results, 1):
        paper = first_number(r["value"])
        status, found, matched_line = "not_attempted", None, None
        if matches is not None:
            steps = (run_out or {}).get("steps", [])
            if steps and all(st["rc"] != 0 for st in steps):
                status = classify_failure(steps[0])
            elif steps and i in matches:
                status, found, matched_line = verify(matches[i], paper, output, tol)
            rows.append({"claim": r["claim"], "metric": r["metric"], "paper": r["value"],
                         "found": found, "matched_line": matched_line, "status": status,
                         "phase1_verified": r.get("verified", True),
                         "match_reasoning": getattr(matches.get(i), "reasoning", "")})
            continue
        for s in by_claim.get(i, []):
            hit = _find_match(s["stdout"], paper, tol) if paper is not None else None
            if hit:
                status, (found, matched_line) = "reproduced", hit
                break
            status = classify_failure(s) if s["rc"] != 0 else "ran_but_mismatch"
        rows.append(
            {
                "claim": r["claim"],
                "metric": r["metric"],
                "paper": r["value"],
                "found": found,
                "matched_line": matched_line,
                "status": status,
                "phase1_verified": r.get("verified", True),
            }
        )
    n_ok = sum(x["status"] == "reproduced" for x in rows)
    return rows, n_ok