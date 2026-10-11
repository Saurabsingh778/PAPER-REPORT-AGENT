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


def compare(results, run_out, tol=0.02):
    """Return (rows, n_reproduced)."""
    by_claim = {}
    for s in (run_out or {}).get("steps", []):
        for i in s.get("claim_indices", []):
            by_claim.setdefault(i, []).append(s)

    rows = []
    for i, r in enumerate(results, 1):
        paper = first_number(r["value"])
        status, found = "not_attempted", None
        for s in by_claim.get(i, []):
            nums = [float(x) for x in NUM.findall(s["stdout"])]
            hits = [n for n in nums if paper is not None and _match(n, paper, tol)]
            if hits:
                status, found = "reproduced", hits[0]
                break
            status = classify_failure(s) if s["rc"] != 0 else "ran_but_mismatch"
        rows.append(
            {
                "claim": r["claim"],
                "metric": r["metric"],
                "paper": r["value"],
                "found": found,
                "status": status,
                "phase1_verified": r.get("verified", True),
            }
        )
    n_ok = sum(x["status"] == "reproduced" for x in rows)
    return rows, n_ok