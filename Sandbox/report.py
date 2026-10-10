"""Write the Phase 2 markdown + json report."""
import json
from pathlib import Path


def write_report(out_dir, rows, n_ok, unreproducible):
    lines = [
        "# Phase 2: Reproduction report\n",
        f"**Reproduced {n_ok}/{len(rows)} claims**\n",
    ]
    for i, r in enumerate(rows, 1):
        flag = "" if r["phase1_verified"] else " (target value not verified in paper text)"
        lines.append(
            f"{i}. [{r['status']}] {r['claim']} | paper={r['paper']} found={r['found']}{flag}"
        )
    lines += ["\n## Judged unreproducible by planner\n"] + [f"- {u}" for u in unreproducible]
    text = "\n".join(lines)
    out = Path(out_dir)
    (out / "phase2.md").write_text(text, encoding="utf-8")
    (out / "phase2.json").write_text(
        json.dumps({"reproduced": n_ok, "total": len(rows), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return text