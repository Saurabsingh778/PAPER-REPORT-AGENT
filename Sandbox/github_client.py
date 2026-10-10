"""GitHub API access: parse repo URLs and read README, file tree and dependency files."""
import base64
import re

import requests

from .config import GH_TOKEN


def gh(path):
    headers = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        headers["Authorization"] = f"Bearer {GH_TOKEN}"
    r = requests.get(f"https://api.github.com{path}", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_repo(url):
    """Return (owner, repo) or None. Rejects stubs like 'https://github.com/'."""
    m = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if not m:
        return None
    return m.group(1), m.group(2).removesuffix(".git")


def inspect_repo(url):
    """Return README text, file tree and dependency files (all via API, no clone)."""
    parsed = parse_repo(url)
    if not parsed:
        return None
    owner, repo = parsed
    info = gh(f"/repos/{owner}/{repo}")
    tree = gh(f"/repos/{owner}/{repo}/git/trees/{info['default_branch']}?recursive=1")
    paths = [t["path"] for t in tree["tree"] if t["type"] == "blob"]

    def fetch(p):
        j = gh(f"/repos/{owner}/{repo}/contents/{p}")
        return base64.b64decode(j["content"]).decode("utf-8", "replace")

    readme = next((fetch(p) for p in paths if p.lower().startswith("readme")), "")
    deps = {
        p: fetch(p)
        for p in paths
        if p.split("/")[-1] in ("requirements.txt", "environment.yml", "pyproject.toml")
    }
    return {
        "owner": owner,
        "repo": repo,
        "branch": info["default_branch"],
        "size_kb": info["size"],
        "files": paths[:300],
        "readme": readme[:12000],
        "deps": deps,
    }