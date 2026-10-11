"""GitHub access: parse repo URLs; read README, file tree, deps, script headers, Kaggle dataset refs."""
import base64
import re

import requests
import shutil
import io
import os
import zipfile
from pathlib import Path

from .config import GH_TOKEN

KAGGLE_INPUT_RE = re.compile(r"/kaggle/input/(?:datasets/)?([\w.-]+)/([\w.-]+)/")


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


def raw(owner, repo, branch, path, limit=None):
    """Fetch a file from raw.githubusercontent.com (not counted against the API rate limit)."""
    r = requests.get(
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}", timeout=30
    )
    if r.status_code != 200:
        return ""
    return r.text[:limit] if limit else r.text


def inspect_repo(url, max_scripts=80, head_chars=1800):
    """README, file tree, dependency files, the head (docstring/config) of every .py file,
    and any Kaggle datasets the scripts expect under /kaggle/input."""
    parsed = parse_repo(url)
    if not parsed:
        return None
    owner, repo = parsed
    info = gh(f"/repos/{owner}/{repo}")
    branch = info["default_branch"]
    tree = gh(f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
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

    # The docstring/config at the top of each script says what it prints and what it needs.
    heads, datasets = {}, set()
    for p in [x for x in paths if x.endswith(".py")][:max_scripts]:
        full = raw(owner, repo, branch, p)
        heads[p] = full[:head_chars]
        for m in KAGGLE_INPUT_RE.finditer(full):
            datasets.add(f"{m.group(1)}/{m.group(2)}")  # owner/slug

    return {
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "size_kb": info["size"],
        "files": paths[:400],
        "readme": readme[:12000],
        "deps": deps,
        "script_heads": heads,
        "kaggle_datasets": sorted(datasets),
    }



def download_repo_locally(url, dest_dir="./Downloads/Repo"):
    """Downloads and extracts a repository to dest_dir/<owner>_<repo>."""
    parsed = parse_repo(url)
    if not parsed:
        print(f"[!] Invalid GitHub URL: {url}")
        return None

    owner, repo = parsed
    print(f"[*] Downloading {owner}/{repo}...")

    headers = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        headers["Authorization"] = f"Bearer {GH_TOKEN}"

    # Get default branch
    repo_info = gh(f"/repos/{owner}/{repo}")
    default_branch = repo_info.get("default_branch", "main")

    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{default_branch}"
    resp = requests.get(zip_url, headers=headers, timeout=60)
    resp.raise_for_status()

    # Destination directory (absolute path to avoid relative cwd confusion)
    target_dir = Path(dest_dir).resolve() / f"{owner}_{repo}"
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # GitHub zips put everything inside a single top-level folder
        root_folder = zf.namelist()[0].split("/")[0]

        for member in zf.infolist():
            # Skip the root container directory itself
            if (
                member.filename == root_folder
                or member.filename == f"{root_folder}/"
            ):
                continue

            # Strip the top-level folder prefix
            rel_path = member.filename.replace(f"{root_folder}/", "", 1)
            target_file = target_dir / rel_path

            if member.is_dir():
                target_file.mkdir(parents=True, exist_ok=True)
            else:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    print(f"[✓] Saved {owner}/{repo} to {target_dir}")
    return str(target_dir)