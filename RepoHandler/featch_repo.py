import json
import os
import sys
from Logger import call_logged, get_phase, phase, start_run
from Sandbox.github_client import download_repo_locally, parse_repo

def get_url(json_file):
    return [
        item["url"] for item in json_file.get("code_repos", []) if "url" in item
    ]

def get_data_url(json_file):
    return [
        item["url"] for item in json_file.get("data_links", []) if "url" in item
    ]

def get_repo_name(url):
    """Extracts the repository name from a GitHub URL."""
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name

def _fetch_one(url, dest, label):
    """Download one GitHub repo into dest/<owner>_<repo>. Never raises: a dead link must not
    stop the others. download_repo_locally names the folder <owner>_<repo>, so that is also the
    name checked for 'already downloaded' (checking the bare repo name never matched)."""
    parsed = parse_repo(url)
    if not parsed:
        get_phase().warning(f"{label} link is not a GitHub repo URL, not downloaded", url=url)
        return
    owner, repo = parsed
    folder = f"{owner}_{repo}"
    if os.path.exists(os.path.join(dest, folder)):
        print(f"Skipping {label} {folder}: Already exists in {dest}")
        get_phase().info(f"{label} already present, skipped", repo=folder, dest=dest)
        return
    print(f"Downloading {label} {folder} to {dest}...")
    try:
        call_logged(download_repo_locally, url, dest_dir=dest, _name=f"download_{label.replace(' ', '_')}")
    except Exception as exc:
        print(f"[!] Could not download {url}: {type(exc).__name__}: {exc}")
        get_phase().warning(f"{label} download failed", url=url, error=f"{type(exc).__name__}: {exc}")

def featch_and_save(json_path, repo_dest="./Downloads/Repo", data_dest="./Downloads/Data"):
    """Download the code repos and data repos listed in a phase 1 extraction JSON.

    Defaults keep the old shared ./Downloads folders; the integrated pipeline passes a
    per-paper cache so one paper's data never ends up in another paper's run."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    repo_urls = get_url(data)
    data_urls = get_data_url(data)

    # Ensure destination directories exist
    os.makedirs(repo_dest, exist_ok=True)
    os.makedirs(data_dest, exist_ok=True)

    get_phase().info("fetch plan", code_repos=repo_urls, data_links=data_urls)

    for url in repo_urls:
        _fetch_one(url, repo_dest, "code repo")

    for url in data_urls:
        if "github.com" in url:
            _fetch_one(url, data_dest, "data repo")
        else:
            get_phase().warning("data link is not a GitHub repo, not downloaded", url=url)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with start_run("fetch_repos", meta={"phase1_json": sys.argv[1]}):
            with phase("phase2_fetch", "download the code repository and data repositories"):
                featch_and_save(sys.argv[1])
    else:
        print("Please provide a path to the JSON file.")
