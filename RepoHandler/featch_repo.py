import json
import os
import sys
from Logger import call_logged, get_phase, phase, start_run
from Sandbox.github_client import download_repo_locally

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

def featch_and_save(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    repo_urls = get_url(data)
    data_urls = get_data_url(data)
    
    repo_dest = "./Downloads/Repo"
    data_dest = "./Downloads/Data"

    # Ensure destination directories exist
    os.makedirs(repo_dest, exist_ok=True)
    os.makedirs(data_dest, exist_ok=True)

    get_phase().info("fetch plan", code_repos=repo_urls, data_links=data_urls)

    for url in repo_urls:
        repo_name = get_repo_name(url)
        target_path = os.path.join(repo_dest, repo_name)

        # Check if the folder already exists
        if os.path.exists(target_path):
            print(f"Skipping {repo_name}: Already exists in {repo_dest}")
            get_phase().info("code repo already present, skipped", repo=repo_name, dest=repo_dest)
        else:
            print(f"Downloading {repo_name} to {repo_dest}...")
            call_logged(download_repo_locally, url, dest_dir=repo_dest, _name="download_code_repo")

    for url in data_urls:
        if "github.com" in url:
            repo_name = get_repo_name(url)
            target_path = os.path.join(data_dest, repo_name)

            # Check if the folder already exists
            if os.path.exists(target_path):
                print(f"Skipping data repo {repo_name}: Already exists in {data_dest}")
                get_phase().info("data repo already present, skipped", repo=repo_name, dest=data_dest)
            else:
                print(f"Downloading data repo {repo_name} to {data_dest}...")
                call_logged(download_repo_locally, url, dest_dir=data_dest, _name="download_data_repo")
        else:
            get_phase().warning("data link is not a GitHub repo, not downloaded", url=url)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with start_run("fetch_repos", meta={"phase1_json": sys.argv[1]}):
            with phase("phase2_fetch", "download the code repository and data repositories"):
                featch_and_save(sys.argv[1])
    else:
        print("Please provide a path to the JSON file.")