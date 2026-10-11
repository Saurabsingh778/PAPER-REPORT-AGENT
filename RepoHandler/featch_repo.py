import json
import os
import sys
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

    for url in repo_urls:
        repo_name = get_repo_name(url)
        target_path = os.path.join(repo_dest, repo_name)
        
        # Check if the folder already exists
        if os.path.exists(target_path):
            print(f"Skipping {repo_name}: Already exists in {repo_dest}")
        else:
            print(f"Downloading {repo_name} to {repo_dest}...")
            download_repo_locally(url, dest_dir=repo_dest)

    for url in data_urls:
        if "github.com" in url:
            repo_name = get_repo_name(url)
            target_path = os.path.join(data_dest, repo_name)
            
            # Check if the folder already exists
            if os.path.exists(target_path):
                print(f"Skipping data repo {repo_name}: Already exists in {data_dest}")
            else:
                print(f"Downloading data repo {repo_name} to {data_dest}...")
                download_repo_locally(url, dest_dir=data_dest)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        featch_and_save(sys.argv[1])
    else:
        print("Please provide a path to the JSON file.")