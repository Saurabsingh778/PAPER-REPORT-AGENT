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


def featch_and_save(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    repo_urls = get_url(data)
    data_urls = get_data_url(data)

    for url in repo_urls:
        download_repo_locally(url, dest_dir="./Downloads/Repo")

    for url in data_urls:
        if "github.com" in url:
            download_repo_locally(url, dest_dir="./Downloads/Data")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        featch_and_save(sys.argv[1])
    else:
        print("Please provide a path to the JSON file.")