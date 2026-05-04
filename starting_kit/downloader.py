"""
Downloads the aims-foundations/measurement-db dataset from HuggingFace
and saves all Parquet files into the local ./Data directory.

Run once before training:
    python downloader.py
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "aims-foundations/measurement-db"
DATA_DIR = Path(__file__).parent / "Data"


def download_measurement_db():
    DATA_DIR.mkdir(exist_ok=True)

    api = HfApi()
    repo_files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    parquet_files = [f for f in repo_files if f.endswith(".parquet")]

    print(f"Found {len(parquet_files)} Parquet files. Downloading to {DATA_DIR} ...")

    for filename in parquet_files:
        dest = DATA_DIR / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            print(f"  [skip] {filename}")
            continue

        print(f"  [download] {filename}")
        tmp_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        print(f"    -> {tmp_path}")

    print("\nDone. All files are in:", DATA_DIR)


if __name__ == "__main__":
    download_measurement_db()
