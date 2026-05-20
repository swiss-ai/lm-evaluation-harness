"""Download  o3-mini-2025-01-31 baseline answers for Arena-Hard v2.0.

Run once:
    python lm_eval/tasks/arena_hard_v2/download_baseline.py

This saves the baseline JSONL into the task directory so that metric.py
can load it locally instead of downloading from HuggingFace at runtime.
"""

import json
import os

from huggingface_hub import hf_hub_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "o3-mini-2025-01-31_baseline.jsonl")


def main():
    print("Downloading o3-mini-2025-01-31 baseline answers from HuggingFace...")
    path = hf_hub_download(
        repo_id="lmarena-ai/arena-hard-auto",
        filename="data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl",
        repo_type="dataset",
    )

    # Copy to task directory (hf_hub_download caches elsewhere)
    count = 0
    with open(path) as src, open(OUTPUT_PATH, "w") as dst:
        for line in src:
            dst.write(line)
            count += 1

    print(f"Saved {count} baseline answers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
