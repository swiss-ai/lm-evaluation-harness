import argparse
import json
import os

import pandas as pd
from datasets import get_dataset_config_names, load_dataset


def count_samples(hf_dataset_id):
    result = {hf_dataset_id: {}}
    try:
        configs = get_dataset_config_names(hf_dataset_id)
    except Exception:
        # Some datasets have no configs, treat as single config
        configs = [None]

    for config in configs:
        config_name = config if config is not None else "default"
        try:
            dataset = (
                load_dataset(hf_dataset_id, config)
                if config
                else load_dataset(hf_dataset_id)
            )
        except Exception as e:
            print(f"Failed to load config {config_name}: {e}")
            continue
        result[hf_dataset_id][config_name] = {}
        for split in dataset.keys():
            try:
                count = len(dataset[split])
            except Exception as e:
                print(f"Failed to count split {split} in config {config_name}: {e}")
                count = None
            result[hf_dataset_id][config_name][split] = count
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Count samples in each split of HuggingFace datasets listed in tasks.csv."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="scripts/tasks.csv",
        help="CSV file with hf_dataset column",
    )
    parser.add_argument(
        "--output", type=str, default="sample_count.jsonl", help="Output JSONL file"
    )
    args = parser.parse_args()

    # Read CSV and extract unique dataset IDs
    df = pd.read_csv(args.csv)
    if "hf_dataset" not in df.columns:
        raise ValueError(f"Column 'hf_dataset' not found in {args.csv}")
    dataset_ids = df["hf_dataset"].dropna().unique()

    with open(args.output, "a") as f:
        for hf_dataset_id in dataset_ids:
            counts = count_samples(hf_dataset_id)
            f.write(json.dumps(counts) + "\n")
            print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
