import argparse
import json

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

FORCE_DOWNLOAD = False  # Set to True to force re-download of datasets


def count_samples_by_config(hf_dataset_id, split):
    result = {}
    try:
        configs = get_dataset_config_names(hf_dataset_id)
    except Exception:
        configs = [None]
    for config in configs:
        config_name = config if config is not None else "default"
        try:
            ds = (
                load_dataset(
                    hf_dataset_id,
                    config,
                    split=split,
                    download_mode=(
                        "force_redownload"
                        if FORCE_DOWNLOAD
                        else "reuse_dataset_if_exists"
                    ),
                )
                if config
                else load_dataset(
                    hf_dataset_id,
                    split=split,
                    download_mode=(
                        "force_redownload"
                        if FORCE_DOWNLOAD
                        else "reuse_dataset_if_exists"
                    ),
                )
            )
            count = len(ds)
        except Exception as e:
            print(
                f"Failed to load {hf_dataset_id} config {config_name} split {split}: {e}"
            )
            count = None
        result[config_name] = count
    return result


def count_samples_by_column(hf_dataset_id, split, column):
    result = {}
    try:
        configs = get_dataset_config_names(hf_dataset_id)
    except Exception:
        configs = [None]
    for config in configs:
        config_name = config if config is not None else "default"
        try:
            ds = (
                load_dataset(
                    hf_dataset_id,
                    config,
                    split=split,
                    download_mode=(
                        "force_redownload"
                        if FORCE_DOWNLOAD
                        else "reuse_dataset_if_exists"
                    ),
                )
                if config
                else load_dataset(
                    hf_dataset_id,
                    split=split,
                    download_mode=(
                        "force_redownload"
                        if FORCE_DOWNLOAD
                        else "reuse_dataset_if_exists"
                    ),
                )
            )
            value_counts = ds.to_pandas()[column].value_counts(dropna=False).to_dict()
        except Exception as e:
            print(
                f"Failed to load {hf_dataset_id} config {config_name} split {split} or column {column}: {e}"
            )
            value_counts = None
        result[config_name] = value_counts
    return result


def count_samples_all_configs_splits(hf_dataset_id):
    result = {}
    try:
        configs = get_dataset_config_names(hf_dataset_id)
    except Exception:
        configs = [None]
    for config in configs:
        config_name = config if config is not None else "default"
        try:
            ds = (
                load_dataset(
                    hf_dataset_id,
                    config,
                    download_mode=(
                        "force_redownload"
                        if FORCE_DOWNLOAD
                        else "reuse_dataset_if_exists"
                    ),
                )
                if config
                else load_dataset(
                    hf_dataset_id,
                    download_mode=(
                        "force_redownload"
                        if FORCE_DOWNLOAD
                        else "reuse_dataset_if_exists"
                    ),
                )
            )
            result[config_name] = {split: len(ds[split]) for split in ds.keys()}
        except Exception as e:
            print(f"Failed to load {hf_dataset_id} config {config_name}: {e}")
            result[config_name] = None
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Count language samples for each HF dataset in tasks.csv."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="scripts/tasks.csv",
        help="CSV file with dataset info",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="language_sample_count.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    all_results = {}
    for _, row in df.iterrows():
        harness_task = row.get("harness_task", None)
        hf_dataset_id = row.get("hf_dataset", None)
        category = row.get("category", None)
        language_info = str(row.get("language_info", "")).strip().lower()
        column = row.get("column", None)
        split = row.get("split", None)
        if pd.isna(harness_task) or pd.isna(hf_dataset_id):
            continue
        if language_info == "config":
            language_count = count_samples_by_config(hf_dataset_id, split)
        elif language_info == "column" and column:
            language_count = count_samples_by_column(hf_dataset_id, split, column)
        else:
            language_count = count_samples_all_configs_splits(hf_dataset_id)
        all_results[harness_task] = {
            "hf_dataset": hf_dataset_id,
            "category": category,
            "language_count": language_count,
        }
        print(json.dumps({harness_task: all_results[harness_task]}, indent=2))

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
