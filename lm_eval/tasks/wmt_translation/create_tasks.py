#!/usr/bin/env python3
"""Generate WMT translation task YAML configs.

Usage:
    # Discover language pairs from pre-downloaded JSON
    python create_tasks.py --from-json /path/to/wmt_data.json

    # Discover language pairs via subset2evaluate (requires pip install subset2evaluate)
    python create_tasks.py --from-subset2evaluate

    # Use hardcoded list (no data source needed)
    python create_tasks.py --hardcoded

Data is NOT generated here — it is downloaded at runtime via subset2evaluate
when the tasks are actually run (see utils.py).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent

HARDCODED_DATASETS = {
    "wmt24": [
        "cs-uk", "en-cs", "en-de", "en-es", "en-hi",
        "en-is", "en-ja", "en-ru", "en-uk", "en-zh", "ja-zh",
    ],
    "wmt24pp": [
        "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES",
        "en-cs_CZ", "en-da_DK", "en-de_DE", "en-el_GR", "en-es_MX",
        "en-et_EE", "en-fa_IR", "en-fi_FI", "en-fil_PH", "en-fr_CA",
        "en-fr_FR", "en-gu_IN", "en-he_IL", "en-hi_IN", "en-hr_HR",
        "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT", "en-ja_JP",
        "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN",
        "en-mr_IN", "en-nl_NL", "en-no_NO", "en-pa_IN", "en-pl_PL",
        "en-pt_BR", "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK",
        "en-sl_SI", "en-sr_RS", "en-sv_SE", "en-sw_KE", "en-sw_TZ",
        "en-ta_IN", "en-te_IN", "en-th_TH", "en-tr_TR", "en-uk_UA",
        "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW", "en-zu_ZA",
    ],
    "wmt25": [
        "cs-de_DE", "cs-uk_UA", "en-ar_EG", "en-bho_IN", "en-cs_CZ",
        "en-et_EE", "en-is_IS", "en-it_IT", "en-ja_JP", "en-ko_KR",
        "en-mas_KE", "en-ru_RU", "en-sr_Cyrl_RS", "en-uk_UA",
        "en-zh_CN", "ja-zh_CN",
    ],
}


def discover_from_json(path):
    """Discover language pairs from pre-downloaded JSON."""
    print(f"Loading {path}...")
    with open(path) as f:
        data = json.load(f)

    by_group = defaultdict(set)
    for item in data:
        ds_name, lang_pair = item["dataset"].split("/")
        by_group[ds_name].add(lang_pair)

    return {k: sorted(v) for k, v in sorted(by_group.items())}


def discover_from_subset2evaluate():
    """Discover language pairs from subset2evaluate."""
    import subset2evaluate.utils

    print("Loading language pairs from subset2evaluate...")
    raw = subset2evaluate.utils.load_data_wmt_all(
        require_human=False,
        name_filter=lambda k: k[0] in {"wmt25", "wmt24", "wmt24pp"},
    )
    by_group = defaultdict(set)
    for k in raw:
        by_group[k[0]].add(k[1])
    return {k: sorted(v) for k, v in sorted(by_group.items())}


def write_task_yamls(datasets_map):
    """Generate per-language-pair YAML task configs."""
    all_tasks = {}
    for ds_name, lang_pairs in sorted(datasets_map.items()):
        for lang_pair in sorted(lang_pairs):
            task_name = f"{ds_name}_{lang_pair}"
            dataset_key = f"{ds_name}/{lang_pair}"

            yaml_path = TASK_DIR / f"{task_name}.yaml"
            content = (
                f"include: _wmt_translation_yaml\n"
                f"task: {task_name}\n"
                f"tag:\n"
                f"- wmt_translation\n"
                f"- {ds_name}_translation\n"
                f"dataset_kwargs:\n"
                f"  dataset_key: {dataset_key}\n"
            )
            with open(yaml_path, "w") as f:
                f.write(content)

            if ds_name not in all_tasks:
                all_tasks[ds_name] = []
            all_tasks[ds_name].append(task_name)

    return all_tasks


def write_group_yamls(all_tasks):
    """Generate group YAML configs."""
    group_names = []
    for ds_name, tasks in sorted(all_tasks.items()):
        group_name = f"{ds_name}_translation"
        group_names.append(group_name)

        lines = [
            f"group: {group_name}",
            "task:",
        ]
        for t in sorted(tasks):
            lines.append(f"- {t}")
        lines += [
            "aggregate_metric_list:",
            "- metric: cometkiwi22",
            "  aggregation: mean",
            "  weight_by_size: true",
            "metadata:",
            "  version: 1.0",
        ]
        yaml_path = TASK_DIR / f"{group_name}.yaml"
        with open(yaml_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Group {group_name}: {len(tasks)} tasks")

    lines = [
        "group: wmt_translation",
        "task:",
    ]
    for g in sorted(group_names):
        lines.append(f"- {g}")
    lines += [
        "aggregate_metric_list:",
        "- metric: cometkiwi22",
        "  aggregation: mean",
        "  weight_by_size: true",
        "metadata:",
        "  version: 1.0",
    ]
    with open(TASK_DIR / "wmt_translation.yaml", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Top-level group wmt_translation: {len(group_names)} sub-groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-json", type=str, help="Path to wmt_data.json")
    source.add_argument(
        "--from-subset2evaluate",
        action="store_true",
        help="Discover from subset2evaluate",
    )
    source.add_argument(
        "--hardcoded",
        action="store_true",
        help="Use hardcoded language pair list",
    )
    args = parser.parse_args()

    if args.from_json:
        datasets_map = discover_from_json(args.from_json)
    elif args.from_subset2evaluate:
        datasets_map = discover_from_subset2evaluate()
    else:
        datasets_map = HARDCODED_DATASETS

    total = sum(len(v) for v in datasets_map.values())
    print(f"\nFound {total} language pairs across {len(datasets_map)} datasets")

    print("\nWriting task YAML configs...")
    all_tasks = write_task_yamls(datasets_map)

    print("\nWriting group YAML configs...")
    write_group_yamls(all_tasks)

    n = sum(len(v) for v in all_tasks.values())
    print(f"\nDone! Generated {n} task configs.")
    print("Data will be downloaded automatically at runtime via subset2evaluate.")


if __name__ == "__main__":
    main()
