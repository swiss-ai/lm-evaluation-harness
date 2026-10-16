"""
Take in a YAML, and output one task per (benchmark, language) of MuBench with this YAML
into one folder per language, plus one group per language (averaging its benchmarks)
and a top-level group (averaging the languages).

Example (local template variant):
    cd lm_eval/tasks/mubench/local
    python ../_generate_configs.py --base_yaml_path _mubench_template_yaml \
        --save_prefix_path mubench_local --task_prefix local
then regenerate the task index from the repo root:
    python scripts/build_task_index.py
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)

# 12 benchmarks
# task-name suffix -> (MuBench dataset folder, per-benchmark overrides)
BENCHMARKS = {
    "arc_challenge": ("ARCChallengeDataset", {}),
    "arc_easy": ("ARCEasyDataset", {}),
    "bmlama": ("BMLAMADataset", {}),
    "gpqa": ("GPQADataset", {}),
    "hellaswag": ("HellaswagDataset", {}),
    "mmlu": ("MMLUDataset", {}),
    "mmlu_pro": ("MMLUProDataset", {}),
    # The paper evaluates SNLI/MultiNLI 10-shot with the local template.
    "mnli": ("MNLIDataset", {"num_fewshot": 10, "fewshot_split": "validation"}),
    "snli": ("SNLIDataset", {"num_fewshot": 10, "fewshot_split": "validation"}),
    "storycloze": ("StoryClozeDataset", {}),
    "truthfulqa": ("TruthfulQADataset", {}),
    "winogrande": ("WinoGrandeDataset", {}),
}

# 61 languages
LANGUAGES = [
    "af", "ar", "az", "bg", "bn", "ca", "ceb", "cs", "da", "de", "el", "en",
    "es", "et", "fa", "fi", "fr", "ga", "gu", "he", "hi", "hr", "hu", "id",
    "is", "it", "ja", "jv", "kk", "km", "kn", "ko", "lt", "lv", "ml", "mr",
    "ms", "my", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sk", "sl", "sq",
    "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "uz", "vi",
    "zh",
]  # fmt: skip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="mubench")
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    parser.add_argument("--template_format", default="local_template")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    prefix = f"mubench_{args.task_prefix}" if args.task_prefix != "" else "mubench"

    language_groups = []
    for lang in tqdm(LANGUAGES):
        os.makedirs(lang, exist_ok=True)
        group = f"{prefix}_{lang}"
        language_groups.append(group)
        language_tasks = []

        for benchmark, (dataset, overrides) in BENCHMARKS.items():
            task = f"{prefix}_{benchmark}_{lang}"
            language_tasks.append(task)
            yaml_dict = {
                "include": f"../{base_yaml_name}",
                # lets `--tasks <prefix>_<benchmark>` run one benchmark in all languages
                "tag": [f"{prefix}_{benchmark}"],
                "task": task,
                "task_alias": benchmark,
                "dataset_name": f"{dataset}_{args.template_format}_{lang}",
                **overrides,
            }

            file_save_path = os.path.join(
                lang, args.save_prefix_path + f"_{benchmark}_{lang}.yaml"
            )
            eval_logger.info(f"Saving yaml for subset {benchmark} to {file_save_path}")
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    yaml_dict,
                    yaml_file,
                    allow_unicode=True,
                    sort_keys=False,
                )

        file_save_path = os.path.join(lang, f"_{args.save_prefix_path}_{lang}.yaml")
        eval_logger.info(f"Saving group config to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": group,
                    "group_alias": lang,
                    "task": language_tasks,
                    "aggregate_metric_list": [
                        {"metric": "acc", "weight_by_size": False}
                    ],
                    "metadata": {"version": 0.0},
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = f"_{args.save_prefix_path}.yaml"

    eval_logger.info(f"Saving top-level group config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": prefix,
                "task": language_groups,
                "aggregate_metric_list": [{"metric": "acc", "weight_by_size": False}],
                "metadata": {"version": 0.0},
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
            sort_keys=False,
        )
