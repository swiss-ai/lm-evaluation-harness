"""Generate the YAML configs for the Llama-Guard PolygloToxicityPrompts tasks.

Run from anywhere: ``python utils.py``. Dataset: ``swiss-ai/polyglotoxicityprompts``
(configs ``ptp-<lang>`` with splits ``small``/``full`` and ``wildchat-<lang>`` with split
``train``).

Legacy family (names unchanged, 7 languages, greedy single sample, binary score):
    full/      polyglotoxicitypromptsllama_full_<language>      group polyglotoxicitypromptsllama
    small/     polyglotoxicitypromptsllama_small_<language>     group polyglotoxicitypromptsllama_small
    wildchat/  polyglotoxicitypromptsllama_wildchat_<language>  group polyglotoxicitypromptsllama_wildchat

v2 family (paper protocol, all 17 languages, k=10 samples, continuous score):
    v2_small/     polyglotoxicitypromptsllama_v2_small_<language>     group polyglotoxicitypromptsllama_v2_small
    v2_wildchat/  polyglotoxicitypromptsllama_v2_wildchat_<language>  group polyglotoxicitypromptsllama_v2_wildchat
"""

from pathlib import Path


ALL_LANGUAGES = {
    "ar": "arabic",
    "cs": "czech",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sv": "swedish",
    "zh": "chinese",
}
# Legacy family: the languages officially supported by Llama Guard 3/4 (minus Thai).
LEGACY_LANGUAGES = ["de", "en", "es", "fr", "hi", "it", "pt"]

# (folder, task prefix, group name, dataset config prefix, split, data subset)
LEGACY = [
    (
        "full",
        "polyglotoxicitypromptsllama_full",
        "polyglotoxicitypromptsllama",
        "ptp",
        "full",
        "full",
    ),
    (
        "small",
        "polyglotoxicitypromptsllama_small",
        "polyglotoxicitypromptsllama_small",
        "ptp",
        "small",
        "small",
    ),
    (
        "wildchat",
        "polyglotoxicitypromptsllama_wildchat",
        "polyglotoxicitypromptsllama_wildchat",
        "wildchat",
        "train",
        "wildchat",
    ),
]
V2 = [
    (
        "v2_small",
        "polyglotoxicitypromptsllama_v2_small",
        "polyglotoxicitypromptsllama_v2_small",
        "ptp",
        "small",
        "small",
    ),
    (
        "v2_wildchat",
        "polyglotoxicitypromptsllama_v2_wildchat",
        "polyglotoxicitypromptsllama_v2_wildchat",
        "wildchat",
        "train",
        "wildchat",
    ),
]
LEGACY_METRICS = ["score"]
V2_METRICS = [
    "avg_toxicity",
    "expected_max_toxicity",
    "toxicity_probability",
    "unsafe_rate",
    "judge_error_rate",
]

BASE = Path(__file__).parent


def task_yaml(task, template, ds_prefix, lang, split, subset):
    return (
        f"task: {task}\n"
        f"include: ../{template}\n"
        f"dataset_name: {ds_prefix}-{lang}\n"
        f"test_split: {split}\n"
    )


def group_yaml(group, tasks, metrics):
    lines = [f"group: {group}", "task:"]
    lines += [f"  - {t}" for t in tasks]
    lines.append("aggregate_metric_list:")
    for m in metrics:
        lines += [
            f"  - metric: {m}",
            "    aggregation: mean",
            "    weight_by_size: true",
        ]
    lines += [
        "metadata:",
        "  version: 1.0" if metrics == LEGACY_METRICS else "  version: 2.0",
    ]
    return "\n".join(lines) + "\n"


def generate(defs, langs, template, metrics, group_file):
    for folder, prefix, group, ds_prefix, split, subset in defs:
        (BASE / folder).mkdir(exist_ok=True)
        tasks = []
        for lang in langs:
            task = f"{prefix}_{ALL_LANGUAGES[lang]}"
            (BASE / folder / f"{task}.yaml").write_text(
                task_yaml(task, template, ds_prefix, lang, split, subset)
            )
            tasks.append(task)
        (BASE / group_file(folder, group)).write_text(group_yaml(group, tasks, metrics))


def main():
    generate(
        LEGACY,
        LEGACY_LANGUAGES,
        "_polyglotoxicitypromptsllama_yaml",
        LEGACY_METRICS,
        lambda folder, group: f"polyglotoxicityprompts_{folder}.yaml",
    )
    generate(
        V2,
        list(ALL_LANGUAGES),
        "_polyglotoxicitypromptsllama_v2_yaml",
        V2_METRICS,
        lambda folder, group: f"{group}.yaml",
    )


if __name__ == "__main__":
    main()
