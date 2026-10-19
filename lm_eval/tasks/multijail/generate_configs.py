#!/usr/bin/env python3
"""Generate the MultiJail per-language task configs and group configs.

Unintentional scenario (paper Sec. 3.1): the human-translated harmful prompt
as-is. Tasks ``multijail_<lang>``, group ``multijail`` (historical names).

Intentional scenario: the English AIM jailbreak instruction with the
(translated) harmful prompt inserted. Tasks ``multijail_intentional_<lang>``,
group ``multijail_intentional``.

Prompt construction lives in ``metric.process_docs_<scenario>_<lang>``.
Run: python generate_configs.py
"""

from pathlib import Path


# Order as in the paper (Table 1): English reference, then HRL, MRL, LRL.
LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "it": "Italian",
    "vi": "Vietnamese",
    "ar": "Arabic",
    "ko": "Korean",
    "th": "Thai",
    "bn": "Bengali",
    "sw": "Swahili",
    "jv": "Javanese",
}

SCENARIOS = {
    "unintentional": ("multijail", "multijail_{lang}"),
    "intentional": ("multijail_intentional", "multijail_intentional_{lang}"),
}

METRICS = ["unsafe", "safe", "invalid", "judge_error"]


def language_config(scenario: str, task: str, lang: str) -> str:
    return (
        f"task: {task}\n"
        "include: _multijail_yaml\n"
        f"process_docs: !function metric.process_docs_{scenario}_{lang}\n"
    )


def group_config(group: str, tasks: list[str]) -> str:
    lines = [f"group: {group}", "task:"]
    lines += [f"  - {t}" for t in tasks]
    lines.append("aggregate_metric_list:")
    for m in METRICS:
        lines += [
            f"  - metric: {m}",
            "    aggregation: mean",
            "    weight_by_size: true",
        ]
    lines += ["metadata:", "  version: 2.0", ""]
    return "\n".join(lines)


def main():
    out_dir = Path(__file__).parent
    for scenario, (group, task_fmt) in SCENARIOS.items():
        tasks = []
        for lang in LANGUAGES:
            task = task_fmt.format(lang=lang)
            tasks.append(task)
            (out_dir / f"{task}.yaml").write_text(
                language_config(scenario, task, lang), encoding="utf-8"
            )
        (out_dir / f"{group}.yaml").write_text(
            group_config(group, tasks), encoding="utf-8"
        )
        print(f"{group}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
