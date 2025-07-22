import argparse
import json
from collections import defaultdict


def get_language_sample_counts(language_count):
    """
    Returns a dict mapping language/country to dict of harness_task: sample_count.
    Handles both flat and nested (config) dicts.
    {lang: count} or {config: {lang: count}}
    """
    flat = {}
    if isinstance(language_count, dict):
        for k, v in language_count.items():
            if isinstance(v, dict):
                for lang, count in v.items():
                    if lang not in flat:
                        flat[lang] = 0 if count is None else count
                    else:
                        # If a language appears in multiple configs, sum counts if both are int
                        if count is not None and isinstance(count, int):
                            flat[lang] += count
            else:
                flat[k] = 0 if v is None else v
    return flat


def main():
    parser = argparse.ArgumentParser(
        description="Create tasks_per_language.json from language_sample_count.json."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="language_count/language_sample_count.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="language_count/tasks_per_language.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    tasks_per_language = defaultdict(lambda: defaultdict(dict))

    for harness_task, info in data.items():
        category = info.get("category")
        language_count = info.get("language_count", {})
        lang_counts = get_language_sample_counts(language_count)
        for lang, count in lang_counts.items():
            tasks_per_language[lang][category][harness_task] = count

    # Convert defaultdicts to dicts for JSON serialization
    tasks_per_language = {
        lang: {cat: task_map for cat, task_map in cat_map.items()}
        for lang, cat_map in tasks_per_language.items()
    }

    with open(args.output, "w") as f:
        json.dump(tasks_per_language, f, indent=2)


if __name__ == "__main__":
    main()
