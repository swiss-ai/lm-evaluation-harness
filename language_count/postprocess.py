import copy
import json


def postprocess_tasks(input_path, output_path=None):
    with open(input_path, "r") as f:
        data = json.load(f)
    data = copy.deepcopy(data)

    # 1. blend: only keep the content of the "multiple-choice-questions" key
    if "blend" in data:
        lc = data["blend"].get("language_count", {})
        if "multiple-choice-questions" in lc:
            data["blend"]["language_count"] = lc["multiple-choice-questions"]

    # 2. polyglotoxicityprompts_full: flatten nested structure
    if "polyglotoxicityprompts_full" in data:
        lc = data["polyglotoxicityprompts_full"].get("language_count", {})
        new_lc = {}
        for group in ["full", "small", "wildchat"]:
            if group in lc:
                for lang, count in lc[group].items():
                    if group not in new_lc:
                        new_lc[group] = {}
                    new_lc[group][lang] = count
        data["polyglotoxicityprompts_full"]["language_count"] = new_lc

    # 3. aya_redteaming: remove 'default' nesting
    if "aya_redteaming" in data:
        lc = data["aya_redteaming"].get("language_count", {})
        if "default" in lc:
            data["aya_redteaming"]["language_count"] = lc["default"]

    # 4. multijail: remove 'default' nesting and flatten
    if "multijail" in data:
        lc = data["multijail"].get("language_count", {})
        if "default" in lc:
            data["multijail"]["language_count"] = lc["default"]

    # Optionally write to a new file, or overwrite
    output_path = output_path or input_path
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Postprocessed file written to {output_path}")


if __name__ == "__main__":
    postprocess_tasks("language_count/language_sample_count.json")
