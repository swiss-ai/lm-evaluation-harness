import csv
import json
import os

# Path to the JSON file
json_path = os.path.join(os.path.dirname(__file__), "language_sample_count.json")
csv_path = os.path.join(os.path.dirname(__file__), "language_sample_count.csv")

# Step 1: Read the JSON file
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Collect all possible language keys and task/config names, and store category/hf_dataset per task
all_languages = set()
task_names = []
task_to_category = {}
task_to_hf_dataset = {}
task_to_lang_counts = {}

for task, info in data.items():
    hf_dataset = info.get("hf_dataset", "")
    category = info.get("category", "")
    language_count = info.get("language_count", {})

    # If language_count is a dict of dicts (multiple configs)
    if all(isinstance(v, dict) for v in language_count.values()):
        for config, lang_dict in language_count.items():
            task_name = f"{task}_{config}"
            task_names.append(task_name)
            task_to_category[task_name] = category
            task_to_hf_dataset[task_name] = hf_dataset
            task_to_lang_counts[task_name] = lang_dict
            all_languages.update(lang_dict.keys())
    else:
        # Single config
        task_name = task
        task_names.append(task_name)
        task_to_category[task_name] = category
        task_to_hf_dataset[task_name] = hf_dataset
        task_to_lang_counts[task_name] = language_count
        all_languages.update(language_count.keys())

# Step 3: Order languages column: lowercase first (languages), then uppercase (countries)
lowercase_langs = sorted([lang for lang in all_languages if lang and lang[0].islower()])
uppercase_langs = sorted([lang for lang in all_languages if lang and lang[0].isupper()])
ordered_languages = lowercase_langs + uppercase_langs

# Step 4: Prepare transposed data
transposed_rows = []

# First row: category
category_row = ["category"] + [task_to_category[task] for task in task_names]
transposed_rows.append(category_row)

# Second row: hf_dataset
hf_dataset_row = ["hf_dataset"] + [task_to_hf_dataset[task] for task in task_names]
transposed_rows.append(hf_dataset_row)

# Language rows
for lang in ordered_languages:
    row = [lang]
    for task in task_names:
        count = task_to_lang_counts[task].get(lang, "")
        row.append(count)
    transposed_rows.append(row)

# Header: first column is 'language', then all task_names
header = ["language"] + task_names

# Step 5: Write to CSV
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in transposed_rows:
        writer.writerow(row)

print(f"Transposed CSV written to {csv_path}")
