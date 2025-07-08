import json
from pprint import pprint

# === Load results ===
with open("out/results.json") as f:
    harness_result_filefile = json.load(f)
harness_results = harness_result_filefile["results"]


with open('src/language_bench.json', 'r') as file:
    LANGUAGE_BENCHMARKS = json.load(file)
ALL_LANGUAGES = LANGUAGE_BENCHMARKS.keys()

SWISS_LANGS = ["German", "French", "Italian", "Romansh"]

EU_LANGS = ["Albanian", "Armenian", "Basque", "Belarusian", "Bulgarian", "Catalan", "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian", "Finnish", "French", "Georgian", "German", "Greek", "Hungarian", "Italian", "Lithuanian", "North Macedonian", "Polish", "Portuguese", "Romanian", "Romansh", "Russian", "Serbian", "Slovak", "Spanish", "Swedish", "Ukrainian"]


# === Function to aggregate ===
def is_valid_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def aggregate_metrics(task_list, results_dict):
    aggregated = {}
    all_metrics = set()

    # Collect all metrics from available tasks
    for task in task_list:
        if task in results_dict:
            all_metrics.update(results_dict[task].keys())

    for metric in all_metrics:
        values = []
        for task in task_list:
            if task in results_dict:
                val = results_dict[task].get(metric)
                if is_valid_number(val):
                    values.append(float(val))

        if values:
            aggregated[metric] = sum(values) / len(values)
        else:
            aggregated[metric] = "N/A"

    return aggregated

# === Compute group aggregates ===
def main():
    for analysis_dimension in ["general_abilities", "factual_agnostic", "factual_regional"]:
        swiss_tasks = [f"{analysis_dimension}_{lang.lower()}" for lang in SWISS_LANGS]
        swiss_agg = aggregate_metrics(swiss_tasks, harness_results)
        task_name = f"{analysis_dimension}_swiss"
        swiss_agg['alias']=task_name
        harness_results[task_name] = swiss_agg
        
        eu_tasks = [f"{analysis_dimension}_{lang.lower()}" for lang in EU_LANGS]
        eu_agg = aggregate_metrics(eu_tasks, harness_results)
        task_name = f"{analysis_dimension}_eu"
        eu_agg['alias']= task_name
        harness_results[task_name] = eu_agg

        gloabL_tasks = [f"{analysis_dimension}_{lang.lower()}" for lang in ALL_LANGUAGES]
        global_agg = aggregate_metrics(gloabL_tasks, harness_results)
        task_name = f"{analysis_dimension}_global"
        global_agg['alias']= task_name
        harness_results[task_name] = global_agg

    # === Print results ===
    pprint(harness_results)

    # dump dict to json
    harness_result_filefile["results"] = harness_results
    

if __name__ == "__main__":
    main()
