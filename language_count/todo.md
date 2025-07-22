JSONs

language_sample_count.json:
{
    "harness_task": {
        "hf_dataset": hf_dataset,
        "category": category,
        "language_count": {"language1": 100, "language2": 200, ...},
    },
    "harness_task2": {
        "hf_dataset": hf_dataset2,
        "category": category2,
        "language_count": {"language1": 100, "language2": 200, ...},
    },
    ...
}


Let's add some postprocessing steps for particular tasks. Implement a funcion postprocess_tasks() that makes these changes to language_sample_count.json.
- blend: only keeps the content of the "multiple-choice-questions" key
- 


Now, implement a script that takes language_sample_count.json and creates a new JSON file that contains the list of tasks per language or country, by category.

tasks_per_language.json:
{
    "language1": {"category1": [list of harness_task's that have this language and category1], "category2": [list of harness_task's that have this language and category2]},
    "language2": {"category1": [list of harness_task's that have this language2 and category1], "category2": [list of harness_task's that have this language2 and category2]},
    ...
    "country1": {"category1": [list of harness_task's that have this country and category1], "category2": [list of harness_task's that have this country and category2]},
}




VISUALIZE

Implement a script visualization.py that uses language_sample_count.json to create plots to visualize the number of samples per language in total, and by category.
