from __future__ import annotations

from typing import Any

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks.olympiadbench import utils


class OlympiadBenchTask(ConfigurableTask):
    VERSION = 1

    def __init__(self, config: dict | None = None, **kwargs):
        config = dict(config or {})
        config.pop("class", None)
        self.olympiadbench_subset = config.get("metadata", {}).get(
            "olympiadbench_subset", config.get("dataset_name", "")
        )
        super().__init__(config=config, **kwargs)

    def doc_to_text(self, doc: dict[str, Any]) -> str:
        prompt = utils.make_prompt(doc, self.olympiadbench_subset)
        problem_text = doc["question"]
        if "physics" in self.olympiadbench_subset and doc.get("context"):
            problem_text = f"{doc['context']}\n{problem_text}"
        return f"{prompt}\n{problem_text}"

    def doc_to_target(self, doc: dict[str, Any]) -> str:
        final_answer = doc.get("final_answer") or []
        return str(final_answer[0]) if final_answer else ""

    def process_results(self, doc, results):
        return utils.process_results(
            doc,
            results[0],
            subset=self.olympiadbench_subset,
        )
