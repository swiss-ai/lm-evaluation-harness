from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


DATA_DIR = Path(__file__).parent / "data"
QUESTION_FILE = DATA_DIR / "mt_bench" / "question.jsonl"

TEMPERATURE_CONFIG = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}


class _ListDocs(list):
    @property
    def features(self) -> dict[str, Any]:
        if not self:
            return {}
        return {key: None for key in self[0]}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _render_history(history: list[dict[str, str]]) -> str:
    lines = []
    for message in history:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content']}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _strip_thinking_traces(text: str) -> str:
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


class MTBenchTask(ConfigurableTask):
    VERSION = 1
    MAX_MULTITURN_STEPS = 2

    def __init__(self, config: dict | None = None, **kwargs):
        config = dict(config or {})
        config.pop("class", None)
        config.setdefault("process_results", self.process_results)
        super().__init__(config=config, **kwargs)

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs) -> None:
        self.dataset = {"test": _ListDocs(_load_jsonl(QUESTION_FILE))}

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["turns"][0]

    def doc_to_target(self, doc):
        return ""

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        arguments = deepcopy(self.config.generation_kwargs)
        arguments.setdefault("until", [])
        arguments.setdefault("max_gen_toks", 1024)
        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, arguments),
            idx=0,
            **kwargs,
        )

    def init_multiturn_state(self, doc: dict, ctx: str, gen_kwargs: dict) -> dict:
        gen_kwargs = deepcopy(gen_kwargs)
        if self.config.metadata.get("use_reference_temperature_config", True):
            temperature = TEMPERATURE_CONFIG.get(doc["category"], 0.7)
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = temperature >= 1e-4

        return {
            "doc": doc,
            "gen_kwargs": gen_kwargs,
            "history": [],
            "turn_idx": 0,
            "awaiting_response": False,
            "responses": [],
            "done": False,
        }

    def multiturn_is_done(self, state: dict) -> bool:
        return state["done"]

    def multiturn_next_request(self, state: dict) -> tuple[str, dict] | None:
        if state["done"]:
            return None
        if not state["awaiting_response"]:
            if state["turn_idx"] >= len(state["doc"]["turns"]):
                state["done"] = True
                return None
            state["history"].append(
                {
                    "role": "user",
                    "content": state["doc"]["turns"][state["turn_idx"]],
                }
            )
            state["awaiting_response"] = True
        return _render_history(state["history"]), deepcopy(state["gen_kwargs"])

    def multiturn_consume_response(self, state: dict, response: str) -> None:
        clean_response = _strip_thinking_traces(response)
        state["responses"].append(clean_response)
        state["history"].append({"role": "assistant", "content": clean_response})
        state["turn_idx"] += 1
        state["awaiting_response"] = False
        if state["turn_idx"] >= len(state["doc"]["turns"]):
            state["done"] = True

    def multiturn_result(self, state: dict) -> dict:
        return {
            "responses": state["responses"],
            "history": state["history"],
        }

    def process_results(self, doc, results):
        item = {
            "question_id": doc["question_id"],
            "category": doc["category"],
            "turns": doc["turns"],
            "reference": doc.get("reference", ["", ""]),
            "responses": results[0]["responses"],
            "history": results[0]["history"],
        }
        return {
            "mt_bench_score": item,
            "mt_bench_turn_1_score": item,
            "mt_bench_turn_2_score": item,
            "mt_bench_judge_success_rate": item,
            "avg_word_count": sum(
                len(response.split()) for response in item["responses"]
            ),
        }

    def aggregation(self):
        from lm_eval.tasks.mt_bench.metric import (
            mt_bench_score,
            mt_bench_turn_1_score,
            mt_bench_turn_2_score,
            mt_bench_judge_success_rate,
        )

        return {
            "mt_bench_score": mt_bench_score,
            "mt_bench_turn_1_score": mt_bench_turn_1_score,
            "mt_bench_turn_2_score": mt_bench_turn_2_score,
            "mt_bench_judge_success_rate": mt_bench_judge_success_rate,
            "avg_word_count": np.mean,
        }

    def higher_is_better(self):
        return {
            "mt_bench_score": True,
            "mt_bench_turn_1_score": True,
            "mt_bench_turn_2_score": True,
            "mt_bench_judge_success_rate": True,
            "avg_word_count": False,
        }
