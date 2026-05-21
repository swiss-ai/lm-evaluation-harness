import numpy as np

from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate


class ListDocs(list):
    @property
    def features(self):
        return {key: None for key in self[0]}


class ScriptedLM(LM):
    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        return [f"response to {request.arguments[0]}" for request in requests]


class ToyMultiturnTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = {"test": ListDocs([{"turns": ["first", "second"]}])}

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "unused initial context"

    def doc_to_target(self, doc):
        return ""

    def init_multiturn_state(self, doc, ctx, gen_kwargs):
        return {"doc": doc, "turn": 0, "responses": []}

    def multiturn_is_done(self, state):
        return state["turn"] >= len(state["doc"]["turns"])

    def multiturn_next_request(self, state):
        return state["doc"]["turns"][state["turn"]], {"until": []}

    def multiturn_consume_response(self, state, response):
        state["responses"].append(response)
        state["turn"] += 1

    def multiturn_result(self, state):
        return state["responses"]

    def process_results(self, doc, results):
        return {
            "acc": int(results[0] == ["response to first", "response to second"])
        }

    def aggregation(self):
        return {"acc": np.mean}

    def higher_is_better(self):
        return {"acc": True}


def test_generate_until_multiturn_batches_episode_waves():
    task = ToyMultiturnTask(
        config={
            "task": "toy_multiturn",
            "output_type": "generate_until_multiturn",
            "test_split": "test",
            "num_fewshot": 0,
            "metric_list": [
                {
                    "metric": "acc",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
            "generation_kwargs": {"until": []},
            "metadata": {"version": 1},
        }
    )

    results = evaluate(
        lm=ScriptedLM(),
        task_dict={"toy_multiturn": task},
        limit=1,
        bootstrap_iters=0,
        log_samples=True,
    )

    assert results["results"]["toy_multiturn"]["acc,none"] == 1.0
