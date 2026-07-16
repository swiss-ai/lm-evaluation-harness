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
        return {"acc": int(results[0] == ["response to first", "response to second"])}

    def aggregation(self):
        return {"acc": np.mean}

    def higher_is_better(self):
        return {"acc": True}


def test_multi_turn_generate_batches_episode_waves():
    task = ToyMultiturnTask(
        config={
            "task": "toy_multiturn",
            "output_type": "multi_turn_generate",
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


class ToyScriptedTask(ConfigurableTask):
    """Uses the scripted convenience hooks (build_initial_messages /
    next_user_turn / should_stop) instead of the core multiturn_* hooks, so
    it exercises ``ConfigurableTask``'s default core-hook implementations."""

    def download(self, *args, **kwargs):
        self.dataset = {"test": ListDocs([{"user_turns": ["q1", "q2", "q3"]}])}

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return ""

    def build_initial_messages(self, doc):
        from lm_eval.api.utils import Message

        return [Message("system", "sys")]

    def next_user_turn(self, doc, history, turn_idx):
        turns = doc["user_turns"]
        return turns[turn_idx] if turn_idx < len(turns) else None

    def should_stop(self, doc, history, turn_idx):
        # Early-exit after the assistant answers the 2nd user turn.
        return turn_idx >= 1


def _scripted_task():
    return ToyScriptedTask(
        config={
            "task": "toy_scripted",
            "output_type": "multi_turn_generate",
            "test_split": "test",
            "num_fewshot": 0,
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
            ],
            "generation_kwargs": {"until": []},
            "metadata": {"version": 1},
        }
    )


def test_scripted_convenience_layer_drives_core_hooks():
    """The default core hooks on ConfigurableTask drive a scripted episode via
    build_initial_messages / next_user_turn / should_stop, honoring early-exit."""

    def fake_chat_template(messages, add_generation_prompt=False):
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)

    task = _scripted_task()
    doc = task.test_docs()[0]
    state = task.init_multiturn_state(
        doc,
        ctx="",
        gen_kwargs={"until": []},
        apply_chat_template=True,
        chat_template=fake_chat_template,
    )

    responses = []
    steps = 0
    while not task.multiturn_is_done(state) and steps < 10:
        nxt = task.multiturn_next_request(state)
        if nxt is None:
            break
        prompt, gen_kwargs = nxt
        # The growing history is rendered through the chat template.
        assert prompt.startswith("system:sys")
        assert gen_kwargs == {"until": []}
        resp = f"answer{state.turn_idx}"
        task.multiturn_consume_response(state, resp)
        responses.append(resp)
        steps += 1

    # should_stop fired after the 2nd turn, so q3 is never generated.
    assert responses == ["answer0", "answer1"]
    assert task.multiturn_result(state) == ["answer0", "answer1"]


def test_scripted_convenience_layer_requires_chat_template():
    import pytest

    task = _scripted_task()
    doc = task.test_docs()[0]
    with pytest.raises(RuntimeError, match="apply_chat_template"):
        task.init_multiturn_state(
            doc, ctx="", gen_kwargs={}, apply_chat_template=False, chat_template=None
        )
