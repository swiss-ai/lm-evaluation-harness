import dataclasses
import json

from datasets import Dataset

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks.multi_if import instructions_registry


# Number of turns played by `multi-if-multiturn` (the dataset has up to 3).
MAX_TURNS = 3


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, str | int | None]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_docs(dataset):
    processed_dataset = []
    for doc in dataset:
        prompts, instruction_ids, kwargs = [], [], []
        for t in (1, 2, 3):
            if not doc[f"turn_{t}_prompt"]:
                break
            prompts.append(json.loads(doc[f"turn_{t}_prompt"])["content"])
            instruction_ids.append(json.loads(doc[f"turn_{t}_instruction_id_list"]))
            kwargs.append(doc[f"turn_{t}_kwargs"])  # raw JSON, decoded in _score
        processed_doc = {
            "key": doc["key"],
            "language": doc["language"],
            "prompt": prompts[0],
            "prompts": prompts,
            "instruction_ids": instruction_ids,
            "kwargs": kwargs,
        }
        processed_dataset.append(processed_doc)
    return Dataset.from_generator(lambda: (doc for doc in processed_dataset))


def _score(doc, i, response):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_ids"][i],
        prompt=doc["prompts"][i],
        kwargs=[json.loads(k) for k in json.loads(doc["kwargs"][i])],
    )
    strict = test_instruction_following_strict(inp, response)
    loose = test_instruction_following_loose(inp, response)
    return {
        "prompt_level_strict_acc": strict.follow_all_instructions,
        "inst_level_strict_acc": strict.follow_instruction_list,
        "prompt_level_loose_acc": loose.follow_all_instructions,
        "inst_level_loose_acc": loose.follow_instruction_list,
    }


def _with_overall(scores, language, prefix=""):
    # `overall` carries the 4 raw scores; agg_overall averages them after
    # aggregation, as in the paper. `overall_<language>` is only emitted for
    # this doc's language, so it aggregates over that language only.
    out = {f"{prefix}{name}": value for name, value in scores.items()}
    out[f"{prefix}overall"] = scores
    out[f"{prefix}overall_{language.lower()}"] = scores
    return out


def process_results(doc, results):
    responses = results[0]
    if isinstance(responses, str):  # multi-if: one response
        return _with_overall(_score(doc, 0, responses), doc["language"])
    # multi-if-multiturn: one response per turn
    out = {}
    for i, response in enumerate(responses):
        scores = _score(doc, i, response)
        out |= _with_overall(scores, doc["language"], prefix=f"turn_{i + 1}_")
    return out


class MultiIFTask(ConfigurableTask):
    def __init__(self, config=None):
        # TaskConfig rejects the `class` key
        config = {k: v for k, v in (config or {}).items() if k != "class"}
        super().__init__(config=config)

    def build_initial_messages(self, doc):
        return []

    def next_user_turn(self, doc, history, turn_idx):
        if turn_idx < min(len(doc["prompts"]), MAX_TURNS):
            return doc["prompts"][turn_idx]
        return None


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc


def agg_overall(items):
    """Paper headline: mean of the 4 aggregated accuracies."""
    return (
        sum(x["prompt_level_strict_acc"] for x in items) / len(items)
        + agg_inst_level_acc([x["inst_level_strict_acc"] for x in items])
        + sum(x["prompt_level_loose_acc"] for x in items) / len(items)
        + agg_inst_level_acc([x["inst_level_loose_acc"] for x in items])
    ) / 4
