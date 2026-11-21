import dataclasses

from lm_eval.api.metrics import is_degenerating_text
from lm_eval.tasks.ifbench import instructions_registry


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


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc


# --- Thinking runs ---------------------------------------------------------------
# Constraints (word counts, "no commas", ...) apply to the final reply, not to the
# reasoning, so the reasoning is stripped before scoring. This is idempotent with the
# model-level strip (`think_end_token` / `autodetect_think_tokens`): a response that
# no longer contains "</think>" is scored as is.

THINK_START = "<think>"
THINK_END = "</think>"


def strip_thinking(response: str) -> str:
    """Final reply of a thinking response.

    The text after the last ``</think>``. A reply that opens ``<think>`` but never
    closes it (truncated mid-reasoning) has no final reply, so it is empty and fails.
    A response with no think tags at all is returned unchanged.
    """
    if THINK_END in response:
        return response.rpartition(THINK_END)[2].strip()
    if response.lstrip().startswith(THINK_START):
        return ""
    return response


def _input_example(doc):
    return InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )


def think_process_results(doc, results):
    """``process_results`` for thinking runs, over the final reply of each response.

    Handles a single response (take_first / pass@k) or the full set of k samples
    (take_first_k for mean@k): prompt-level metrics are averaged over the samples and
    the instruction-level lists are concatenated. ``degeneration`` is computed on the
    raw response, since looping happens in the reasoning.
    """
    response = results[0]
    responses = list(response) if isinstance(response, list) else [response]
    inp = _input_example(doc)

    finals = [strip_thinking(r) for r in responses]
    strict = [test_instruction_following_strict(inp, r) for r in finals]
    loose = [test_instruction_following_loose(inp, r) for r in finals]
    degens = [int(is_degenerating_text(r.lower())) for r in responses]
    n = len(responses)

    return {
        "prompt_level_strict_acc": sum(o.follow_all_instructions for o in strict) / n,
        "inst_level_strict_acc": [b for o in strict for b in o.follow_instruction_list],
        "prompt_level_loose_acc": sum(o.follow_all_instructions for o in loose) / n,
        "inst_level_loose_acc": [b for o in loose for b in o.follow_instruction_list],
        "degeneration": sum(degens) / n,
    }


class ThinkPassAtKFilter:
    """pass@k: returns a response whose final reply follows every instruction, else
    the first. Strict success is preferred over loose, so the strict and loose
    prompt-level metrics are both a true pass@k whenever one response satisfies both.
    Chain a ``take_first`` after this filter."""

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list, doc):
            inp = _input_example(doc)

            def score(resp):
                final = strip_thinking(resp)
                return (
                    test_instruction_following_strict(
                        inp, final
                    ).follow_all_instructions,
                    test_instruction_following_loose(
                        inp, final
                    ).follow_all_instructions,
                )

            # `max` keeps the first maximal element, so an all-failing set of samples
            # falls back to the first response.
            return max(resp_list, key=score)

        return [[select(r, d)] for r, d in zip(resps, docs)]
