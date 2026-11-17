"""Multi-IF, all three turns (facebookresearch/Multi-IF, metrics.py).

Each conversation is replayed turn by turn with the model's own earlier replies
fed back as assistant messages (task.MultiIF3TurnTask). Turn N's reply is
checked against turn N's instruction list, which in the dataset already
carries the instructions of the earlier turns. A turn with no prompt (56 rows
have no turn 3) is not generated and not scored, as upstream skips it.

Per turn, upstream's "overall" is the unweighted mean of four numbers pooled
over the conversations that have that turn: prompt-level strict, instruction-
level strict, prompt-level loose and instruction-level loose accuracy.
"""

from __future__ import annotations

import json
import math

from lm_eval.tasks.multi_if.utils import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)


TURNS = (1, 2, 3)


def _missing(value) -> bool:
    return value is None or value == "None" or len(str(value)) == 0


def process_docs(dataset):
    def turn(doc, n):
        if _missing(doc[f"turn_{n}_prompt"]):
            return None
        return {
            "prompt": json.loads(doc[f"turn_{n}_prompt"])["content"],
            "instruction_id_list": json.loads(doc[f"turn_{n}_instruction_id_list"]),
            # Kept as JSON strings: Arrow would merge the per-instruction dicts
            # into one struct type across rows.
            "kwargs": list(json.loads(doc[f"turn_{n}_kwargs"])),
        }

    def to_doc(doc):
        turns = []
        for n in TURNS:
            t = turn(doc, n)
            if t is None:
                break  # a later turn without its predecessor can't be replayed
            turns.append(t)
        return {"key": doc["key"], "language": doc["language"], "turns": turns}

    return dataset.map(to_doc, remove_columns=dataset.column_names)


def _score_turn(spec, response):
    inp = InputExample(
        key=0,
        instruction_id_list=spec["instruction_id_list"],
        prompt=spec["prompt"],
        kwargs=[json.loads(kw) for kw in spec["kwargs"]],
    )
    strict = test_instruction_following_strict(inp, response)
    loose = test_instruction_following_loose(inp, response)
    return {
        "prompt_strict": bool(strict.follow_all_instructions),
        "prompt_loose": bool(loose.follow_all_instructions),
        "inst_strict": [bool(x) for x in strict.follow_instruction_list],
        "inst_loose": [bool(x) for x in loose.follow_instruction_list],
    }


def process_results(doc, results):
    # multi_turn_generate hands over [[reply_1, reply_2, ...]].
    if len(results) == 1 and isinstance(results[0], list):
        results = results[0]
    if isinstance(results, str):
        results = [results]
    scores = {}
    for n in TURNS:
        if n <= len(doc["turns"]):
            response = results[n - 1] if n - 1 < len(results) else ""
            scores[n] = _score_turn(doc["turns"][n - 1], response or "")
        else:
            scores[n] = None
    out = {f"turn{n}_overall": scores[n] for n in TURNS}
    out["overall"] = [scores[n] for n in TURNS]
    return out


def _overall(items) -> float:
    """Upstream turn_N_all_languages_overall over the conversations that have turn N."""
    items = [i for i in items if i is not None]
    if not items:
        return math.nan
    inst_strict = [x for i in items for x in i["inst_strict"]]
    inst_loose = [x for i in items for x in i["inst_loose"]]
    parts = [
        sum(i["prompt_strict"] for i in items) / len(items),
        sum(inst_strict) / len(inst_strict) if inst_strict else math.nan,
        sum(i["prompt_loose"] for i in items) / len(items),
        sum(inst_loose) / len(inst_loose) if inst_loose else math.nan,
    ]
    return sum(parts) / len(parts)


def agg_turn_overall(items) -> float:
    return _overall(items)


def agg_overall(items) -> float:
    """Mean of the three turns' overall scores."""
    per_turn = [_overall([item[n - 1] for item in items]) for n in TURNS]
    per_turn = [s for s in per_turn if not math.isnan(s)]
    return sum(per_turn) / len(per_turn) if per_turn else math.nan
