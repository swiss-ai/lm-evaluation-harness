import json

from lm_eval.tasks.ifeval_multilingual.grader import scoring


def row_kwargs(doc):
    """Per instruction, the kwargs the checker takes: the JSON of `native` for a native:* instruction, else IFEval's
    fields that are set (as google/IFEval)."""
    out = []
    for kwargs in doc["kwargs"]:
        if kwargs.get("native"):
            out.append(json.loads(kwargs["native"]))
        else:
            out.append(
                {k: v for k, v in kwargs.items() if k != "native" and v is not None}
            )
    return out


def process_results(doc, results):
    code = doc["id"].split(":")[1]  # native:<code>:<n>
    row = {
        "prompt": doc["prompt"],
        "instruction_id_list": doc["instruction_id_list"],
        "kwargs": row_kwargs(doc),
    }
    scores = scoring.score(row, results[0], code)
    return {
        "prompt_level_strict_acc": all(scores["strict"]),
        "inst_level_strict_acc": scores["strict"],
        "prompt_level_loose_acc": all(scores["loose"]),
        "inst_level_loose_acc": scores["loose"],
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    return sum(flat_items) / len(flat_items)
