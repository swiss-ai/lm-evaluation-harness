import os
import re

import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def num_workers(predictions: list[list[str]]) -> int:
    """Threads for ``code_eval``: one per candidate program.

    Custom metrics are called once per document, so ``compute`` only ever sees that
    doc's candidates and HF's default of 4 splits a self-consistency run's k samples
    into several waves for no reason. Never drops below that default.
    """
    candidates = sum(len(p) for p in predictions)
    return max(4, min(candidates, os.cpu_count() or 4))


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    # Only compute pass@k for k that don't exceed the number of available samples
    # (code_eval errors otherwise). Lets a single-sample "first" filter coexist
    # with a multi-sample filter under one k=[1, N] metric entry.
    n = min(len(p) for p in predictions)
    k = [kk for kk in k if kk <= n] or [1]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
        num_workers=num_workers(predictions),
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + clean_text(r.replace("```python\n", "")) for r in resp]
        for resp, doc in zip(resps, docs)
    ]


def clean_text(text: str) -> str:
    return re.sub(r"\n(▁+)", lambda m: "\n" + " " * len(m.group(1)), text)


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"]
            + (clean_text(r) if r.find("```") == -1 else clean_text(r[: r.find("```")]))
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
