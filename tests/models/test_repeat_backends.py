from types import SimpleNamespace

import pytest

from lm_eval.api.instance import Instance


def attempts():
    params = {"temperature": 0.7, "max_gen_toks": 99, "until": ["STOP"], "seed": 7}
    return [
        Instance(
            "generate_until",
            {},
            ("question", params),
            0,
            metadata=("task", 1, 3),
            repeat_index=i,
        )
        for i in [2, 0, 1]
    ]


@pytest.mark.parametrize("concurrency", [1, 2])
def test_litellm_routes_distinct_seeds(monkeypatch, concurrency):
    import sys

    from lm_eval.models.api_models import JsonChatStr
    from lm_eval.models.litellm_llms import LiteLLMChatCompletion

    sent = []

    def complete(**payload):
        sent.append(payload)
        return SimpleNamespace(
            model_dump=lambda: {
                "choices": [{"index": 0, "message": {"content": "answer"}}]
            }
        )

    async def acomplete(**payload):
        return complete(**payload)

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=complete, acompletion=acomplete),
    )
    model = LiteLLMChatCompletion(model="provider/test", num_concurrent=concurrency)
    reqs = attempts()
    for req in reqs:
        req.arguments = (
            JsonChatStr('[{"role":"user","content":"question"}]'),
            req.args[1],
        )
    assert model.generate_until(reqs) == ["answer"] * 3
    assert sorted(p["seed"] for p in sent) == [7, 8, 9]
