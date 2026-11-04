from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.anthropic_llms import AnthropicChat
from lm_eval.models.gguf import GGUFLM
from lm_eval.models.ibm_watsonx_ai import WatsonxLLM
from lm_eval.models.textsynth import TextSynthLM
from lm_eval.models.utils import retry_on_specific_exceptions


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


@pytest.mark.parametrize("backend", ["gguf", "textsynth", "watsonx"])
def test_legacy_apis_preserve_reordered_attempts_and_task_settings(
    monkeypatch, backend
):
    sent = []
    if backend == "gguf":
        model = GGUFLM(base_url="http://example.invalid")

        def complete(**kwargs):
            sent.append(kwargs)
            return {"choices": [{"text": "answer"}]}

        monkeypatch.setattr(model, "gguf_completion", complete)
    elif backend == "textsynth":
        monkeypatch.setenv("TEXTSYNTH_API_SECRET_KEY", "test")
        model = TextSynthLM(engine="test")

        def complete(**kwargs):
            sent.append(kwargs["json"])
            return SimpleNamespace(json=lambda: {"text": "answer"})

        monkeypatch.setattr("lm_eval.models.textsynth.textsynth_completion", complete)
    else:
        model = object.__new__(WatsonxLLM)
        LM.__init__(model)
        model.generate_params = {"random_seed": 123, "max_new_tokens": 12}

        def complete(context, params):
            sent.append(params)
            return "answer"

        model.model = SimpleNamespace(generate_text=complete)
    reqs = attempts()
    assert model.generate_until(reqs) == ["answer"] * 3
    seed_key = "random_seed" if backend == "watsonx" else "seed"
    budget_key = "max_new_tokens" if backend == "watsonx" else "max_tokens"
    assert [p[seed_key] for p in sent] == [9, 7, 8]
    assert all(p[budget_key] == 99 and p["temperature"] == 0.7 for p in sent)
    assert reqs[0].args[1]["seed"] == 7
    if backend == "watsonx":
        assert model.generate_params == {"random_seed": 123, "max_new_tokens": 12}


def test_anthropic_blocks_are_one_answer():
    output = {
        "content": [
            {"type": "thinking", "thinking": "private"},
            {"type": "text", "text": "A"},
            {"type": "text", "text": "B"},
        ]
    }
    assert AnthropicChat.parse_generations(None, output) == ["AB"]


def test_retry_exhaustion_preserves_terminal_error(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("lm_eval.models.utils.time.sleep", sleep)
    failed = Mock(side_effect=ConnectionError("offline"))
    wrapped = retry_on_specific_exceptions([ConnectionError], max_retries=3)(failed)
    with pytest.raises(ConnectionError, match="offline"):
        wrapped(seed=7)
    assert failed.call_count == 3
    assert sleep.call_count == 2
    assert all(call.kwargs == {"seed": 7} for call in failed.call_args_list)


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


def test_azure_forwards_sampling_and_concurrency_options(monkeypatch):
    from lm_eval.models.openai_completions import AzureOpenaiChatCompletionsLM

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-only")
    model = AzureOpenaiChatCompletionsLM(
        model="deployment",
        base_url="https://example.invalid",
        seed=7,
        num_concurrent=3,
        max_retries=2,
    )
    assert model._seed == 7 and model._concurrent == 3 and model.max_retries == 2
    assert model.header == {"api-key": "test-only"}
    payload = model._create_payload(
        [{"role": "user", "content": "question"}],
        gen_kwargs={"temperature": 0.7, "max_gen_toks": 99},
        seed=9,
    )
    assert payload["seed"] == 9 and payload["max_completion_tokens"] == 99


def test_sglang_translates_seed_to_sampling_params():
    from lm_eval.models.sglang_generate_API import SGLANGGENERATEAPI

    model = SGLANGGENERATEAPI(
        model="test", base_url="http://example.invalid", tokenizer_backend=None
    )
    payload = model._create_payload(
        "question",
        generate=True,
        gen_kwargs={"temperature": 0.7, "max_gen_toks": 99},
        seed=9,
    )
    assert payload["sampling_params"]["sampling_seed"] == 9
    assert payload["sampling_params"]["max_new_tokens"] == 99
