import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.models.api_models import create_image_prompt
from lm_eval.models.openai_completions import LocalCompletionsAPI


@pytest.fixture
def api():
    return LocalCompletionsAPI(
        base_url="http://test-url.com", tokenizer_backend=None, model="gpt-3.5-turbo"
    )


@pytest.fixture
def api_tokenized():
    return LocalCompletionsAPI(
        base_url="http://test-url.com",
        model="EleutherAI/pythia-1b",
        tokenizer_backend="huggingface",
    )


@pytest.fixture
def api_batch_ssl_tokenized():
    return LocalCompletionsAPI(
        base_url="https://test-url.com",
        model="EleutherAI/pythia-1b",
        verify_certificate=False,
        num_concurrent=2,
        tokenizer_backend="huggingface",
    )


def test_create_payload_generate(api):
    messages = ["Generate a story"]
    gen_kwargs = {
        "max_tokens": 100,
        "temperature": 0.7,
        "until": ["The End"],
        "do_sample": True,
        "seed": 1234,
    }
    payload = api._create_payload(messages, generate=True, gen_kwargs=gen_kwargs)

    assert payload == {
        "prompt": ["Generate a story"],
        "model": "gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["The End"],
        "seed": 1234,
    }


def test_create_payload_loglikelihood(api):
    messages = ["The capital of France is"]
    payload = api._create_payload(messages, generate=False, gen_kwargs=None)

    assert payload == {
        "model": "gpt-3.5-turbo",
        "prompt": ["The capital of France is"],
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True,
        "temperature": 0,
        "seed": 1234,
    }


@pytest.mark.parametrize(
    "input_messages, generate, gen_kwargs, expected_payload",
    [
        (
            ["Hello, how are"],
            True,
            {"max_gen_toks": 100, "temperature": 0.7, "until": ["hi"]},
            {
                "prompt": "Hello, how are",
                "model": "gpt-3.5-turbo",
                "max_tokens": 100,
                "temperature": 0.7,
                "stop": ["hi"],
                "seed": 1234,
            },
        ),
        (
            ["Hello, how are", "you"],
            True,
            {},
            {
                "prompt": "Hello, how are",
                "model": "gpt-3.5-turbo",
                "max_tokens": 256,
                "temperature": 0,
                "stop": [],
                "seed": 1234,
            },
        ),
    ],
)
def test_model_generate_call_usage(
    api, input_messages, generate, gen_kwargs, expected_payload
):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Act
        result = api.model_call(
            input_messages, generate=generate, gen_kwargs=gen_kwargs
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "json" in kwargs
        assert kwargs["json"] == expected_payload
        assert result == {"result": "success"}


def test_model_call_acquires_configured_rate_limiter():
    limited_api = LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="test-model",
        requests_per_minute=30,
    )
    with (
        patch.object(limited_api._rate_limiter, "acquire") as acquire,
        patch("requests.post") as mock_post,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        limited_api.model_call(["hello"], gen_kwargs={})

    acquire.assert_called_once_with()


def test_async_model_call_acquires_configured_rate_limiter():
    limited_api = LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="test-model",
        requests_per_minute=30,
    )
    response = AsyncMock()
    response.ok = True
    response.json = AsyncMock(return_value={"choices": [{"index": 0, "text": "ok"}]})
    response.raise_for_status = lambda: None
    session = MagicMock()
    session.post.return_value = DummyAsyncContextManager(response)

    async def run():
        with patch.object(
            limited_api._rate_limiter, "acquire_async", new_callable=AsyncMock
        ) as acquire:
            result = await limited_api.amodel_call(
                session,
                asyncio.Semaphore(1),
                ["hello"],
                gen_kwargs={},
            )
        return result, acquire

    result, acquire = asyncio.run(run())

    assert result == ["ok"]
    acquire.assert_awaited_once_with()


@pytest.mark.parametrize(
    "input_messages, generate, gen_kwargs, expected_payload",
    [
        (
            [[1, 2, 3, 4, 5]],
            False,
            None,
            {
                "model": "EleutherAI/pythia-1b",
                "prompt": [[1, 2, 3, 4, 5]],
                "max_tokens": 1,
                "logprobs": 1,
                "echo": True,
                "seed": 1234,
                "temperature": 0,
            },
        ),
    ],
)
def test_model_tokenized_call_usage(
    api_tokenized, input_messages, generate, gen_kwargs, expected_payload
):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Act
        result = api_tokenized.model_call(
            input_messages, generate=generate, gen_kwargs=gen_kwargs
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "json" in kwargs
        assert kwargs["json"] == expected_payload
        assert result == {"result": "success"}


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(
            "LocalChatCompletion",
            id="local-chat-completions",
        ),
        pytest.param(
            "OpenAIChatCompletion",
            id="openai-chat-completions",
        ),
    ],
)
def test_chat_template_payload_does_not_add_top_level_type(model_cls):
    from lm_eval.models import openai_completions

    model = getattr(openai_completions, model_cls)(
        base_url="http://test-url.com",
        model="test-model",
    )
    chat = [{"role": "user", "content": "Reply with one word: hello"}]

    messages = model.create_message((model.apply_chat_template(chat),))
    payload = model._create_payload(messages, generate=True, gen_kwargs={})

    assert payload["messages"] == chat
    assert "type" not in payload["messages"][0]


@pytest.mark.parametrize("include_legacy_type", [False, True])
def test_create_image_prompt_uses_content_parts_without_top_level_type(
    include_legacy_type,
):
    class DummyImage:
        def save(self, buf, format):
            buf.write(b"image-bytes")

    chat = [{"role": "user", "content": "Describe this image"}]
    if include_legacy_type:
        chat[0]["type"] = "text"

    messages = create_image_prompt([DummyImage()], json.loads(json.dumps(chat)))

    assert "type" not in messages[-1]
    assert messages[-1]["content"][0]["type"] == "image_url"
    assert messages[-1]["content"][1] == {
        "type": "text",
        "text": "Describe this image",
    }


class DummyAsyncContextManager:
    def __init__(self, result):
        self.result = result

    async def __aenter__(self):
        return self.result

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.parametrize(
    "expected_inputs, expected_ctxlens, expected_cache_keys",
    [
        (
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            [3, 3, 3, 3],
            ["cache_key1", "cache_key2", "cache_key3", "cache_key4"],
        ),
    ],
)
def test_get_batched_requests_with_no_ssl(
    api_batch_ssl_tokenized, expected_inputs, expected_ctxlens, expected_cache_keys
):
    with (
        patch(
            "lm_eval.models.api_models.TCPConnector", autospec=True
        ) as mock_connector,
        patch(
            "lm_eval.models.api_models.ClientSession", autospec=True
        ) as mock_client_session,
        patch(
            "lm_eval.models.openai_completions.LocalCompletionsAPI.parse_logprobs",
            autospec=True,
        ) as mock_parse,
    ):
        mock_session_instance = AsyncMock()
        mock_post_response = AsyncMock()
        mock_post_response.status = 200
        mock_post_response.ok = True
        mock_post_response.json = AsyncMock(return_value={"mocked": "response"})
        mock_post_response.raise_for_status = lambda: None
        mock_session_instance.post = lambda *args, **kwargs: DummyAsyncContextManager(
            mock_post_response
        )
        mock_client_session.return_value.__aenter__.return_value = mock_session_instance
        mock_parse.return_value = [(1.23, True), (4.56, False)]

        async def run():
            return await api_batch_ssl_tokenized.get_batched_requests(
                expected_inputs,
                expected_cache_keys,
                generate=False,
                ctxlens=expected_ctxlens,
            )

        result_batches = asyncio.run(run())

        mock_connector.assert_called_with(limit=2, ssl=False)
        assert result_batches


def test_local_completionsapi_remote_tokenizer_authenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    LocalCompletionsAPI(
        base_url="https://secure-server",
        tokenizer_backend="remote",
        verify_certificate=True,
        ca_cert_path="secure.crt",
        auth_token="secure-token",  # noqa: S106
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == "secure-token"  # noqa: S105


def test_local_completionsapi_remote_tokenizer_unauthenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    LocalCompletionsAPI(
        base_url="http://localhost:8000",
        tokenizer_backend="remote",
        verify_certificate=False,
        ca_cert_path=None,
        auth_token=None,
    )
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["verify_certificate"] is False
    assert captured["ca_cert_path"] is None
    assert captured["auth_token"] is None


def test_localchatcompletion_remote_tokenizer_authenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    from lm_eval.models.openai_completions import LocalChatCompletion

    LocalChatCompletion(
        base_url="https://secure-server",
        tokenizer_backend="remote",
        verify_certificate=True,
        ca_cert_path="secure.crt",
        auth_token="secure-token",  # noqa: S106
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == "secure-token"  # noqa: S105


def test_localchatcompletion_remote_tokenizer_unauthenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    from lm_eval.models.openai_completions import LocalChatCompletion

    LocalChatCompletion(
        base_url="http://localhost:8000",
        tokenizer_backend="remote",
        verify_certificate=False,
        ca_cert_path=None,
        auth_token=None,
    )
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["verify_certificate"] is False
    assert captured["ca_cert_path"] is None
    assert captured["auth_token"] is None


class _RecordingTokenizer:
    """Stands in for an HF tokenizer and records the template kwargs it is given."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, chat_history, **kwargs):
        self.calls.append(kwargs)
        effort = kwargs.get("reasoning_effort", "medium")
        return f"Reasoning: {effort}\n" + chat_history[-1]["content"]


def _completions_with_tokenizer(**kwargs):
    model = LocalCompletionsAPI(
        base_url="http://test-url.com",
        model="test-model",
        tokenizer_backend=None,
        **kwargs,
    )
    model.tokenizer_backend = "huggingface"
    model.tokenizer = _RecordingTokenizer()
    return model


def test_local_completions_renders_chat_template_args():
    model = _completions_with_tokenizer(chat_template_args={"reasoning_effort": "high"})
    chat = [{"role": "user", "content": "What is 2+2?"}]

    assert model.apply_chat_template(chat) == "Reasoning: high\nWhat is 2+2?"
    assert model.tokenizer.calls[0]["reasoning_effort"] == "high"


def test_local_completions_without_chat_template_args_is_unchanged():
    model = _completions_with_tokenizer()
    chat = [{"role": "user", "content": "What is 2+2?"}]

    assert model.apply_chat_template(chat) == "Reasoning: medium\nWhat is 2+2?"
    assert "reasoning_effort" not in model.tokenizer.calls[0]


def _chat_model(**kwargs):
    from lm_eval.models.openai_completions import LocalChatCompletion

    return LocalChatCompletion(
        base_url="http://test-url.com", model="test-model", **kwargs
    )


def _chat_payload(model, messages, **gen_kwargs):
    import json

    prompt = model.apply_chat_template(messages)
    return model._create_payload(
        json.loads(prompt.prompt), generate=True, gen_kwargs=dict(gen_kwargs)
    )


def test_chat_template_args_travel_as_chat_template_kwargs():
    model = _chat_model(chat_template_args={"reasoning_effort": "high"})

    payload = _chat_payload(model, [{"role": "user", "content": "hi"}])

    assert payload["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_chat_payload_without_template_args_has_no_chat_template_kwargs():
    payload = _chat_payload(_chat_model(), [{"role": "user", "content": "hi"}])

    assert "chat_template_kwargs" not in payload
    assert "continue_final_message" not in payload


def test_chat_payload_sends_tools_from_json_gen_kwarg():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    import json

    payload = _chat_payload(
        _chat_model(),
        [{"role": "user", "content": "hi"}],
        tools=json.dumps(tools),
    )

    assert payload["tools"] == tools


def test_chat_payload_continues_a_closing_assistant_message():
    payload = _chat_payload(
        _chat_model(),
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Answer:"},
        ],
    )

    assert payload["continue_final_message"] is True
    assert payload["add_generation_prompt"] is False


def test_chat_parse_generations_appends_tool_calls_after_marker():
    import json

    from lm_eval.models.openai_completions import TOOL_CALLS_MARKER

    calls = [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]
    out = {
        "choices": [{"index": 0, "message": {"content": "Sure.", "tool_calls": calls}}]
    }

    (text,) = _chat_model().parse_generations(out)

    assert text == "Sure." + TOOL_CALLS_MARKER + json.dumps(calls)


def test_chat_parse_generations_puts_reasoning_between_think_markers():
    model = _chat_model(think_start_token="<think>", think_end_token="</think>")  # noqa: S106
    out = {
        "choices": [
            {"index": 0, "message": {"content": "391", "reasoning": "17*23=391"}}
        ]
    }

    assert model.parse_generations(out) == ["<think>17*23=391</think>391"]


def test_chat_parse_generations_leaves_answerless_reasoning_unclosed():
    # The server's parser puts an unclosed reasoning entirely in the reasoning field.
    model = _chat_model(think_start_token="<think>", think_end_token="</think>")  # noqa: S106
    out = {
        "choices": [{"index": 0, "message": {"content": None, "reasoning": "17*23 is"}}]
    }

    assert model.parse_generations(out) == ["<think>17*23 is"]


def test_chat_parse_generations_drops_reasoning_without_think_markers():
    out = {
        "choices": [
            {"index": 0, "message": {"content": "391", "reasoning": "17*23=391"}}
        ]
    }

    assert _chat_model().parse_generations(out) == ["391"]


def test_openai_chat_refuses_chat_template_args():
    from lm_eval.models.openai_completions import OpenAIChatCompletion

    model = OpenAIChatCompletion(
        model="gpt-x", chat_template_args={"reasoning_effort": "high"}
    )

    with pytest.raises(ValueError, match="chat_template_args"):
        model._create_payload(
            [{"role": "user", "content": "hi"}], generate=True, gen_kwargs={}
        )


def test_local_completions_passes_task_template_kwargs_to_tokenizer():
    model = _completions_with_tokenizer()
    tools = [{"type": "function", "function": {"name": "f"}}]

    model.apply_chat_template([{"role": "user", "content": "hi"}], tools=tools)

    assert model.tokenizer.calls[0]["tools"] == tools


def test_task_template_kwargs_refused_when_server_renders_template():
    from lm_eval.models.openai_completions import LocalChatCompletion

    model = LocalChatCompletion(base_url="http://test-url.com", model="test-model")

    with pytest.raises(ValueError, match=r"\['tools'\]"):
        model.apply_chat_template([{"role": "user", "content": "hi"}], tools=[{}])


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_local_completions_folds_enable_thinking_into_template_args(enable_thinking):
    model = _completions_with_tokenizer(
        enable_thinking=enable_thinking,
        chat_template_args={"reasoning_effort": "low"},
    )

    model.apply_chat_template([{"role": "user", "content": "hi"}])

    assert model.tokenizer.calls[0]["enable_thinking"] is enable_thinking
    assert model.tokenizer.calls[0]["reasoning_effort"] == "low"


def test_enable_thinking_travels_as_chat_template_kwargs():
    model = _chat_model(enable_thinking=False)

    payload = _chat_payload(model, [{"role": "user", "content": "hi"}])

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def _thinking_api(**kwargs):
    return LocalCompletionsAPI(
        base_url="http://test-url.com",
        model="test-model",
        tokenizer_backend=None,
        **kwargs,
    )


def _generate(model, text, until, finish_reason=None):
    from lm_eval.api.instance import Instance

    instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=("Question: Janet has 16 eggs...\nAnswer:", {"until": until}),
        idx=0,
    )
    sent: list[dict] = []

    def model_call(messages, generate=True, gen_kwargs=None, **_):
        sent.append(
            model._create_payload(messages, generate=True, gen_kwargs=gen_kwargs)
        )
        choice = {"index": 0, "text": text}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        return {"choices": [choice]}

    with patch.object(model, "model_call", side_effect=model_call):
        result = model.generate_until([instance])
    return result, instance, sent


REASONING = (
    "<think>\nAnalyze. Question: Janet has 16 eggs, eats 3, bakes 4.\n\nSo 9 * 2 = 18.\n</think>\n\n"
    "The answer is 18.\nQuestion: next one"
)


def test_api_thinking_holds_stops_back_and_strips_the_reasoning():
    model = _thinking_api(think_start_token="<think>", think_end_token="</think>")  # noqa: S106
    result, instance, sent = _generate(model, REASONING, ["Question:", "\n\n"])

    # the task stops don't go to the server (they'd cut into the reasoning)...
    assert "Question:" not in sent[0]["stop"] and "\n\n" not in sent[0]["stop"]
    # ...and apply to the answer once the reasoning is stripped
    assert result == ["The answer is 18.\n"]
    info = instance.length_info[0]
    assert info["thinking_format_correct"] == 1
    assert info["thinking_length_chars"] > 0


def test_api_without_think_tokens_is_unchanged():
    model = _thinking_api()
    result, instance, sent = _generate(model, "The answer is 18.", ["Question:"])
    assert sent[0]["stop"][0] == "Question:"
    assert result == ["The answer is 18."]
    assert "skip_special_tokens" not in sent[0]
    assert "thinking_length_chars" not in instance.length_info[0]


@pytest.mark.parametrize(("reason", "flag"), [("length", 1), ("stop", 0)])
def test_api_records_whether_the_server_hit_the_token_budget(reason, flag):
    model = _thinking_api(think_start_token="<think>", think_end_token="</think>")  # noqa: S106
    result, instance, _ = _generate(
        model, REASONING, ["Question:"], finish_reason=reason
    )
    assert instance.length_info[0]["generation_truncated"] == flag
    # the reasoning is still stripped; the answer is a plain string to the task
    assert result == ["The answer is 18.\n"]


def test_api_without_a_finish_reason_records_no_truncation_flag():
    _, instance, _ = _generate(_thinking_api(), "The answer is 18.", ["Question:"])
    assert "generation_truncated" not in instance.length_info[0]


def test_concurrent_answers_carry_their_finish_reason(api):
    from lm_eval.models.api_models import with_finish_reasons

    outputs = {
        "choices": [
            {"index": 1, "text": "b", "finish_reason": "length"},
            {"index": 0, "text": "a", "finish_reason": "stop"},
        ]
    }
    tagged = with_finish_reasons(api.parse_generations(outputs), outputs)
    assert tagged == ["a", "b"]
    assert [t.finish_reason for t in tagged] == ["stop", "length"]
    # a response that doesn't line up one choice per text is left untouched
    assert with_finish_reasons(["a"], {"choices": []}) == ["a"]


def test_api_keeps_special_tokens_when_the_markers_are_special():
    model = _thinking_api(think_end_token="<|channel|>final<|message|>")  # noqa: S106
    model.keep_special_tokens = (
        True  # as derived from a tokenizer where they're special
    )
    _, _, sent = _generate(
        model, "analysis ... <|channel|>final<|message|>18", ["Question:"]
    )
    assert sent[0]["skip_special_tokens"] is False


def test_special_tokens_are_read_from_the_tokenizer():
    from types import SimpleNamespace

    from lm_eval.models.api_models import _special_tokens

    class Added:
        def __init__(self, text, special):
            self.text, self.special = text, special

        def __str__(self):
            return self.text

    tokenizer = SimpleNamespace(
        all_special_tokens=["<|return|>"],
        added_tokens_decoder={
            1: Added("<|channel|>", True),
            2: Added("<think>", False),
        },
    )
    assert _special_tokens(tokenizer) == {"<|return|>", "<|channel|>"}
    assert _special_tokens(None) == set()


def test_request_progress_reports_about_every_five_percent(caplog):
    import logging

    from lm_eval.models.api_models import _RequestProgress

    progress = _RequestProgress("generate_until", 240)
    with caplog.at_level(logging.INFO, logger="lm_eval.models.api_models"):
        for _ in range(240):
            progress.add(1)
    lines = [
        r.getMessage() for r in caplog.records if "requests done" in r.getMessage()
    ]
    assert len(lines) == 20  # every 12 of 240
    assert lines[0].startswith("generate_until: 12/240 requests done (")
    assert lines[-1].startswith("generate_until: 240/240 requests done (")


def test_concurrent_requests_log_progress_and_retries(api, caplog):
    import logging

    calls = {"n": 0}

    async def amodel_call(*, messages, **_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")  # first call is retried
        return ["answer"] * len(messages)

    api._concurrent = 4
    with (
        patch.object(api, "amodel_call", side_effect=amodel_call),
        caplog.at_level(logging.INFO, logger="lm_eval.models.api_models"),
    ):
        results = asyncio.run(
            api.get_batched_requests(
                [f"prompt {i}" for i in range(5)],
                [None] * 5,
                generate=True,
                gen_kwargs={},
            )
        )
    assert [r for batch in results for r in batch] == ["answer"] * 5
    lines = [
        r.getMessage() for r in caplog.records if "requests done" in r.getMessage()
    ]
    assert lines[-1].startswith("generate_until: 5/5 requests done (")
    assert lines[-1].endswith("1 retries so far)")


def _echo_choice(token_logprobs):
    return {
        "index": 0,
        "logprobs": {
            "token_logprobs": token_logprobs,
            "top_logprobs": [None] + [{"x": lp} for lp in token_logprobs[1:]],
        },
    }


def test_parse_logprobs_scores_only_the_continuation():
    # context [1, 2, 3], continuation [4, 5], then the one generated token
    out = {"choices": [_echo_choice([None, -1.0, -2.0, -3.0, -4.0, -9.0])]}
    [(logprob, _)] = LocalCompletionsAPI.parse_logprobs(
        out, tokens=[[1, 2, 3, 4, 5]], ctxlens=[3]
    )
    assert logprob == -7.0


def test_parse_logprobs_ignores_an_extra_server_bos():
    # The server tokenized one more token at the start (its own BOS before the
    # chat template's) than the local tokenizer that counted ctxlen: the score
    # must still cover exactly the continuation, not the last context token.
    out = {"choices": [_echo_choice([None, -0.5, -1.0, -2.0, -30.0, -3.0, -4.0, -9.0])]}
    [(logprob, _)] = LocalCompletionsAPI.parse_logprobs(
        out, tokens=[[1, 2, 3, 99, 4, 5]], ctxlens=[4]
    )
    assert logprob == -7.0


def test_parse_logprobs_without_tokens_slices_from_ctxlen():
    out = {"choices": [_echo_choice([None, -1.0, -2.0, -3.0, -4.0, -9.0])]}
    [(logprob, _)] = LocalCompletionsAPI.parse_logprobs(out, ctxlens=[3])
    assert logprob == -7.0


class _InfoAPI(LocalCompletionsAPI):
    """Returns Generations carrying backend metadata, as Apertus 2 does."""

    def parse_generations(self, outputs, **kwargs):
        from lm_eval.models.api_models import Generation

        texts = super().parse_generations(outputs, **kwargs)
        tagged = []
        for text in texts:
            gen = Generation(text)
            gen.generation_info = {"generation_custom": 1, "note": text}
            tagged.append(gen)
        return tagged


@pytest.mark.parametrize("num_concurrent", [1, 2])
def test_generation_info_reaches_length_info(num_concurrent):
    from lm_eval.api.instance import Instance

    model = _InfoAPI(
        base_url="http://test-url.com",
        model="test-model",
        tokenizer_backend=None,
        num_concurrent=num_concurrent,
    )
    instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=("Question?", {"until": []}),
        idx=0,
    )
    outputs = {"choices": [{"index": 0, "text": "42", "finish_reason": "length"}]}

    async def send_async(session, payload):
        return outputs

    with (
        patch.object(model, "_send", return_value=outputs) as send,
        patch.object(model, "_asend", side_effect=send_async) as send_concurrent,
    ):
        result = model.generate_until([instance])
    assert result == ["42"]
    info = instance.length_info[0]
    assert info["generation_custom"] == 1
    assert info["note"] == "42"
    assert info["generation_truncated"] == 1
    assert send.called is (num_concurrent == 1)
    assert send_concurrent.called is (num_concurrent > 1)
