from unittest.mock import Mock, patch

import requests

from lm_eval.tasks.harmbench.prompt_utils import prepare_prompt
from lm_eval.tasks.harmbench.remote import request_completion


class WordTokenizer:
    def encode(self, text, *, add_special_tokens):
        del add_special_tokens
        return text.split()

    def decode(self, token_ids, *, skip_special_tokens):
        del skip_special_tokens
        return " ".join(token_ids)


def test_prepare_prompt_preserves_short_completion():
    prompt, original_tokens, retained_tokens, prompt_tokens = prepare_prompt(
        "one two",
        build_prompt=lambda completion: f"prefix {completion} suffix",
        tokenizer=WordTokenizer(),
        max_completion_tokens=512,
        max_context_tokens=2048,
        max_new_tokens=1,
    )

    assert prompt == "prefix one two suffix"
    assert original_tokens == retained_tokens == 2
    assert prompt_tokens == 4


def test_prepare_prompt_right_truncates_to_completion_limit():
    completion = " ".join(str(index) for index in range(514))

    prompt, original_tokens, retained_tokens, prompt_tokens = prepare_prompt(
        completion,
        build_prompt=lambda prepared: f"prefix {prepared} suffix",
        tokenizer=WordTokenizer(),
        max_completion_tokens=512,
        max_context_tokens=2048,
        max_new_tokens=1,
    )

    assert original_tokens == 514
    assert retained_tokens == 512
    assert prompt_tokens == 514
    assert prompt.endswith("510 511 suffix")


def test_prepare_prompt_uses_remaining_context_budget():
    prompt, original_tokens, retained_tokens, prompt_tokens = prepare_prompt(
        "one two three four",
        build_prompt=lambda completion: f"prefix words {completion} suffix",
        tokenizer=WordTokenizer(),
        max_completion_tokens=512,
        max_context_tokens=6,
        max_new_tokens=1,
    )

    assert original_tokens == 4
    assert retained_tokens == 2
    assert prompt == "prefix words one two suffix"
    assert prompt_tokens == 5


@patch("lm_eval.tasks.harmbench.remote.acquire_judge_rate_limit")
@patch("lm_eval.tasks.harmbench.remote.requests.post")
def test_request_completion_uses_raw_completion_endpoint(mock_post, mock_acquire):
    response = Mock(ok=True)
    response.json.return_value = {"choices": [{"text": "yes"}]}
    mock_post.return_value = response

    result = request_completion(
        "[INST] classify [/INST]",
        api_url="https://example.test/v1",
        api_key="secret",
        model="user/cais/HarmBench",
        prompt_tokens=5,
    )

    assert result == "yes"
    mock_acquire.assert_called_once_with("https://example.test/v1:user/cais/HarmBench")
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "https://example.test/v1/completions"
    assert mock_post.call_args.kwargs["json"]["prompt"] == "[INST] classify [/INST]"
    assert "messages" not in mock_post.call_args.kwargs["json"]


@patch("lm_eval.tasks.harmbench.remote.time.sleep")
@patch("lm_eval.tasks.harmbench.remote.acquire_judge_rate_limit")
@patch("lm_eval.tasks.harmbench.remote.requests.post")
def test_request_completion_does_not_retry_permanent_error(
    mock_post, mock_acquire, mock_sleep, caplog
):
    response = Mock(ok=False, status_code=400, text="maximum context length exceeded")
    mock_post.return_value = response

    result = request_completion(
        "prompt",
        api_url="https://example.test/v1",
        api_key="secret",
        model="user/cais/HarmBench",
        prompt_tokens=2049,
        max_retries=6,
    )

    assert result is None
    assert mock_post.call_count == 1
    assert mock_acquire.call_count == 1
    mock_sleep.assert_not_called()
    assert "HTTP 400: maximum context length exceeded" in caplog.text


@patch("lm_eval.tasks.harmbench.remote.time.sleep")
@patch("lm_eval.tasks.harmbench.remote.acquire_judge_rate_limit")
@patch("lm_eval.tasks.harmbench.remote.requests.post")
def test_request_completion_retries_transient_error(
    mock_post, mock_acquire, mock_sleep
):
    success = Mock(ok=True)
    success.json.return_value = {"choices": [{"text": "no"}]}
    mock_post.side_effect = [
        requests.ConnectionError("temporarily unavailable"),
        success,
    ]

    result = request_completion(
        "prompt",
        api_url="https://example.test/v1",
        api_key="secret",
        model="user/cais/HarmBench",
        prompt_tokens=10,
        max_retries=2,
    )

    assert result == "no"
    assert mock_post.call_count == 2
    assert mock_acquire.call_count == 2
    mock_sleep.assert_called_once_with(1)
