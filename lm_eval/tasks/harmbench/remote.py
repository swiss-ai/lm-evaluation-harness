"""Remote completion client for the HarmBench classifier."""

from __future__ import annotations

import logging
import time

import requests

from lm_eval.api.rate_limiter import acquire_judge_rate_limit


eval_logger = logging.getLogger(__name__)

# A judge other than the HarmBench classifier (e.g. HARMBENCH_JUDGE_MODEL=
# google/gemma-4-31B-it) is an instruct model: it gets the classifier's prompt
# without the Llama-2 [INST]/<<SYS>> markup as a chat message, a few tokens to
# answer in, and its first word is read as the label.
CHAT_JUDGE_MAX_TOKENS = 5
_LLAMA2_MARKUP = ("[INST] <<SYS>>\n", "<</SYS>>\n", " [/INST]", "[/INST]")


def uses_classifier(model: str) -> bool:
    return "harmbench" in model.lower()


def as_chat_prompt(prompt: str) -> str:
    for mark in _LLAMA2_MARKUP:
        prompt = prompt.replace(mark, "")
    return prompt.strip()


def parse_label(text: str) -> str | None:
    words = text.strip().lower().split()
    if not words:
        return None
    first = words[0].strip(".,:;!\"'*`")
    return first if first in {"yes", "no"} else None


_RETRYABLE_STATUS_CODES = {408, 425, 429}
_MAX_ERROR_BODY_CHARS = 2_000


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500


def request_completion(
    prompt: str,
    *,
    api_url: str,
    api_key: str | None,
    model: str,
    prompt_tokens: int,
    temperature: float = 0.0,
    max_tokens: int = 1,
    max_retries: int = 6,
    chat: bool = False,
) -> str | None:
    """Request a completion, retrying only transient endpoint failures.

    Raw by default (the HarmBench classifier's own prompt format); ``chat`` sends the
    prompt as one user message to /chat/completions, for a general instruct judge.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens}
    if chat:
        payload["messages"] = [{"role": "user", "content": prompt}]
        endpoint = f"{api_url.rstrip('/')}/chat/completions"
    else:
        payload["prompt"] = prompt
        endpoint = f"{api_url.rstrip('/')}/completions"

    for attempt in range(1, max_retries + 1):
        try:
            acquire_judge_rate_limit(f"{api_url}:{model}")
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=2000,
            )
        except requests.RequestException as exc:
            eval_logger.warning(
                "HarmBench judge request %d/%d failed for model %s "
                "(%d prompt tokens): %s",
                attempt,
                max_retries,
                model,
                prompt_tokens,
                exc,
            )
        except Exception:
            eval_logger.exception(
                "Unexpected HarmBench judge failure for model %s (%d prompt tokens)",
                model,
                prompt_tokens,
            )
            return None
        else:
            if response.ok:
                try:
                    choice = response.json()["choices"][0]
                    return choice["message"]["content"] if chat else choice["text"]
                except (KeyError, TypeError, ValueError, IndexError):
                    eval_logger.exception(
                        "HarmBench judge returned an invalid success response "
                        "for model %s: %s",
                        model,
                        response.text[:_MAX_ERROR_BODY_CHARS],
                    )
                    return None

            response_body = response.text[:_MAX_ERROR_BODY_CHARS]
            eval_logger.error(
                "HarmBench judge request %d/%d failed for model %s "
                "(%d prompt tokens, max_tokens=%d): HTTP %d: %s",
                attempt,
                max_retries,
                model,
                prompt_tokens,
                max_tokens,
                response.status_code,
                response_body,
            )
            if not _is_retryable_status(response.status_code):
                return None

        if attempt < max_retries:
            wait_seconds = min(2 ** (attempt - 1), 15)
            eval_logger.info(
                "Retrying HarmBench judge request in %d seconds", wait_seconds
            )
            time.sleep(wait_seconds)

    eval_logger.error(
        "HarmBench judge failed after %d attempts for model %s (%d prompt tokens)",
        max_retries,
        model,
        prompt_tokens,
    )
    return None
