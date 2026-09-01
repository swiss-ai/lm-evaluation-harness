"""Remote completion client for the HarmBench classifier."""

from __future__ import annotations

import logging
import time

import requests

from lm_eval.api.rate_limiter import acquire_judge_rate_limit


eval_logger = logging.getLogger(__name__)

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
) -> str | None:
    """Request a raw completion, retrying only transient endpoint failures."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
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
                    return response.json()["choices"][0]["text"]
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
