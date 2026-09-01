"""Prompt-length helpers for the HarmBench classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Callable


class Tokenizer(Protocol):
    """Subset of the Hugging Face tokenizer API used by HarmBench."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...


def prepare_prompt(
    completion: str,
    *,
    build_prompt: Callable[[str], str],
    tokenizer: Tokenizer,
    max_completion_tokens: int,
    max_context_tokens: int,
    max_new_tokens: int,
) -> tuple[str, int, int, int]:
    """Build a prompt after truncating its completion to the available budget.

    The beginning of the completion is retained, as in the HarmBench reference
    implementation. The return value contains the prompt, original completion
    token count, retained completion token count, and final prompt token count.
    """

    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    original_completion_tokens = len(completion_tokens)
    capped_completion_token_count = min(
        original_completion_tokens, max_completion_tokens
    )
    if capped_completion_token_count == original_completion_tokens:
        capped_completion = completion
    else:
        capped_completion = tokenizer.decode(
            completion_tokens[:capped_completion_token_count],
            skip_special_tokens=True,
        )

    capped_prompt = build_prompt(capped_completion)
    capped_prompt_tokens = tokenizer.encode(capped_prompt, add_special_tokens=False)
    if len(capped_prompt_tokens) + max_new_tokens <= max_context_tokens:
        return (
            capped_prompt,
            original_completion_tokens,
            capped_completion_token_count,
            len(capped_prompt_tokens),
        )

    # Find the longest token prefix whose fully rendered prompt fits. Measuring
    # the rendered prompt avoids assumptions about token boundaries where the
    # completion is inserted into the classifier template.
    lower_bound = 0
    upper_bound = capped_completion_token_count
    retained_token_count = 0
    prompt = build_prompt("")
    prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))

    while lower_bound <= upper_bound:
        candidate_token_count = (lower_bound + upper_bound) // 2
        prepared_completion = tokenizer.decode(
            completion_tokens[:candidate_token_count], skip_special_tokens=True
        )
        candidate_prompt = build_prompt(prepared_completion)
        candidate_prompt_token_count = len(
            tokenizer.encode(candidate_prompt, add_special_tokens=False)
        )

        if candidate_prompt_token_count + max_new_tokens <= max_context_tokens:
            retained_token_count = candidate_token_count
            prompt = candidate_prompt
            prompt_token_count = candidate_prompt_token_count
            lower_bound = candidate_token_count + 1
        else:
            upper_bound = candidate_token_count - 1

    return (
        prompt,
        original_completion_tokens,
        retained_token_count,
        prompt_token_count,
    )
