"""Explicit, lazily connected attacker service. No calls during task discovery."""

import os

from lm_eval.tasks.polar_bench.task import final_answer


class Attacker:
    def __init__(self, model):
        from openai import OpenAI

        url = os.getenv("POLAR_ATTACKER_BASE_URL")
        if not url:
            raise ValueError("Protocol 5 requires POLAR_ATTACKER_BASE_URL")
        self.model = model
        self.client = OpenAI(
            base_url=url,
            api_key=os.getenv("POLAR_ATTACKER_API_KEY", "EMPTY"),
            timeout=120,
            max_retries=2,
        )

    def chat(self, system_prompt, messages, max_tokens, seed):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                "Attacker reached its 500-token limit; attack was not silently accepted"
            )
        text, status = final_answer(
            choice.message.content, {}, "final_only", "<think>", "</think>"
        )
        if status != "final":
            raise RuntimeError("Attacker produced no complete visible attack")
        return text
