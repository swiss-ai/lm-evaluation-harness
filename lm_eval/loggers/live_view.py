"""A live view of a running API evaluation: its first requests and how many are done.

lm-eval writes its sample files only after the whole evaluation has returned, so a
long run shows nothing until the end. An API model records, as each generation
returns, into two small files next to the run's --output_path (set by the CLI,
or by LM_EVAL_LIVE_VIEW):

  live_view.jsonl           the first KEEP requests: prompt, answer, reasoning, tokens
  live_view_progress.json   how many generation requests have finished, and when

A reader outside the run (the evals service's status writer) picks them up while the
run goes on. Nothing here may disturb the evaluation: every failure is swallowed.
Not named samples_*.jsonl on purpose: the result collectors glob for that.
"""

from __future__ import annotations

import json
import logging
import os
import time


eval_logger = logging.getLogger(__name__)

ENV = "LM_EVAL_LIVE_VIEW"
KEEP = 20  # requests kept in full
TEXT_CAP = 6000  # characters of one prompt / answer
REASONING_CAP = 20000
PROGRESS_EVERY_SECONDS = 2.0

_PROGRESS_NAME = "live_view_progress.json"
_state: dict[str, dict] = {}
_warned = False


def default_path(output_path: str | None) -> str | None:
    """Where the live view goes for a run with this --output_path: a file beside it."""
    if not output_path:
        return None
    directory = output_path
    if directory.endswith((".json", ".jsonl")):
        directory = os.path.dirname(directory) or "."
    return os.path.join(directory, "live_view.jsonl")


def _cap(text: str, cap: int) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n[... {len(text) - cap} more characters]"


def _write_progress(path: str, done: int) -> None:
    progress_path = os.path.join(os.path.dirname(path), _PROGRESS_NAME)
    temporary = progress_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump({"done": done, "updated": int(time.time())}, handle)
    os.replace(temporary, progress_path)


def record(
    prompt: str,
    answer: str,
    reasoning: str = "",
    output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> None:
    """One finished generation. A no-op unless the live view is switched on."""
    global _warned
    path = os.environ.get(ENV)
    if not path:
        return
    try:
        state = _state.setdefault(path, {"done": 0, "written": 0.0})
        state["done"] += 1
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if state["done"] <= KEEP:
            line = {
                "n": state["done"],
                "prompt": _cap(prompt, TEXT_CAP),
                "answer": _cap(answer, TEXT_CAP),
                "reasoning": _cap(reasoning, REASONING_CAP),
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
            with open(path, "a") as handle:
                handle.write(json.dumps(line) + "\n")
        now = time.monotonic()
        if state["done"] <= KEEP or now - state["written"] >= PROGRESS_EVERY_SECONDS:
            state["written"] = now
            _write_progress(path, state["done"])
    except Exception as exc:  # noqa: BLE001 - never disturb the evaluation
        if not _warned:
            _warned = True
            eval_logger.warning(
                "live view not written (%s: %s)", type(exc).__name__, exc
            )


def finish() -> None:
    """The last count, which the throttle may have held back."""
    path = os.environ.get(ENV)
    state = _state.get(path or "")
    if not path or not state:
        return
    try:
        _write_progress(path, state["done"])
    except Exception as exc:  # noqa: BLE001 - never disturb the evaluation
        eval_logger.debug("live view progress not written: %s", exc)
