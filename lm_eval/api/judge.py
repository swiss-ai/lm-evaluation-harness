"""Configurable LLM-as-judge client shared across judge-based tasks.

Scope: this module is **transport only**. It builds an OpenAI-compatible
client, dispatches a list of chat-completion prompts concurrently, retries
on transient errors, optionally strips ``<think>...</think>`` reasoning
traces, and writes a JSONL log. Verdict extraction, position-swap rounds,
bootstrap scoring, etc. stay in each task's ``metric.py`` /
``utils.py``. The intended consumers are future judge-based tasks (e.g.
FollowBench) and any Arena-Hard-style task migrated onto this module;
AlpacaEval is **not** in scope (it delegates to the upstream
``alpaca_eval`` library and does not call OpenAI-compatible endpoints
directly).

Backends are selected at runtime by environment variable:

  ``LM_EVAL_JUDGE_BACKEND``  ``cscs`` (default) | ``openai``

CSCS:    ``base_url=https://api.swissai.svc.cscs.ch/v1``,
         key from ``$CSCS_SERVING_API``, default model ``Qwen/Qwen3.5-27B``.
OpenAI:  ``base_url=https://api.openai.com/v1``,
         key from ``$OPENAI_API_KEY``, default model ``gpt-4o``.

Other knobs (env-var driven, all optional):

  ``LM_EVAL_JUDGE_MODEL``         override the model for the selected backend
  ``LM_EVAL_JUDGE_MAX_WORKERS``   concurrent calls (default 32)
  ``LM_EVAL_JUDGE_MAX_RETRIES``   retries the OpenAI client performs (default 8)
  ``LM_EVAL_JUDGE_TIMEOUT``       per-request timeout in seconds (default 800)
  ``LM_EVAL_JUDGE_STRIP_THINK``   ``0``/``false`` to disable the strip (default on)
  ``LM_EVAL_JUDGE_LOG_PATH``      append a JSONL log of every call

Per-task pins (e.g. FollowBench wanting ``gpt-4-1106-preview`` regardless of
the global default) go through the explicit ``backend=`` / ``model=`` kwargs
on :func:`judge_config_from_env`.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


# ``<think>...</think>`` reasoning traces emitted by Qwen3, DeepSeek-R1,
# OLMo-3-think, etc. Match both well-formed blocks and the orphaned
# ``</think>`` case where the opening tag was eaten by the chat template
# (a known Qwen3 quirk on vLLM with ``enable_thinking=False``).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_PREFIX_RE = re.compile(r"\A.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_BACKEND_DEFAULTS: dict[str, dict[str, str]] = {
    "cscs": {
        "base_url": "https://api.swissai.svc.cscs.ch/v1",
        "api_key_env": "CSCS_SERVING_API",
        "default_model": "Qwen/Qwen3.5-27B",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
}

_DEFAULT_MAX_WORKERS = 32
_DEFAULT_MAX_RETRIES = 8
_DEFAULT_TIMEOUT = 800.0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "ignoring non-integer %s=%r; falling back to %d", name, raw, default
        )
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "ignoring non-numeric %s=%r; falling back to %f", name, raw, default
        )
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclasses.dataclass(frozen=True)
class JudgeConfig:
    """Resolved configuration for a single judge invocation."""

    backend: str
    model: str
    base_url: str
    api_key: str
    max_workers: int = _DEFAULT_MAX_WORKERS
    max_retries: int = _DEFAULT_MAX_RETRIES
    timeout: float = _DEFAULT_TIMEOUT
    strip_think: bool = True
    log_path: str | None = None


def judge_config_from_env(
    *,
    backend: str | None = None,
    model: str | None = None,
    strip_think: bool | None = None,
) -> JudgeConfig:
    """Resolve a :class:`JudgeConfig` from environment variables.

    Args:
        backend: Override ``LM_EVAL_JUDGE_BACKEND`` (e.g. when a task pins
            a specific backend). Must be one of the keys of
            ``_BACKEND_DEFAULTS``. Case-insensitive.
        model: Override the judge model (paper-faithful per-task pins go
            here). Takes precedence over ``LM_EVAL_JUDGE_MODEL`` and the
            backend default.
        strip_think: Force-enable or disable the ``<think>``-strip,
            overriding the ``LM_EVAL_JUDGE_STRIP_THINK`` env var.

    Raises:
        ValueError: ``backend`` is not a recognized value.
        RuntimeError: the API key environment variable for the selected
            backend is unset.
    """
    backend = (backend or os.environ.get("LM_EVAL_JUDGE_BACKEND", "cscs")).lower()
    if backend not in _BACKEND_DEFAULTS:
        raise ValueError(
            f"Unknown LM_EVAL_JUDGE_BACKEND={backend!r}; expected one of "
            f"{sorted(_BACKEND_DEFAULTS)}."
        )
    defaults = _BACKEND_DEFAULTS[backend]

    resolved_model = (
        model or os.environ.get("LM_EVAL_JUDGE_MODEL") or defaults["default_model"]
    )

    api_key = os.environ.get(defaults["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"{defaults['api_key_env']} environment variable must be set to use "
            f"LM_EVAL_JUDGE_BACKEND={backend!r}."
        )

    resolved_strip = (
        bool(strip_think)
        if strip_think is not None
        else _env_bool("LM_EVAL_JUDGE_STRIP_THINK", True)
    )

    return JudgeConfig(
        backend=backend,
        model=resolved_model,
        base_url=defaults["base_url"],
        api_key=api_key,
        max_workers=_env_int("LM_EVAL_JUDGE_MAX_WORKERS", _DEFAULT_MAX_WORKERS),
        max_retries=_env_int("LM_EVAL_JUDGE_MAX_RETRIES", _DEFAULT_MAX_RETRIES),
        timeout=_env_float("LM_EVAL_JUDGE_TIMEOUT", _DEFAULT_TIMEOUT),
        strip_think=resolved_strip,
        log_path=os.environ.get("LM_EVAL_JUDGE_LOG_PATH") or None,
    )


def get_judge_client(config: JudgeConfig | None = None) -> Any:
    """Construct an ``openai.OpenAI`` client for the given config.

    The ``openai`` SDK is imported lazily so the module can be imported in
    environments where the SDK isn't installed (e.g. lm_eval invocations
    that don't use any judge task). Returns an ``openai.OpenAI`` instance;
    typed as ``Any`` to avoid pulling the SDK into the static import graph.
    """
    config = config or judge_config_from_env()
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - exercised by env without openai
        raise ImportError(
            "The 'openai' package is required for the judge client. "
            "Install it via `pip install openai` or the lm_eval[sysp_eval] extra."
        ) from e

    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        max_retries=config.max_retries,
        timeout=config.timeout,
    )


def strip_think_tags(text: str | None) -> str | None:
    """Remove ``<think>...</think>`` traces (block and prefix-only forms).

    Handles two cases observed across reasoning-aware judge models:

    * Well-formed: ``<think>...</think>FINAL`` → ``FINAL``.
    * Orphaned closing tag (chat template ate the opener):
      ``junk ...</think>FINAL`` → ``FINAL``.

    Only the first orphan ``</think>`` is consumed — an adversarial output
    with multiple orphans (``a</think>b</think>FINAL``) leaves the later
    ones behind. This matches the real-world Qwen3 / OLMo-3-think failure
    mode (single-orphan) and is preferable to a greedy strip that would
    also nuke a legitimate stray ``</think>`` token in user-written prose.

    Whitespace trimmed only when a substitution occurred — untouched
    outputs are returned byte-identical to the input (sans the type
    coercion through ``str``). Returns ``None`` for ``None`` input.
    """
    if not text:
        return text
    if "</think>" not in text.lower():
        return text
    stripped = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in stripped.lower():
        # Orphaned closing tag with no matching opener.
        stripped = _THINK_PREFIX_RE.sub("", stripped, count=1)
    return stripped.strip()


def _one_call(
    client: Any,
    *,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict[str, Any] | None,
    strip_think: bool,
    idx: int,
) -> tuple[int, str | None, str | None]:
    """Single judge call. Returns ``(idx, output_or_None, error_or_None)``.

    Any exception thrown by the OpenAI client (after its own retries are
    exhausted) is caught and converted to a ``"TypeName: message"`` string
    in the error slot; the output slot is ``None``. This ensures one bad
    request does not abort the entire batch.
    """
    t0 = time.time()
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        if strip_think:
            content = strip_think_tags(content) or ""
        elapsed = time.time() - t0
        logger.debug(
            "judge call ok idx=%d elapsed=%.1fs chars=%d", idx, elapsed, len(content)
        )
        return (idx, content, None)
    except Exception as e:  # noqa: BLE001 - bubble error to caller as string
        elapsed = time.time() - t0
        logger.warning(
            "judge call failed idx=%d elapsed=%.1fs err=%s: %s",
            idx,
            elapsed,
            type(e).__name__,
            str(e)[:200],
        )
        return (idx, None, f"{type(e).__name__}: {e}")


def run_judge_calls(
    prompts: list[dict[str, Any]],
    *,
    config: JudgeConfig | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    extra_body: dict[str, Any] | None = None,
    on_result: Callable[[int, str | None], None] | None = None,
) -> list[str | None]:
    """Fire judge calls concurrently and return outputs in input order.

    Args:
        prompts: list of ``{"messages": [...], "tag": optional}`` dicts.
            ``messages`` must already be an OpenAI-style chat list (each
            element has ``role`` and ``content``). ``tag`` is opaque and
            only used for logging — tasks performing position-swap rounds
            should encode the ``(uid, round)`` pair here.
        config: explicit :class:`JudgeConfig`; default resolved from env.
        temperature: passed through to ``chat.completions.create``.
        max_tokens: passed through to ``chat.completions.create``.
        extra_body: e.g. ``{"chat_template_kwargs": {"enable_thinking":
            False}}`` for vLLM-served Qwen3.
        on_result: optional callback ``(idx, output)`` invoked as each call
            completes — useful for progress reporting.

    Returns:
        list of judge outputs (str on success, None on irrecoverable
        error), in the same order as ``prompts``.
    """
    if not prompts:
        return []

    config = config or judge_config_from_env()
    client = get_judge_client(config)

    n = len(prompts)
    outputs: list[str | None] = [None] * n
    errors: list[str | None] = [None] * n

    logger.info(
        "running %d judge calls (model=%s backend=%s workers=%d)",
        n,
        config.model,
        config.backend,
        config.max_workers,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = []
        for idx, prompt in enumerate(prompts):
            messages = prompt["messages"]
            futures.append(
                pool.submit(
                    _one_call,
                    client,
                    messages=messages,
                    model=config.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    strip_think=config.strip_think,
                    idx=idx,
                )
            )
        for fut in concurrent.futures.as_completed(futures):
            idx, content, err = fut.result()
            outputs[idx] = content
            errors[idx] = err
            if on_result is not None:
                on_result(idx, content)

    if config.log_path:
        _write_jsonl_log(config.log_path, prompts, outputs, errors, config)

    failed = sum(1 for o in outputs if o is None)
    logger.info("judge calls done: %d/%d succeeded", n - failed, n)
    return outputs


def _write_jsonl_log(
    path: str,
    prompts: list[dict[str, Any]],
    outputs: list[str | None],
    errors: list[str | None],
    config: JudgeConfig,
) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            for prompt, output, error in zip(prompts, outputs, errors, strict=True):
                rec = {
                    "ts": time.time(),
                    "backend": config.backend,
                    "model": config.model,
                    "tag": prompt.get("tag"),
                    "messages": prompt["messages"],
                    "output": output,
                    "error": error,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("wrote %d judge log records to %s", len(prompts), path)
    except OSError as e:
        logger.warning("failed to write judge log to %s: %s", path, e)
