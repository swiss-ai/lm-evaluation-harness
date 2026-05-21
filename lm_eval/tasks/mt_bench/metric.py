from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any


eval_logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
JUDGE_PROMPT_FILE = DATA_DIR / "judge_prompts.jsonl"
DEFAULT_CSCS_API_BASE = "https://api.swissai.svc.cscs.ch/v1"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.5-27B"
NEED_REF_CATS = {"math", "reasoning", "coding", "arena-hard-200"}

ONE_SCORE_PATTERN = re.compile(r"\[\[(\d+\.?\d*)\]\]")
ONE_SCORE_PATTERN_BACKUP = re.compile(r"\[(\d+\.?\d*)\]")

_JUDGE_CACHE: dict[str, list[dict[str, Any]]] = {}


def _load_judge_prompts() -> dict[str, dict[str, Any]]:
    prompts = {}
    with JUDGE_PROMPT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompt = json.loads(line)
                prompts[prompt["name"]] = prompt
    return prompts


def _judge_model() -> str:
    return os.getenv("MT_BENCH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def _judge_api_key() -> str:
    api_key = (
        os.getenv("MT_BENCH_JUDGE_API_KEY")
        or os.getenv("CSCS_SERVING_API")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise OSError(
            "Set MT_BENCH_JUDGE_API_KEY, CSCS_SERVING_API, or OPENAI_API_KEY "
            "to use MT-Bench LLM judging."
        )
    return api_key


def _judge_api_base() -> str | None:
    if os.getenv("MT_BENCH_JUDGE_API_BASE"):
        return os.getenv("MT_BENCH_JUDGE_API_BASE")
    if os.getenv("OPENAI_BASE_URL"):
        return os.getenv("OPENAI_BASE_URL")
    if os.getenv("CSCS_SERVING_API"):
        return DEFAULT_CSCS_API_BASE
    return None


def _judge_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the openai package to use mt_bench.") from exc

    base_url = _judge_api_base()
    kwargs = {"api_key": _judge_api_key()}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _prompt_name(item: dict[str, Any], turn: int) -> str:
    use_ref = item["category"] in NEED_REF_CATS
    if turn == 1:
        return "single-math-v1" if use_ref else "single-v1"
    return "single-math-v1-multi-turn" if use_ref else "single-v1-multi-turn"


def _reference(item: dict[str, Any], idx: int) -> str:
    reference = item.get("reference") or []
    return reference[idx] if idx < len(reference) else ""


def _build_judge_prompt(
    item: dict[str, Any], turn: int, prompts: dict[str, dict[str, Any]]
) -> tuple[str, str, str]:
    prompt_name = _prompt_name(item, turn)
    prompt = prompts[prompt_name]
    turns = item["turns"]
    responses = item["responses"]
    answer_1 = responses[0] if responses else ""
    answer_2 = responses[1] if len(responses) > 1 else ""

    if turn == 1:
        kwargs = {
            "question": turns[0],
            "answer": answer_1,
            "ref_answer_1": _reference(item, 0),
        }
    else:
        kwargs = {
            "question_1": turns[0],
            "question_2": turns[1],
            "answer_1": answer_1,
            "answer_2": answer_2,
            "ref_answer_1": _reference(item, 0),
            "ref_answer_2": _reference(item, 1),
        }
    return (
        prompt["system_prompt"],
        prompt["prompt_template"].format(**kwargs),
        prompt_name,
    )


def _parse_score(judgment: str) -> float:
    match = re.search(ONE_SCORE_PATTERN, judgment)
    if not match:
        match = re.search(ONE_SCORE_PATTERN_BACKUP, judgment)
    if not match:
        return math.nan
    score = float(match.group(1))
    return score if 1.0 <= score <= 10.0 else math.nan


def _judge_one(
    client: Any,
    model: str,
    item: dict[str, Any],
    turn: int,
    prompts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    system_prompt, user_prompt, prompt_name = _build_judge_prompt(item, turn, prompts)
    content = ""
    error = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=int(os.getenv("MT_BENCH_JUDGE_MAX_TOKENS", "2048")),
        )
        content = response.choices[0].message.content or ""
        score = _parse_score(content)
    except Exception as exc:  # noqa: BLE001 - failed judge calls are logged and skipped.
        score = math.nan
        error = str(exc)

    return {
        "question_id": item["question_id"],
        "category": item["category"],
        "turn": turn,
        "judge_model": model,
        "judge_prompt": prompt_name,
        "user_prompt": user_prompt,
        "judgment": content,
        "score": score,
        "judge_error": error,
        "tstamp": time.time(),
    }


def _items_cache_key(items: list[dict[str, Any]]) -> str:
    payload = {
        "model": _judge_model(),
        "items": [
            {
                "question_id": item["question_id"],
                "category": item["category"],
                "turns": item["turns"],
                "reference": item.get("reference"),
                "responses": item["responses"],
            }
            for item in items
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _append_judge_log(rows: list[dict[str, Any]]) -> None:
    path = os.getenv("MT_BENCH_JUDGE_LOG_PATH")
    if not path:
        return
    log_path = Path(path)
    if log_path.parent != Path("."):
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_judging(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []

    cache_key = _items_cache_key(items)
    if cache_key in _JUDGE_CACHE:
        return _JUDGE_CACHE[cache_key]

    prompts = _load_judge_prompts()
    model = _judge_model()
    client = _judge_client()
    max_workers = int(os.getenv("MT_BENCH_JUDGE_MAX_WORKERS", "4"))

    eval_logger.info(
        "Running MT-Bench judging on %d questions with %s (%d workers)",
        len(items),
        model,
        max_workers,
    )

    tasks = [(item, turn) for item in items for turn in (1, 2)]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_judge_one, client, model, item, turn, prompts): (
                item,
                turn,
            )
            for item, turn in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: (row["question_id"], row["turn"]))
    _append_judge_log(rows)
    _JUDGE_CACHE[cache_key] = rows
    return rows


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def mt_bench_score(items: list[dict[str, Any]]) -> float:
    rows = _run_judging(items)
    return _nanmean([row["score"] for row in rows])


def mt_bench_turn_1_score(items: list[dict[str, Any]]) -> float:
    rows = _run_judging(items)
    return _nanmean([row["score"] for row in rows if row["turn"] == 1])


def mt_bench_turn_2_score(items: list[dict[str, Any]]) -> float:
    rows = _run_judging(items)
    return _nanmean([row["score"] for row in rows if row["turn"] == 2])


def mt_bench_judge_success_rate(items: list[dict[str, Any]]) -> float:
    rows = _run_judging(items)
    if not rows:
        return float("nan")
    return sum(not math.isnan(row["score"]) for row in rows) / len(rows)
