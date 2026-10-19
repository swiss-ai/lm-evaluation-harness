"""Persist per-sample LLM-judge verdicts to a JSONL sidecar.

Judge-based tasks usually defer their judge calls to the aggregation function,
so the verdict for an individual sample never reaches ``--log_samples``. This
module writes one JSON line per judged sample so that results can be joined
back to prompts/responses afterwards (e.g. to search the pretraining index).

Records go to ``$LM_EVAL_JUDGE_LOG_DIR/<task>.jsonl`` (default:
``./judge_logs``). Writing is thread-safe and de-duplicated per process on
``(task, sample_id, response)`` so that several aggregation functions sharing
one cached judge call only log each sample once.

Common record fields (add task-specific ones freely):
    task, lang, sample_id, prompt, response, judge_model, judge_raw,
    label, score
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any


eval_logger = logging.getLogger(__name__)

JUDGE_LOG_DIR_ENV = "LM_EVAL_JUDGE_LOG_DIR"

_LOCK = threading.Lock()
_SEEN: set[tuple[str, str, str]] = set()


def judge_log_dir() -> str:
    return os.getenv(JUDGE_LOG_DIR_ENV) or os.path.join(os.getcwd(), "judge_logs")


def sample_id_for(text: str) -> str:
    """Stable id for a prompt when the dataset has no id column."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def log_judge_record(task: str, record: dict[str, Any]) -> None:
    """Append one judge record for ``task``; never raises."""
    try:
        sample_id = str(record.get("sample_id") or sample_id_for(str(record.get("prompt", ""))))
        key = (task, sample_id, str(record.get("response", "")))
        with _LOCK:
            if key in _SEEN:
                return
            _SEEN.add(key)
            out_dir = judge_log_dir()
            os.makedirs(out_dir, exist_ok=True)
            row = {"task": task, "sample_id": sample_id, "ts": time.time(), **record}
            with open(os.path.join(out_dir, f"{task}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        eval_logger.warning(f"judge_log: failed to write record for {task}: {e}")
