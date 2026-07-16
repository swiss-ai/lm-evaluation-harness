#!/usr/bin/env python3
"""Generate Swiss AI Charter reference completions with an OpenAI-compatible model.

Example:
    OPENAI_API_KEY=... python scripts/generate_swiss_ai_charter_references.py \
        --model gpt-4.1-mini \
        --output swiss_ai_charter_references.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "lm_eval" / "tasks" / "swiss_ai_charter_alignment" / "utils.py"

utils: Any | None = None


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's prompt directly while following "
    "the Swiss AI Charter guidance supplied in the conversation."
)


def get_task_utils() -> Any:
    global utils
    if utils is not None:
        return utils

    spec = importlib.util.spec_from_file_location("swiss_ai_charter_utils", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Swiss AI Charter utils from {UTILS_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["swiss_ai_charter_utils"] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise ImportError(
            "Install the lm-eval harness dependencies, including datasets and parquet "
            "support, before generating Swiss AI Charter references."
        ) from exc
    utils = module
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference answers for Swiss AI Charter Alignment."
    )
    parser.add_argument("--model", required=True, help="OpenAI-compatible chat model.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--subset",
        default="full",
        choices=["full", "prism", "wildchat", "synthetic"],
        help="Prompt subset to generate. Defaults to the joined 660-prompt dataset.",
    )
    parser.add_argument(
        "--prompts-path",
        default=None,
        help="Optional parquet override for prompts.",
    )
    parser.add_argument(
        "--charter-path",
        default=None,
        help="Optional charter JSON override.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help=(
            "Environment variable containing the API key. Defaults to "
            "OPENAI_API_KEY, then CSCS_SERVING_API."
        ),
    )
    parser.add_argument("--api-key", default=None, help="API key value.")
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL, or the "
            "SwissAI endpoint when CSCS_SERVING_API is used."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=800)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of prompts to generate, useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load prompts and print the generation count without calling the API.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompt_id values already present in the output JSONL.",
    )
    parser.add_argument(
        "--charter-context",
        choices=["none", "target_article", "full"],
        default="target_article",
        help="How much charter text to include in each reference-generation prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent to the reference model.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send SwissAI chat_template_kwargs to disable thinking traces.",
    )
    return parser.parse_args()


def openai_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the openai package to generate references.") from exc

    key_env = args.api_key_env
    api_key = args.api_key
    if not api_key and key_env:
        api_key = os.getenv(key_env)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CSCS_SERVING_API")
    if not api_key:
        raise OSError("Set OPENAI_API_KEY, CSCS_SERVING_API, or pass --api-key.")

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if (
        not base_url
        and os.getenv("CSCS_SERVING_API")
        and not os.getenv("OPENAI_API_KEY")
    ):
        base_url = get_task_utils().DEFAULT_API_BASE

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def load_docs(args: argparse.Namespace) -> list[dict[str, Any]]:
    task_utils = get_task_utils()
    dataset = task_utils.create_dataset(
        subset=args.subset,
        prompts_path=args.prompts_path,
        charter_path=args.charter_path,
    )
    return list(dataset["test"])


def charter_text(doc: dict[str, Any], args: argparse.Namespace) -> str:
    if args.charter_context == "none":
        return ""

    articles = get_task_utils()._load_charter(args.charter_path)  # noqa: SLF001
    if args.charter_context == "target_article":
        return articles[str(doc["article_id"])]["text"]

    return "\n\n".join(article["text"] for article in articles.values())


def render_user_message(doc: dict[str, Any], args: argparse.Namespace) -> str:
    context = charter_text(doc, args)
    prompt = str(doc["prompt"]).strip()
    if not context:
        return f"Respond to the following user prompt:\n\n{prompt}"
    return (
        "Swiss AI Charter guidance:\n"
        f"{context}\n\n"
        "Respond to the following user prompt:\n\n"
        f"{prompt}"
    )


def existing_prompt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_id = row.get("prompt_id")
            if prompt_id:
                ids.add(str(prompt_id))
    return ids


def should_send_swissai_extra_body(args: argparse.Namespace) -> bool:
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or ""
    if (
        not base_url
        and os.getenv("CSCS_SERVING_API")
        and not os.getenv("OPENAI_API_KEY")
    ):
        base_url = get_task_utils().DEFAULT_API_BASE
    return args.disable_thinking or "swissai" in base_url


def generate_one(
    client: Any, doc: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": render_user_message(doc, args)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if should_send_swissai_extra_body(args):
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    started = time.time()
    response = client.chat.completions.create(**request)
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    completion = (getattr(message, "content", None) or "").strip()

    return {
        "prompt_id": str(doc["prompt_id"]),
        "article_id": str(doc["article_id"]),
        "article_title": str(doc.get("article_title", "")),
        "subset": str(doc.get("subset", "")),
        "source_id": str(doc.get("source_id", "")),
        "source_dataset": str(doc.get("source_dataset", "")),
        "prompt": str(doc["prompt"]),
        "reference_completion": get_task_utils()._strip_thinking_traces(  # noqa: SLF001
            completion
        ),
        "reference_model": args.model,
        "reference_charter_context": args.charter_context,
        "latency_s": round(time.time() - started, 3),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs = load_docs(args)
    if args.limit is not None:
        docs = docs[: args.limit]
    if args.resume:
        done = existing_prompt_ids(output_path)
        docs = [doc for doc in docs if str(doc["prompt_id"]) not in done]

    print(f"Generating {len(docs)} reference completions with {args.model}")
    if args.dry_run:
        return
    if not docs:
        return

    client = openai_client(args)
    with (
        output_path.open("a", encoding="utf-8") as handle,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        future_to_doc = {
            executor.submit(generate_one, client, doc, args): doc for doc in docs
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_doc)):
            row = future.result()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed = idx + 1
            if completed % 25 == 0 or completed == len(docs):
                print(f"Completed {completed}/{len(docs)}")


if __name__ == "__main__":
    main()
