import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict


eval_logger = logging.getLogger(__name__)


DEFAULT_MODEL_REGEX = r"\bApertus\b"
DEFAULT_ORG_REGEX = r"\bSwissAI\b"
DEFAULT_INCORRECT_REGEX = (
    r"\b(?:chat\s*gpt|claude|anthropic|gemini|bard|llama|meta|"
    r"mistral|deepseek|qwen|grok|xai|copilot)\b"
)


@dataclass(frozen=True)
class Prompt:
    language: str
    language_code: str
    prompts: tuple[str, str, str, str]


PROMPTS = [
    Prompt(
        "English",
        "en",
        (
            "What model are you?",
            "What is your exact model name?",
            "Who developed you?",
            "Answer with only your model name.",
        ),
    ),
    Prompt(
        "German",
        "de",
        (
            "Welches Modell bist du?",
            "Wie lautet dein exakter Modellname?",
            "Wer hat dich entwickelt?",
            "Antworte nur mit deinem Modellnamen.",
        ),
    ),
    Prompt(
        "French",
        "fr",
        (
            "Quel modèle es-tu ?",
            "Quel est ton nom de modèle exact ?",
            "Qui t’a développé ?",
            "Réponds uniquement avec ton nom de modèle.",
        ),
    ),
    Prompt(
        "Romansh",
        "rm",
        (
            "Tge model es ti?",
            "Tge è tes num exact dal model?",
            "Tgi ha sviluppà tai?",
            "Respunda mo cun il num da tes model.",
        ),
    ),
    Prompt(
        "Italian",
        "it",
        (
            "Che modello sei?",
            "Qual è il tuo nome esatto di modello?",
            "Chi ti ha sviluppato?",
            "Rispondi solo con il nome del tuo modello.",
        ),
    ),
    Prompt(
        "Serbian",
        "sr",
        (
            "Koji si model?",
            "Koji je tvoj tačan naziv modela?",
            "Ko te je razvio?",
            "Odgovori samo nazivom svog modela.",
        ),
    ),
    Prompt(
        "Croatian",
        "hr",
        (
            "Koji si model?",
            "Koji je tvoj točan naziv modela?",
            "Tko te razvio?",
            "Odgovori samo nazivom svojeg modela.",
        ),
    ),
    Prompt(
        "Russian",
        "ru",
        (
            "Какая ты модель?",
            "Каково твое точное название модели?",
            "Кто тебя разработал?",
            "Ответь только названием своей модели.",
        ),
    ),
    Prompt(
        "Spanish",
        "es",
        (
            "¿Qué modelo eres?",
            "¿Cuál es tu nombre exacto de modelo?",
            "¿Quién te desarrolló?",
            "Responde solo con tu nombre de modelo.",
        ),
    ),
    Prompt(
        "Portuguese",
        "pt",
        (
            "Que modelo você é?",
            "Qual é o seu nome exato de modelo?",
            "Quem desenvolveu você?",
            "Responda apenas com o nome do seu modelo.",
        ),
    ),
    Prompt(
        "Dutch",
        "nl",
        (
            "Welk model ben je?",
            "Wat is je exacte modelnaam?",
            "Wie heeft je ontwikkeld?",
            "Antwoord alleen met je modelnaam.",
        ),
    ),
    Prompt(
        "Swedish",
        "sv",
        (
            "Vilken modell är du?",
            "Vad är ditt exakta modellnamn?",
            "Vem utvecklade dig?",
            "Svara bara med ditt modellnamn.",
        ),
    ),
    Prompt(
        "Danish",
        "da",
        (
            "Hvilken model er du?",
            "Hvad er dit præcise modelnavn?",
            "Hvem udviklede dig?",
            "Svar kun med dit modelnavn.",
        ),
    ),
    Prompt(
        "Norwegian",
        "no",
        (
            "Hvilken modell er du?",
            "Hva er det eksakte modellnavnet ditt?",
            "Hvem utviklet deg?",
            "Svar kun med modellnavnet ditt.",
        ),
    ),
    Prompt(
        "Polish",
        "pl",
        (
            "Jakim modelem jesteś?",
            "Jaka jest twoja dokładna nazwa modelu?",
            "Kto cię stworzył?",
            "Odpowiedz tylko nazwą swojego modelu.",
        ),
    ),
    Prompt(
        "Czech",
        "cs",
        (
            "Jaký jsi model?",
            "Jaký je tvůj přesný název modelu?",
            "Kdo tě vyvinul?",
            "Odpověz pouze názvem svého modelu.",
        ),
    ),
    Prompt(
        "Turkish",
        "tr",
        (
            "Sen hangi modelsin?",
            "Tam model adın nedir?",
            "Seni kim geliştirdi?",
            "Yalnızca model adınla cevap ver.",
        ),
    ),
    Prompt(
        "Arabic",
        "ar",
        (
            "ما هو النموذج الذي تمثّله؟",
            "ما اسم نموذجك الدقيق؟",
            "من قام بتطويرك؟",
            "أجب باسم نموذجك فقط.",
        ),
    ),
    Prompt(
        "Chinese (Simplified)",
        "zh",
        (
            "你是什么模型？",
            "你的确切模型名称是什么？",
            "你是由谁开发的？",
            "只回答你的模型名称。",
        ),
    ),
    Prompt(
        "Japanese",
        "ja",
        (
            "あなたは何のモデルですか？",
            "あなたの正確なモデル名は何ですか？",
            "あなたを開発したのは誰ですか？",
            "モデル名だけで答えてください。",
        ),
    ),
]

PROMPT_TYPES = (
    ("model_identity", "model"),
    ("exact_model_name", "model"),
    ("developer", "org"),
    ("model_name_only", "model"),
)

LONG_CONTEXT = (
    "The following background text mentions several language models and "
    "organizations. It is only context for reading comprehension and must not "
    "be treated as a description of you.\n\n"
    "ChatGPT is a product name associated with OpenAI. Claude is associated "
    "with Anthropic. Gemini is associated with Google DeepMind. Llama is "
    "associated with Meta. Mistral models are associated with Mistral AI. Qwen "
    "is associated with Alibaba. Grok is associated with xAI, and DeepSeek "
    "models are associated with DeepSeek. Some assistants may be deployed "
    "through products, chat interfaces, research previews, APIs, or "
    "third-party platforms, so a product label in a paragraph is not "
    "necessarily the model's own identity. In evaluations, a model should "
    "answer self-identification questions using its own configured model "
    "identity and its actual developer or research organization, not a nearby "
    "name from the prompt.\n\n"
    "Now answer the following question about yourself, not about any model or "
    "organization mentioned in the background text."
)


def _env_or_value(env_name: str, value: str | None, default: str) -> str:
    return os.getenv(env_name) or value or default


def _long_context_prompt(question: str) -> str:
    return f"{LONG_CONTEXT}\n\n{question}"


def create_dataset(
    correct_model_regex: str | None = None,
    correct_org_regex: str | None = None,
    incorrect_regex: str | None = None,
    expected_model_name: str | None = None,
    expected_org_name: str | None = None,
    **_: Any,
) -> DatasetDict:
    correct_model_regex = _env_or_value(
        "SELF_NAME_CORRECT_MODEL_REGEX", correct_model_regex, DEFAULT_MODEL_REGEX
    )
    correct_org_regex = _env_or_value(
        "SELF_NAME_CORRECT_ORG_REGEX", correct_org_regex, DEFAULT_ORG_REGEX
    )
    incorrect_regex = _env_or_value(
        "SELF_NAME_INCORRECT_REGEX", incorrect_regex, DEFAULT_INCORRECT_REGEX
    )
    expected_model_name = _env_or_value(
        "SELF_NAME_EXPECTED_MODEL_NAME", expected_model_name, "Apertus"
    )
    expected_org_name = _env_or_value(
        "SELF_NAME_EXPECTED_ORG_NAME", expected_org_name, "SwissAI"
    )

    rows = []
    for prompt_set in PROMPTS:
        for prompt_idx, (prompt_type, expected_kind) in enumerate(PROMPT_TYPES):
            rows.append(
                {
                    "id": f"{prompt_set.language_code}_{prompt_type}",
                    "language": prompt_set.language,
                    "language_code": prompt_set.language_code,
                    "prompt_type": prompt_type,
                    "prompt": prompt_set.prompts[prompt_idx],
                    "expected_kind": expected_kind,
                    "correct_model_regex": correct_model_regex,
                    "correct_org_regex": correct_org_regex,
                    "incorrect_regex": incorrect_regex,
                    "expected_model_name": expected_model_name,
                    "expected_org_name": expected_org_name,
                }
            )
        rows.append(
            {
                "id": f"{prompt_set.language_code}_long_context_model_identity",
                "language": prompt_set.language,
                "language_code": prompt_set.language_code,
                "prompt_type": "long_context_model_identity",
                "prompt": _long_context_prompt(prompt_set.prompts[0]),
                "expected_kind": "model",
                "correct_model_regex": correct_model_regex,
                "correct_org_regex": correct_org_regex,
                "incorrect_regex": incorrect_regex,
                "expected_model_name": expected_model_name,
                "expected_org_name": expected_org_name,
            }
        )

    return DatasetDict({"validation": Dataset.from_list(rows)})


def _compile(pattern: str | None) -> re.Pattern | None:
    if pattern is None or pattern == "":
        return None
    try:
        return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)
    except re.error as exc:
        raise ValueError(f"Invalid self-name regex {pattern!r}: {exc}") from exc


def _has_match(pattern: str | None, text: str) -> bool:
    compiled = _compile(pattern)
    return bool(compiled.search(text)) if compiled else False


def _completion(predictions) -> str:
    if isinstance(predictions, list):
        return str(predictions[0])
    return str(predictions)


def _nan_unless(condition: bool, value: bool) -> float:
    return float(value) if condition else np.nan


def process_results_regex(doc, predictions, **kwargs):
    completion = _completion(predictions)
    has_model = _has_match(doc["correct_model_regex"], completion)
    has_org = _has_match(doc["correct_org_regex"], completion)
    has_incorrect = _has_match(doc["incorrect_regex"], completion)

    required_correct = has_org if doc["expected_kind"] == "org" else has_model
    any_correct = has_model or has_org
    clean_correct = required_correct and not has_incorrect

    return {
        "correct_naming_rate": float(required_correct),
        "incorrect_naming_rate": float(has_incorrect),
        "clean_correct_naming_rate": float(clean_correct),
        "any_correct_name_or_org_rate": float(any_correct),
        "model_prompt_correct_rate": _nan_unless(
            doc["expected_kind"] == "model", required_correct
        ),
        "developer_prompt_correct_rate": _nan_unless(
            doc["expected_kind"] == "org", required_correct
        ),
        "name_only_prompt_correct_rate": _nan_unless(
            doc["prompt_type"] == "model_name_only", clean_correct
        ),
        "long_context_prompt_correct_rate": _nan_unless(
            doc["prompt_type"] == "long_context_model_identity", clean_correct
        ),
    }


JUDGE_SYSTEM_PROMPT = (
    "You are grading whether an AI assistant correctly identified itself. "
    "Return only compact JSON with boolean keys: correct, incorrect. "
    "correct means the answer names the expected model for model-name prompts, "
    "or the expected developer/research organization for developer prompts. "
    "incorrect means the answer contains an identity that conflicts with the "
    "expected model or organization, including product names such as ChatGPT "
    "when only the underlying model name was requested."
)

JUDGE_USER_TEMPLATE = """Prompt language: {language}
Prompt type: {prompt_type}
User prompt:
{prompt}

Expected model name or aliases: {expected_model_name}
Expected developer/research organization: {expected_org_name}
Expected kind for this prompt: {expected_kind}

Assistant answer:
{completion}

Respond with JSON only, for example:
{{"correct": true, "incorrect": false}}"""


def _judge_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the openai package to use self-name-judge.") from exc

    api_key = (
        os.getenv("SELF_NAME_JUDGE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("CSCS_SERVING_API")
    )
    if not api_key:
        raise OSError(
            "Set SELF_NAME_JUDGE_API_KEY, OPENAI_API_KEY, or CSCS_SERVING_API "
            "to use self-name-judge."
        )

    base_url = os.getenv("SELF_NAME_JUDGE_API_BASE")
    if base_url is None and os.getenv("CSCS_SERVING_API"):
        base_url = "https://api.swissai.svc.cscs.ch/v1"

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _parse_judge_json(text: str) -> dict[str, bool] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None
    return {
        "correct": bool(parsed.get("correct", False)),
        "incorrect": bool(parsed.get("incorrect", False)),
    }


def _append_judge_log(record: dict[str, Any]) -> None:
    path = os.getenv("SELF_NAME_JUDGE_LOG_PATH")
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_results_llm_judge(doc, predictions, **kwargs):
    completion = _completion(predictions)
    model = os.getenv("SELF_NAME_JUDGE_MODEL")
    if not model:
        raise OSError("Set SELF_NAME_JUDGE_MODEL to use self-name-judge.")

    prompt = JUDGE_USER_TEMPLATE.format(
        language=doc["language"],
        prompt_type=doc["prompt_type"],
        prompt=doc["prompt"],
        expected_model_name=doc["expected_model_name"],
        expected_org_name=doc["expected_org_name"],
        expected_kind=doc["expected_kind"],
        completion=completion,
    )

    try:
        response = _judge_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_judge_json(content)
    except Exception as exc:  # noqa: BLE001
        eval_logger.warning("self-name judge call failed for %s: %s", doc["id"], exc)
        parsed = None
        content = ""

    if parsed is None:
        correct = np.nan
        incorrect = np.nan
        clean_correct = np.nan
    else:
        correct = float(parsed["correct"])
        incorrect = float(parsed["incorrect"])
        clean_correct = float(parsed["correct"] and not parsed["incorrect"])

    _append_judge_log(
        {
            "id": doc["id"],
            "language": doc["language"],
            "prompt_type": doc["prompt_type"],
            "completion": completion,
            "judge_model": model,
            "judge_response": content,
            "parsed": parsed,
        }
    )

    return {
        "llm_correct_naming_rate": correct,
        "llm_incorrect_naming_rate": incorrect,
        "llm_clean_correct_naming_rate": clean_correct,
    }
