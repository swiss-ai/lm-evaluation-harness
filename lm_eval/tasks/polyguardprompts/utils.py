"""Document processing for PolyGuardPrompts (ToxicityPrompts/PolyGuardPrompts).

The HF dataset has a single ``test`` split with 29,325 rows: the 1,725
WildGuardTest prompt/response pairs in 17 languages (16 GPT-4o translations +
English), one row per (``id``, ``language``).

Generation subtasks (``polyguardprompts_<lang>``): Apertus answers each prompt.
Before that we drop *translator refusals*: rows where the GPT-4o translator
refused to translate a harmful prompt and the "prompt" is a refusal message
(e.g. "Lo siento, pero no puedo ayudarte con esa solicitud."). Such a row is
not a harmful request, so it would only dilute the harmful-response rate. We
then de-duplicate exact prompt strings within a language (keeping the lowest
``id``). English needs neither step.

Classifier subtasks (``polyguardprompts_cls_<lang>``) keep every row as
released, to stay comparable with the paper's classifier numbers.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets


LANGUAGES = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "zh": "Chinese",
}

# Refusal openings/phrases written by the translator, per language. A row is a
# translator refusal if its (short) prompt matches one of these and the English
# source prompt is not itself such a message.
_REFUSAL_PATTERNS = {
    "Arabic": r"(عذر[ًا]*|آسف|أعتذر|نعتذر).{0,40}لا (يمكنني|أستطيع|يمكننا|نستطيع)|لا (يمكنني|أستطيع) (المساعدة|مساعدتك|ترجمة)",
    "Chinese": r"(抱歉|对不起|很遗憾)[，,、\s]*(但)?(我|我们)?(无法|不能|没法)",
    "Czech": r"omlouvám se.{0,40}(nemohu|nemůžu)|(nemohu|nemůžu) (vám )?(pomoci|přeložit|splnit)",
    "Dutch": r"(sorry|het spijt me).{0,40}(ik kan|kan ik)|ik kan (u |je |jou )?(niet helpen|daar niet mee helpen|dit niet vertalen)",
    "French": r"(désolée?|je regrette).{0,40}(je ne peux|ne puis)|je ne peux pas (vous |t')?(aider|traduire)",
    "German": r"(entschuldigung|es tut mir leid|leider).{0,60}(kann ich|ich kann)|(kann|darf) ich (diesen|den|diese).{0,30}nicht übersetzen|ich kann (bei|dir|ihnen) .{0,30}nicht (helfen|behilflich)",
    "Hindi": r"(माफ़ कीजिए|माफ कीजिए|क्षमा करें|मुझे खेद है).{0,60}नहीं|नहीं कर (सकता|सकती)",
    "Italian": r"mi dispiace.{0,40}non posso|non posso (aiutarti|assisterti|tradurre)",
    "Japanese": r"(翻訳|お応え|お受け|お手伝い|お答え|作成|生成|提供)(すること)?(が|は)?できません|お断り",
    "Korean": r"(도와드릴|도움을 드릴|번역할|번역해 ?드릴|처리할|응할|작성할|제공할) 수 없",
    "Polish": r"przepraszam.{0,40}nie (mogę|jestem w stanie)|nie mogę (pomóc|przetłumaczyć|spełnić)",
    "Portuguese": r"(desculpe|lamento|sinto muito).{0,40}não (posso|consigo)|não posso (ajudar|traduzir)",
    "Russian": r"(извините|простите|к сожалению).{0,40}(не могу|не смогу)|не могу (помочь|перевести|выполнить)",
    "Spanish": r"lo siento.{0,40}no puedo|no puedo (ayudarte|traducir|cumplir|proporcionar)",
    "Swedish": r"(tyvärr|jag är ledsen|förlåt).{0,40}(kan jag inte|jag kan inte)|jag kan (tyvärr )?inte (hjälpa|översätta)",
    "Thai": r"ขออภัย|ขอโทษ.{0,40}ไม่สามารถ|ไม่สามารถ(ช่วย|แปล)",
}
_EN_REFUSAL = re.compile(
    r"(sorry|apologi[sz]e).{0,40}(can't|cannot|unable)|i (can't|cannot) (help|assist|translate)",
    re.IGNORECASE,
)
_REFUSAL_RE = {k: re.compile(v, re.IGNORECASE) for k, v in _REFUSAL_PATTERNS.items()}
# Translator refusals are one or two sentences; real prompts that contain such
# a phrase are much longer than the English-source refusal would justify.
_MAX_REFUSAL_CHARS = 300


def _english_prompts(dataset: datasets.Dataset) -> dict[int, str]:
    en = dataset.filter(lambda d: d["language"] == "English")
    return dict(zip(en["id"], en["prompt"]))


def is_translator_refusal(prompt: str, prompt_en: str, language: str) -> bool:
    if language == "English" or language not in _REFUSAL_RE:
        return False
    text = prompt.strip()
    if len(text) > _MAX_REFUSAL_CHARS:
        return False
    if _EN_REFUSAL.search(prompt_en or ""):
        return False
    # The refusal must be much shorter than the source prompt.
    if prompt_en and len(text) > 0.8 * len(prompt_en.strip()):
        return False
    return bool(_REFUSAL_RE[language].search(text))


def _with_meta(
    dataset: datasets.Dataset, language: str, lang_code: str
) -> datasets.Dataset:
    en_prompts = _english_prompts(dataset)
    ds = dataset.filter(lambda d: d["language"] == language)

    def add(doc):
        prompt_en = en_prompts.get(doc["id"], "")
        return {
            "lang": lang_code,
            "prompt_en": prompt_en,
            "translator_refusal": is_translator_refusal(
                doc["prompt"], prompt_en, language
            ),
        }

    return ds.map(add)


def process_docs_generation(
    dataset: datasets.Dataset, language: str, lang_code: str
) -> datasets.Dataset:
    ds = _with_meta(dataset, language, lang_code)
    ds = ds.filter(lambda d: not d["translator_refusal"])
    ds = ds.sort("id")
    seen: set[str] = set()
    keep = []
    for i, p in enumerate(ds["prompt"]):
        key = p.strip()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return ds.select(keep)


def process_docs_classifier(
    dataset: datasets.Dataset, language: str, lang_code: str
) -> datasets.Dataset:
    return _with_meta(dataset, language, lang_code)


def _make(fn, language, code):
    def _f(dataset):
        return fn(dataset, language, code)

    _f.__name__ = f"{fn.__name__}_{code}"
    return _f


# One named function per language, referenced from the yamls as
# ``!function utils.process_docs_<lang>`` / ``utils.process_docs_cls_<lang>``.
for _code, _lang in LANGUAGES.items():
    globals()[f"process_docs_{_code}"] = _make(process_docs_generation, _lang, _code)
    globals()[f"process_docs_cls_{_code}"] = _make(
        process_docs_classifier, _lang, _code
    )
