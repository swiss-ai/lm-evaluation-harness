"""Runtime data loading for WMT translation tasks.

Downloads WMT translation data via subset2evaluate on first run, caches in memory,
and returns per-language-pair datasets for each task.
"""

import logging
from functools import lru_cache

import datasets


eval_logger = logging.getLogger(__name__)

LANG_NAMES = {
    "en": "English",
    "cs": "Czech",
    "ja": "Japanese",
    "de": "German",
    "es": "Spanish",
    "hi": "Hindi",
    "is": "Icelandic",
    "ru": "Russian",
    "uk": "Ukrainian",
    "zh": "Chinese",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "da": "Danish",
    "el": "Greek",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zu": "Zulu",
    "bho": "Bhojpuri",
    "mas": "Maasai",
}

LOCALE_NAMES = {
    "ar_EG": "Arabic (Egypt)",
    "ar_SA": "Arabic (Saudi Arabia)",
    "bg_BG": "Bulgarian",
    "bn_IN": "Bengali",
    "ca_ES": "Catalan",
    "cs_CZ": "Czech",
    "da_DK": "Danish",
    "de_DE": "German",
    "el_GR": "Greek",
    "es_MX": "Spanish (Mexico)",
    "et_EE": "Estonian",
    "fa_IR": "Persian",
    "fi_FI": "Finnish",
    "fil_PH": "Filipino",
    "fr_CA": "French (Canada)",
    "fr_FR": "French",
    "gu_IN": "Gujarati",
    "he_IL": "Hebrew",
    "hi_IN": "Hindi",
    "hr_HR": "Croatian",
    "hu_HU": "Hungarian",
    "id_ID": "Indonesian",
    "is_IS": "Icelandic",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
    "kn_IN": "Kannada",
    "ko_KR": "Korean",
    "lt_LT": "Lithuanian",
    "lv_LV": "Latvian",
    "ml_IN": "Malayalam",
    "mr_IN": "Marathi",
    "nl_NL": "Dutch",
    "no_NO": "Norwegian",
    "pa_IN": "Punjabi",
    "pl_PL": "Polish",
    "pt_BR": "Portuguese (Brazil)",
    "pt_PT": "Portuguese",
    "ro_RO": "Romanian",
    "ru_RU": "Russian",
    "sk_SK": "Slovak",
    "sl_SI": "Slovenian",
    "sr_RS": "Serbian",
    "sr_Cyrl_RS": "Serbian (Cyrillic)",
    "sv_SE": "Swedish",
    "sw_KE": "Swahili (Kenya)",
    "sw_TZ": "Swahili (Tanzania)",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "th_TH": "Thai",
    "tr_TR": "Turkish",
    "uk_UA": "Ukrainian",
    "ur_PK": "Urdu",
    "vi_VN": "Vietnamese",
    "zh_CN": "Chinese (Simplified)",
    "zh_TW": "Chinese (Traditional)",
    "zu_ZA": "Zulu",
    "bho_IN": "Bhojpuri",
    "mas_KE": "Maasai",
}


def _lang_display_name(code):
    if code in LOCALE_NAMES:
        return LOCALE_NAMES[code]
    base = code.split("_")[0]
    return LANG_NAMES.get(base, code)


def _lang_base_name(code):
    base = code.split("_")[0]
    return LANG_NAMES.get(base, code)


def _make_prompt(src, src_lang, tgt_lang):
    src_name = _lang_base_name(src_lang)
    tgt_display = _lang_display_name(tgt_lang)
    return (
        f"You are a professional {src_name} to {tgt_display} translator, "
        f"tasked with providing translations suitable for use in "
        f"{tgt_display} ({tgt_lang}). Your goal is to accurately convey the "
        f"meaning and nuances of the original {src_name} text while adhering "
        f"to {tgt_display} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {tgt_display} translation, without any additional "
        f"explanations or commentary. "
        f"Please translate the following {src_name} text into "
        f"{tgt_display} ({tgt_lang}):\n\n{src}"
    )


@lru_cache(maxsize=1)
def _download_wmt_data():
    """Download all WMT data via subset2evaluate (cached after first call).

    Returns None if subset2evaluate is not installed (allows graceful
    task registration in environments without the optional dependency).
    """
    try:
        import subset2evaluate.utils
    except ImportError:
        eval_logger.warning(
            "subset2evaluate is not installed — WMT translation data "
            "unavailable. Install with: pip install subset2evaluate"
        )
        return None

    eval_logger.info(
        "Downloading WMT data via subset2evaluate (first run may be slow)..."
    )
    raw = subset2evaluate.utils.load_data_wmt_all(
        require_human=False,
        name_filter=lambda k: k[0] in {"wmt25", "wmt24", "wmt24pp"},
    )
    data = [line | {"dataset": f"{k[0]}/{k[1]}"} for k, v in raw.items() for line in v]
    eval_logger.info(f"Loaded {len(data):,} items across {len(raw)} language pairs")
    return data


def load_wmt_data(**kwargs):
    """Load WMT translation data for a specific language pair.

    Called by lm-eval-harness via custom_dataset. Receives dataset_kwargs
    from the task YAML.

    Args:
        dataset_key: e.g. "wmt25/en-zh_CN" — identifies the dataset and language pair.
    """
    dataset_key = kwargs["dataset_key"]
    dataset_name, lang_pair = dataset_key.split("/")
    src_lang, tgt_lang = lang_pair.split("-", 1)

    all_data = _download_wmt_data()

    if all_data is None:
        eval_logger.warning(
            f"Returning placeholder dataset for {dataset_key} "
            f"(subset2evaluate not installed)"
        )
        return {"test": datasets.Dataset.from_list([{"prompt": "", "src": ""}])}

    items = [
        {
            "prompt": _make_prompt(item["src"], src_lang, tgt_lang),
            "src": item["src"],
        }
        for item in all_data
        if item["dataset"] == dataset_key
    ]

    if not items:
        raise ValueError(
            f"No data found for dataset_key={dataset_key!r}. "
            f"Available keys: {sorted(set(d['dataset'] for d in all_data))}"
        )

    eval_logger.info(f"WMT task {dataset_key}: {len(items)} test items")
    ds = datasets.Dataset.from_list(items)
    return {"test": ds}
