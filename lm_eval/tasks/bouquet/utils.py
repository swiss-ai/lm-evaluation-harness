"""BOUQuET (facebook/bouquet), sentence level, reference-based chrF/BLEU.

Each task is one direction, e.g. ``bouquet_eng_Latn-deu_Latn``, read from
``benchmark_data/sentence_level/test/<src>-<tgt>.parquet``. The dataset is gated:
the job needs an HF token whose account has been granted access.
"""

LANG_NAMES = {'deu_Latn': 'German', 'fra_Latn': 'French', 'ita_Latn': 'Italian', 'spa_Latn': 'Spanish', 'roh_Latn': 'Romansh', 'eng_Latn': 'English'}


def _lang_name(code):
    return LANG_NAMES.get(code, code)


def _make_prompt(src, src_lang, tgt_lang):
    s, t = _lang_name(src_lang), _lang_name(tgt_lang)
    return (
        f"You are a professional {s} to {t} translator. Translate the following "
        f"{s} sentence into {t}. Produce only the {t} translation, without any "
        f"explanations or commentary:\n\n{src}"
    )


def process_docs(dataset):
    dataset = dataset.filter(lambda d: bool(d["src_text"]) and bool(d["tgt_text"]))

    def _add_prompt(doc):
        doc["prompt"] = _make_prompt(doc["src_text"], doc["src_lang"], doc["tgt_lang"])
        return doc

    return dataset.map(_add_prompt)
