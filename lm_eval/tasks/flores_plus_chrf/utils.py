"""FLORES+ (openlanguagedata/flores_plus), devtest, reference-based chrF/BLEU.

The dataset has one file per language (``devtest/<lang>.jsonl``) whose rows share
an ``id``. A pair is built by joining the source and target files on ``id``.
The dataset is gated: the job needs an HF token with access.
"""

import json

import datasets
from huggingface_hub import hf_hub_download

REPO = "openlanguagedata/flores_plus"
LANG_NAMES = {'deu_Latn': 'German', 'fra_Latn': 'French', 'ita_Latn': 'Italian', 'spa_Latn': 'Spanish', 'eng_Latn': 'English'}


def _rows(lang):
    path = hf_hub_download(REPO, f"devtest/{lang}.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["text"] for r in map(json.loads, f)}


def load_pair(**kwargs):
    src_lang, tgt_lang = kwargs["src_lang"], kwargs["tgt_lang"]
    src, tgt = _rows(src_lang), _rows(tgt_lang)
    ids = sorted(set(src) & set(tgt))
    rows = [
        {"id": i, "source": src[i], "target": tgt[i], "src_lang": src_lang, "tgt_lang": tgt_lang}
        for i in ids
    ]
    return {"test": datasets.Dataset.from_list(rows)}


def _make_prompt(src, src_lang, tgt_lang):
    s, t = LANG_NAMES.get(src_lang, src_lang), LANG_NAMES.get(tgt_lang, tgt_lang)
    return (
        f"You are a professional {s} to {t} translator. Translate the following "
        f"{s} sentence into {t}. Produce only the {t} translation, without any "
        f"explanations or commentary:\n\n{src}"
    )


def process_docs(dataset):
    def _add_prompt(doc):
        doc["prompt"] = _make_prompt(doc["source"], doc["src_lang"], doc["tgt_lang"])
        return doc

    return dataset.map(_add_prompt)
