"""FLORES-200 (facebook/flores), devtest, reference-based chrF/BLEU.

The dataset has one parquet per language (``data/language/<lang>/devtest-...``)
whose rows share an ``id``. A pair is built by joining the source and target files on ``id``.
The dataset is gated: the job needs an HF token with access.
"""

import datasets
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "facebook/flores"
LANG_NAMES = {'deu_Latn': 'German', 'fra_Latn': 'French', 'ita_Latn': 'Italian', 'spa_Latn': 'Spanish', 'eng_Latn': 'English'}


def _rows(lang):
    path = hf_hub_download(
        REPO, f"data/language/{lang}/devtest-00000-of-00001.parquet", repo_type="dataset"
    )
    t = pq.read_table(path, columns=["id", "sentence"]).to_pydict()
    return dict(zip(t["id"], t["sentence"]))


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
