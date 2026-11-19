"""MMLU-ProX health, zero-shot step-by-step, one task per language (en, fr, de, hi, sw), for chat/thinking models.

Same questions in every language (li-lab/MMLU-ProX test split, category "health"), so scores compare across languages.
The prompt is in the target language: MMLU-ProX's own instruction for health ("think step by step and finish with
'<the answer is (X)>'" in that language), its question/options labels, and its answer phrase. The answer is read
from the LAST answer phrase in the response, the target-language one or the English "the answer is" / "Final Answer:"
(a model that switches to English is still scored on its choice).
"""
from functools import partial

from lm_eval.tasks.mmlu_prox.lang_libs import LANG_LIBS

SUBJECT = {"en": "health", "fr": "santé", "de": "Gesundheit", "hi": "स्वास्थ्य", "sw": "afya"}
LETTERS = "ABCDEFGHIJ"


def process_docs(dataset):
    return dataset.filter(lambda x: x["category"] == "health")


def doc_to_text(doc, lang):
    q_label, opt_label, _, instruction, _, ans = LANG_LIBS[lang][:6]
    suffix = ans.format("X")
    instruction = instruction.format(subject=SUBJECT[lang], ans_suffix=suffix)
    # the upstream sentence introduces a list of solved examples; zero-shot, keep only its second sentence
    instruction = instruction.split(". ", 1)[1] if lang != "hi" else instruction.split("। ", 1)[1]
    opts = "".join(f"{LETTERS[i]}. {doc[f'option_{i}']}\n" for i in range(10) if doc.get(f"option_{i}") is not None)
    return f"{q_label} {doc['question']}\n{opt_label}\n{opts}\n{instruction}"


doc_to_text_en = partial(doc_to_text, lang="en")
doc_to_text_fr = partial(doc_to_text, lang="fr")
doc_to_text_de = partial(doc_to_text, lang="de")
doc_to_text_hi = partial(doc_to_text, lang="hi")
doc_to_text_sw = partial(doc_to_text, lang="sw")
