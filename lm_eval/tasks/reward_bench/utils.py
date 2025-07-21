import random
import datasets

generative_rm_system_prompt = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
    "The question provided is in {language}. "
    "You should choose the assistant that follows the user's instructions and answers the user's question best. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Also, make sure that the assistant responses are in {language}. "
    "Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best.'
)

generative_rm_user_prompt = (
    "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
)

LANGUAGE_MAPPING = {
    "en_Latn": "English",
    "arb_Arab": "Arabic",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "spa_Latn": "Spanish",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "hin_Deva": "Hindi",
    "ces_Latn": "Czech",
    "nld_Latn": "Dutch",
    "pol_Latn": "Polish",
    "tur_Latn": "Turkish",
    "vie_Viet": "Vietnamese",
    "ell_Grek": "Greek",
    "heb_Hebr": "Hebrew",
    "ind_Latn": "Indonesian",
    "pes_Arab": "Persian",
    "ron_Latn": "Romanian",
    "ukr_Cyrl": "Ukrainian"
}

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        dice = random.random()
        if dice < 0.5:
            answer = "[[A]]"
            answer_a = doc["chosen"]
            answer_b = doc["rejected"]
        else:
            answer = "[[B]]"
            answer_a = doc["rejected"]
            answer_b = doc["chosen"]
        out_doc = {
            **doc,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "target": answer,
            "language": LANGUAGE_MAPPING[doc.get("language", "en_Latn")],
        }
        return out_doc

    return dataset.map(_process_doc)

def generative_rm_doc_to_text(doc):
    return generative_rm_system_prompt + "\n\n" + generative_rm_user_prompt.format(
        question=doc["prompt"],
        answer_a=doc["answer_a"],
        answer_b=doc["answer_b"]
    )