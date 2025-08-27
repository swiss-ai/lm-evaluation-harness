import ast

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Convert options from string representation to list."""
    return dataset.map(lambda x: {"options": ast.literal_eval(x["options"])})


def doc_to_target_mcq(doc):
    """Get answer index (e.g. 0) for multiple choice evaluation mode."""
    return int(doc["answer"])


def doc_to_target_gen(doc):
    """Convert answer (e.g. "0") into corresponding letter (e.g. "A") for generation evaluation mode."""
    answer_idx = int(doc["answer"])
    return chr(ord("A") + answer_idx)


def doc_to_choice(doc):
    """Return raw options for multiple choice evaluation mode."""
    return doc["options"]


def doc_to_text_mcq(doc):
    """Format prompt for multiple choice evaluation mode."""
    options = doc["options"]

    text = f"Passage: {doc['context'].strip()}\n"
    text += f"Question: {doc['question'].strip()}\n"
    text += "Choices:\n"
    text += f"A. {options[0]}\n"
    text += f"B. {options[1]}\n"
    text += f"C. {options[2]}\n"
    text += f"D. {options[3]}\n"
    text += "Please choose the most suitable one among A, B, C and D as the answer to this question."

    return text


def doc_to_text_gen(doc):
    """Format prompt for generation evaluation mode."""
    options = doc["options"]

    text = f"Passage: {doc['context'].strip()}\n"
    text += f"Question: {doc['question'].strip()}\n"
    text += "Choices:\n"
    text += f"A. {options[0]}\n"
    text += f"B. {options[1]}\n"
    text += f"C. {options[2]}\n"
    text += f"D. {options[3]}\n"
    text += "Please choose the most suitable one among A, B, C and D as the answer to this question."

    return text
