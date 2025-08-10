import ast

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Convert options from string representation to list."""
    return dataset.map(lambda x: {"options": ast.literal_eval(x["options"])})


def doc_to_choice(doc):
    """Parse options and format them as A, B, C, D choices. Only used for multiple choice evaluation mode."""
    return [f"{chr(ord('A') + i)}. {x}" for i, x in enumerate(doc["options"])]


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
    text += "Please choose the most suitable one among A, B, C and D as the answer to this question, and return it in the following JSON format:\n"
    text += "{'answer': '[choice]'}\n"
    text += "where [choice] must be one of A, B, C and D."

    return text
