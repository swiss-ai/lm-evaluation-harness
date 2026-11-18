def doc_to_text_cot(doc) -> str:
    # The Text subset's question already ends with "Answer Choices: (A) ... (J)".
    prompt = f"Question: {doc['question'].strip()}\n"
    prompt += (
        "First, think step-by-step. End your response with exactly this format:\n"
        "Final Answer: X\n(where X is one of A, B, C, D, E, F, G, H, I or J)."
    )
    return prompt
