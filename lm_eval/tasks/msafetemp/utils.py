"""Row selection for the MSafeTemp HarmBench-track tasks (``swiss-ai/MSafeTemp``, ``harmbench``)."""


def hb_asr_docs(dataset):
    """HB-ASR: translated harmful variants (main template and paraphrases), 60 languages."""
    return dataset.filter(lambda c: c == "harmful", input_columns=["category"])


def hb_or_docs(dataset):
    """HB-OR: benign minimal pairs of class ``direct`` (safe to answer fully), 60 languages."""
    return dataset.filter(
        lambda c, b, k: c == "benign" and b == "main" and k == "direct",
        input_columns=["category", "branch", "benign_class"],
    )


def hb_asr_jb_docs(dataset):
    """HB-ASR-JB: culturally localized (PH) prompts in jailbreak wrappers, 300 per language, 58 languages."""
    return dataset.filter(
        lambda c, b: c == "posthoc" and b == "pyrit_wrapped",
        input_columns=["category", "branch"],
    )


def hb_or_jb_docs(dataset):
    """HB-OR-JB: benign prompts in jailbreak wrappers, 300 per language, 60 languages."""
    return dataset.filter(
        lambda c, b: c == "benign" and b == "pyrit_wrapped",
        input_columns=["category", "branch"],
    )
