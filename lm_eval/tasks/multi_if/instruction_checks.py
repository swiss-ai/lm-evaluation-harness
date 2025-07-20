import re

# Placeholder examples; add full set from original repo.
def must_include_phrase(response, phrase, case_sensitive=False, **_):
    if not case_sensitive:
        return phrase.lower() in response.lower()
    return phrase in response

def min_characters(response, min_chars, **_):
    return len(response) >= int(min_chars)

def max_words(response, max_words, **_):
    return len(response.strip().split()) <= int(max_words)

INSTRUCTION_REGISTRY = {
    "must_include_phrase": must_include_phrase,
    "min_characters": min_characters,
    "max_words": max_words,
    # add remaining instruction id -> function mappings here
}
