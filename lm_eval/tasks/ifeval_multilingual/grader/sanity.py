"""
Deterministic checks of the catalog pipeline (generate_catalog.py) and of language profiles (profile.py):

gate(type_id, script):        types that cannot apply to a script are rejected before any LLM call
check_kwargs(type_id, kwargs): checker settings, (kwargs, problem) with problem None if usable
    control characters        JSON "\\b" is a backspace, not a regex word boundary: the pattern never matches
    \\b in non-Latin patterns  Python's \\b fails next to Devanagari vowel signs etc.: the pattern never matches
    local format example      "required" must match the example, "forbidden" must not
settings_problem(type_id, type_def, kwargs): the three checks of a type's settings together (the right names,
    check_kwargs, the checker runs with them); None if usable
Scripts are ISO 15924 codes (Latn, Arab, Deva, Hans…), from langcodes or --script.
"""

import re
import unicodedata

from . import checkers

# scripts with no alphabet a letter rule could name: thousands of characters or syllables
NO_ALPHABET = {"Hans", "Hant", "Jpan", "Kore", "Ethi"}

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REGEX_KWARGS = ("required", "forbidden")
# types scored by LocalFormatChecker: the example must match "required" and none of "forbidden"
FORMAT_TYPES = ("numbers:large_numbers", "numbers:currency", "dates:numeric_date")


def script_of(code):
    """ISO 15924 script of a language code (gsw -> Latn, zh -> Hans, hi -> Deva)."""
    import langcodes
    return langcodes.Language.get(code).maximize().script


def gate(type_id, script):
    """Why the type cannot apply to the script, or None."""
    if type_id.startswith("letters:") and type_id != "letters:diacritics" and script in NO_ALPHABET:
        return f"letter types need an alphabet (script {script})"
    if type_id == "letters:diacritics" and script not in ("Arab", "Hebr"):
        return f"no vowel mark set for script {script} (Arabic, Hebrew only)"
    if type_id == "punctuation:sentence_end" and script in ("Latn", "Cyrl", "Grek"):
        return f"languages in script {script} end sentences with the ASCII full stop"
    if type_id.startswith("han:") and script not in ("Hans", "Hant", "Jpan"):
        return f"the language is not written in Chinese characters (script {script})"
    if type_id == "script:script_variant" and script not in ("Hans", "Hant"):
        return f"script variants (OpenCC) exist only for Chinese (script {script})"
    return None


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)


def check_kwargs(type_id, kwargs):
    """(kwargs, problem): problem is None or a message."""
    for s in strings(kwargs):
        if CONTROL.search(s):
            return kwargs, f"control character in {s!r} (in JSON, '\\b' is a backspace; write '\\\\b' for a word boundary)"
    for key in REGEX_KWARGS:
        for pattern in kwargs.get(key, []):
            # \b fails next to combining marks (Devanagari vowel signs ी ा, Arabic harakat), which are not \w
            if "\\b" in pattern and any(unicodedata.category(c).startswith("M") for c in pattern):
                return kwargs, (f"pattern {pattern!r}: '\\b' does not work next to combining marks (e.g. after a "
                                "Devanagari vowel sign); use plain substrings or (?<!\\S) / (?!\\S)")
    if type_id == "punctuation:sentence_end":
        mark, foreign = kwargs.get("mark", ""), kwargs.get("foreign", "")
        if len(mark) != 1 or mark.isascii():
            return kwargs, f"\"mark\" must be one non-ASCII character (got {mark!r})"
        if foreign not in (".", "?"):
            return kwargs, f"\"foreign\" must be \".\" or \"?\" (got {foreign!r})"
    if type_id in FORMAT_TYPES:
        example = kwargs.get("example", "")
        try:
            hit_required = [p for p in kwargs.get("required", []) if re.search(p, example)]
            hit_forbidden = [p for p in kwargs.get("forbidden", []) if re.search(p, example)]
        except re.error as e:
            return kwargs, f"invalid regex: {e}"
        if not example or not hit_required:
            return kwargs, f"no \"required\" pattern matches the local-format example {example!r}"
        if hit_forbidden:
            return kwargs, f"the \"forbidden\" pattern(s) {hit_forbidden} also match the local-format example {example!r}"
    if type_id == "script:script_variant" and kwargs.get("conversion") not in ("s2tw", "s2hk", "t2s"):
        return kwargs, ("conversion must be \"s2tw\" (Traditional, Taiwan standard), \"s2hk\" (Traditional, Hong Kong "
                        "standard) or \"t2s\" (Simplified); not the generic \"s2t\" (old variants such as 喫 for 吃)")
    return kwargs, None


def settings_problem(type_id, type_def, kwargs, run=True):
    """Why these settings cannot be used with the type (data_types.py), or None. run=False leaves out the last check,
    for a checker that needs what this machine lacks (a tagger)."""
    if set(kwargs) != set(type_def["kwargs"]):
        return f"parameters {sorted(kwargs)} instead of {sorted(type_def['kwargs'])}"
    problem = check_kwargs(type_id, kwargs)[1]
    if problem:
        return problem
    if run:   # the checker must run with them (e.g. the regexes compile)
        try:
            lows = {name: low for name, (low, _) in type_def["sampled"].items()}
            checkers.build(type_def["checker"], type_id, kwargs, lows).check_following("")
        except Exception as e:
            return f"parameters unusable by the checker: {type(e).__name__}: {e}"
    return None
