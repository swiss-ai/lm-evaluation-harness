import re

from datasets import Dataset


def extract_year_from_url(url: str) -> str:
    """Extract the year from an AIME problem URL."""
    match = re.search(r"index\.php/(\d{4})_", url)
    if not match:
        raise ValueError(f"Could not extract year from URL: {url}")
    return match.group(1)


def process_2022(dataset: Dataset) -> Dataset:
    """Filter dataset to only include problems from 2022."""

    def filter_by_year(doc):
        try:
            doc_year = extract_year_from_url(doc["url"])
            return doc_year == "2022"
        except ValueError:
            return False

    return dataset.filter(filter_by_year)


def process_2023(dataset: Dataset) -> Dataset:
    """Filter dataset to only include problems from 2023."""

    def filter_by_year(doc):
        try:
            doc_year = extract_year_from_url(doc["url"])
            return doc_year == "2023"
        except ValueError:
            return False

    return dataset.filter(filter_by_year)


def process_2024(dataset: Dataset) -> Dataset:
    """Filter dataset to only include problems from 2024."""

    def filter_by_year(doc):
        try:
            doc_year = extract_year_from_url(doc["url"])
            return doc_year == "2024"
        except ValueError:
            return False

    return dataset.filter(filter_by_year)


from collections import Counter

from lm_eval.api.metrics import is_degenerating_text


# AIME answers are integers in [0, 999], and the prompt ("Question: ...\nAnswer:")
# never asks for a boxed answer -- so extraction cannot depend on \boxed{}: a model
# that simply ends with "Answer: 16" has to be scored too. Order matters below: an
# explicit answer marker beats trailing inline math, which is often a congruence
# ("$$2016 \equiv 16 \pmod{1000}$$") whose last number is the modulus.
_INT_RE = re.compile(r"[+-]?\d[\d,]*")
_ANSWER_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?", re.IGNORECASE
)
_MATH_SPAN_RE = re.compile(r"\$\$?(.+?)\$\$?", re.DOTALL)


def _as_int(text: str) -> str | None:
    """Canonical form of a bare integer answer (``"016"``, ``"1,260"``), else ``None``."""
    cleaned = text.strip().strip("$").replace(",", "").replace(" ", "")
    return str(int(cleaned)) if re.fullmatch(r"[+-]?\d+", cleaned) else None


def extract_answer(response: str) -> str:
    """Extract the final answer from a response (``""`` if it has none).

    Tries, in order: ``\\boxed{...}``, the first number after the last explicit
    answer marker, the last ``$...$`` span when it is a bare number, and finally the
    last number anywhere in the response.
    """
    boxed = last_boxed_only_string(response)
    if boxed is not None:
        try:
            content = remove_boxed(boxed)
        except (AssertionError, IndexError):
            content = None
        if content:
            return _as_int(content) or content.strip()

    markers = list(_ANSWER_MARKER_RE.finditer(response))
    if markers:
        match = _INT_RE.search(response[markers[-1].end() :])
        if match:
            return _as_int(match.group(0)) or ""

    spans = _MATH_SPAN_RE.findall(response)
    if spans:
        value = _as_int(spans[-1])
        if value is not None:
            return value

    numbers = _INT_RE.findall(response)
    return _as_int(numbers[-1]) or "" if numbers else ""


def _normalize_answer(response: str) -> str:
    """Extract and normalize a response's answer so answers can be compared/voted."""
    try:
        return strip_string(extract_answer(response))
    except Exception:
        return extract_answer(response)


class MajorityVoteFilter:
    """Self-consistency (maj@k) filter for AIME.

    Reuses the task's answer extraction to vote: for each problem it extracts and
    normalizes the answer of every sampled response, takes the majority vote, and
    returns a representative *full* response whose answer matches the winner.
    Returning the full text (rather than the bare answer) keeps the downstream
    ``process_results`` extraction and the ``degeneration`` metric working unchanged.
    Chain a ``take_first`` after this filter to unwrap the single voted response.
    """

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list):
            normalized = [_normalize_answer(r) for r in resp_list]
            # Responses with no extractable answer must not win the vote.
            votable = [a for a in normalized if a] or normalized
            winner = Counter(votable).most_common(1)[0][0]
            for resp, norm in zip(resp_list, normalized):
                if norm == winner:
                    return resp
            return resp_list[0]

        return map(lambda r: [select(r)], resps)


def _score_response(response: str, target: str) -> int:
    """1 if the response's extracted answer matches the target, else 0."""
    answer = extract_answer(response)
    if not answer:
        return 0
    # Compare integers as integers so a zero-padded gold ("016") still matches.
    return int(is_equiv(answer, _as_int(target) or target))


def _doc_target(doc: dict) -> str:
    """The gold answer as a string, regardless of the answer field's casing."""
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    return str(doc[answer_key])


class PassAtKFilter:
    """pass@k filter for AIME.

    A problem counts as solved if *any* of the k sampled responses is correct.
    Using the gold answer (available via ``docs``), this returns a correct sample
    when one exists, else the first sample, so the downstream single-response
    scoring in ``process_results`` yields 1 iff at least one attempt was correct.
    With n == k samples this equals the standard unbiased pass@k estimator.
    Chain a ``take_first`` after this filter to unwrap the selected response.
    """

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list, doc):
            target = _doc_target(doc)
            for resp in resp_list:
                if _score_response(resp, target):
                    return resp
            return resp_list[0]

        return [[select(r, d)] for r, d in zip(resps, docs)]


def process_results(doc: dict, results: list) -> dict[str, float]:
    # A filter may collapse the repeats to a single response (e.g. take_first,
    # maj@k, pass@k) or pass them all through (e.g. take_first_k for mean@k).
    # Normalize to a list so a single metric path handles both.
    response = results[0]
    responses = list(response) if isinstance(response, list) else [response]

    target = _doc_target(doc)

    # For a single response this is just 0/1 (per-sample accuracy); for the full
    # set of k samples it is the mean accuracy over the k attempts (mean@k / avg@k).
    scores = [_score_response(r, target) for r in responses]
    degens = [int(is_degenerating_text(r.lower())) for r in responses]

    return {
        "exact_match": sum(scores) / len(scores),
        "degeneration": sum(degens) / len(degens),
    }


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
