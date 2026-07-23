import logging
import re
import signal
from collections import Counter
from functools import cache
from importlib.metadata import version

import datasets


eval_logger = logging.getLogger(__name__)

INVALID_ANSWER = "[invalidanswer]"


try:
    import antlr4  # noqa: F401
    import sympy
    from math_verify import parse, verify
    from sympy.parsing.latex import parse_latex

    assert version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`sympy`, `math_verify` and `antlr4-python3-runtime==4.11` are required for generating translation task prompt templates. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    candidates = results[0]

    unnormalized_answer = get_unnormalized_answer(candidates)
    answer = normalize_final_answer(unnormalized_answer)

    if is_equiv(answer, doc["answer"]):
        retval = 1
    else:
        retval = 0

    # math_verify
    _mvres = verify(
        gold=parse(doc["solution"]),
        target=parse(candidates),
    )
    mathval = 1 if _mvres else 0

    res = {
        "exact_match": retval,
        "math_verify": mathval,
    }
    return res


# --- Self-consistency (score-first / mean@k / maj@k / pass@k) -----------------
# The filters below collapse the k repeats to one representative response; the
# mean@k filter (take_first_k) instead passes them all through, and
# process_results_sc averages exact_match / math_verify over the list.
#
# Scoring one response is expensive -- `is_equiv` runs sympy `simplify` under a 5s
# timeout and math_verify has to parse the whole generation -- and the same
# response is scored many times over: once inside PassAtKFilter, and again for
# every filter that selects it (score-first / maj@k / pass@k all tend to pick a
# response that mean@k already scored). So per-response scores and the parsed gold
# solution are memoized. The cache keys are response/solution strings that the
# Instances hold onto anyway, so this costs references, not copies.


@cache
def _parse_gold(solution: str):
    """math_verify's parse of the gold solution -- identical for all k samples of a doc."""
    return parse(solution)


@cache
def _score(response: str, gold_answer: str, solution: str) -> tuple[int, int]:
    """``(exact_match, math_verify)`` for a single response against a doc's gold."""
    answer = _sc_answer(response)
    em = 1 if answer and is_equiv(answer, gold_answer) else 0
    try:
        mv = 1 if verify(gold=_parse_gold(solution), target=parse(response)) else 0
    except Exception:
        mv = 0
    return em, mv


def _score_doc(response: str, doc: dict) -> tuple[int, int]:
    return _score(response, doc["answer"], doc["solution"])


def _sc_answer(response: str) -> str:
    """Normalized final answer extracted from a response (``""`` if it has none).

    Prefers the Minerva few-shot format (``Final Answer: The final answer is ...``)
    and falls back to the last ``\\boxed{...}``, which is what models actually emit
    zero-shot. Without the fallback every zero-shot response extracts as
    ``[invalidanswer]``: exact_match collapses to 0 and the majority vote becomes a
    tie between non-answers. Returning ``""`` rather than the sentinel is what lets
    the filters below drop responses with no answer.
    """
    answer = get_unnormalized_answer(response)
    if answer == INVALID_ANSWER:
        boxed = last_boxed_only_string(response)
        if boxed is None:
            return ""
        try:
            answer = remove_boxed(boxed)
        except AssertionError:
            return ""
    return normalize_final_answer(answer)


class MajorityVoteFilter:
    """maj@k: majority vote over the normalized answers, returning a representative
    full response whose answer matches the winner (empty answers ignored unless all
    are empty). Chain a ``take_first`` after this filter."""

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list):
            answers = [_sc_answer(r) for r in resp_list]
            votable = [a for a in answers if a] or answers
            winner = Counter(votable).most_common(1)[0][0]
            for resp, ans in zip(resp_list, answers):
                if ans == winner:
                    return resp
            return resp_list[0]

        return [[select(r)] for r in resps]


class PassAtKFilter:
    """pass@k: returns a correct response if any of the k exists, else the first.
    With n == k this is the unbiased pass@k. Chain a ``take_first`` after this filter.

    A response correct under *both* exact_match and math_verify is preferred, so
    that both reported metrics are a true pass@k whenever one response satisfies
    both; otherwise exact_match wins the tie-break."""

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list, doc):
            # `max` keeps the first maximal element, so an all-wrong set of samples
            # falls back to the first response.
            scored = [(_score_doc(resp, doc), resp) for resp in resp_list]
            return max(scored, key=lambda item: (sum(item[0]), item[0][0]))[1]

        return [[select(r, d)] for r, d in zip(resps, docs)]


def process_results_sc(doc: dict, results: list) -> dict[str, float]:
    """Self-consistency-aware variant of ``process_results``.

    Handles a single response (take_first / maj@k / pass@k) or the full set of k
    samples (take_first_k for mean@k), averaging exact_match and math_verify over
    the list. For a single response this reduces to the same 0/1 as
    ``process_results``.
    """
    response = results[0]
    responses = list(response) if isinstance(response, list) else [response]

    scores = [_score_doc(cand, doc) for cand in responses]

    return {
        "exact_match": sum(em for em, _ in scores) / len(scores),
        "math_verify": sum(mv for _, mv in scores) / len(scores),
    }


def last_boxed_only_string(string: str) -> str | None:
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


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
