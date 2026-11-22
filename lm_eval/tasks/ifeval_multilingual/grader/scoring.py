"""
Scores responses to IFEval-format rows in other languages, strict and loose as IFEval does. Each instruction of a row uses:
    "native:<type>"     the checker of the language-specific type (data_types.py)
    "native:<entry>"    the checker of a language-profile entry (entries.json)
    rewrite_checker     in the manifest for that language: a checker of this file (English ones would fail on the language)
    otherwise           IFEval's own checker (lm_eval/tasks/ifeval)

Copied from ritranslation-hooks configs/benchmark/ifeval (evaluate/scoring.py, native/*), commit 34f1cdb; only the
imports and the profile lookup (entries.json instead of the profile files) changed.
"""

import functools
import json
import os
import re
import unicodedata

from lm_eval.tasks.ifeval import instructions_registry

from . import checkers, manifest
from .checkers import Checker
from .data_types import DATA_TYPES, NATIVE_PREFIX

HERE = os.path.dirname(os.path.abspath(__file__))
# no word spaces: a character count stands in for a word count (notes/multilingual_ifeval_eval.md: open choice)
CHARACTERS_PER_WORD = 1.5
COMMAS = ",،，、"
# sentence ends of the languages in the manifest's own_sentence_end group, and the common ones
SENTENCE_END = re.compile(r"[.!?…。！？।॥۔؟።։။។]+(?=\s|$|[^\x00-\x7f]|[\"')\]])")
QUOTED_OPTION = re.compile(r"['\"‘’“”«»„‚「」『』]\s*([^'\"‘’“”«»„‚「」『』]+?)\s*['\"‘’“”«»„‚「」『』]")
# an apostrophe between two letters belongs to a word (l'entraînement, don't), it does not open or close a quotation
WORD_APOSTROPHE = re.compile(r"(?<=\w)['’](?=\w)")




def _registry():
    """IFEval's checker classes (this harness's lm_eval/tasks/ifeval)."""
    return instructions_registry.INSTRUCTION_DICT


class UppercaseAnyLanguage(Checker):
    description = "All capitals."

    def check_following(self, value):
        # ß has no one-letter capital in common use: "STRAßE" is all capitals (history 255, B)
        return value.replace("ß", "ẞ").isupper()


class LowercaseAnyLanguage(Checker):
    description = "All lower case."

    def check_following(self, value):
        return value.islower()


class ConstrainedResponseTranslated(Checker):
    """The response contains one of the options the (translated) prompt gives in quotes."""

    description = "Answer with one of the options."

    def check_following(self, value):
        plain = lambda text: WORD_APOSTROPHE.sub("\x00", text)
        value = plain(value)
        return any(_whole(option, value) for option in self.options(plain(self.kwargs.get("prompt", ""))))

    @staticmethod
    def options(prompt):
        """The quoted options: those of the sentence (or line) that quotes the most, not a title quoted elsewhere in the
        question (pl „Zimna wojna”, sr "Na Drini ćuprija", history 255, G)."""
        parts = [p for p in re.split(r"(?<=[.!?。？！؟।])\s+|\n", prompt) if p.strip()]
        best = max(parts, key=lambda p: len(QUOTED_OPTION.findall(p)), default="")
        return QUOTED_OPTION.findall(best) if len(QUOTED_OPTION.findall(best)) >= 2 else QUOTED_OPTION.findall(prompt)


class NoCommaMultilingual(Checker):
    description = "No commas."

    def check_following(self, value):
        return not any(c in value for c in COMMAS)


def _whole(option, value):
    """The option in the value, not inside a longer word ("No" not in "Nothing", "Da" not in "Danas", 예 not in 예약);
    a copula attached after it is allowed (예입니다)."""
    for m in re.finditer(re.escape(option), value):
        before = value[m.start() - 1] if m.start() else ""
        after = value[m.end()] if m.end() < len(value) else ""
        if before.isalpha() and option[:1].isalpha():
            continue
        if after.isalpha() and option[-1:].isalpha() and not value[m.end():].startswith(COPULAS):
            continue
        return True
    return False


# endings a language without spaces before them puts right after an answer word: 예입니다, はいです
COPULAS = ("입니다", "이에요", "예요", "요.", "요!", "요 ", "です", "だ", "是的")


def _compare(count, relation, threshold):
    return count < threshold if relation == "less than" else count >= threshold


class NumberSentencesMultilingual(Checker):
    description = "{relation} {num_sentences} sentences."

    def check_following(self, value):
        # a period after digits inside a date or number ends no sentence: ko 2026. 11. 25.에 (history 255, D)
        value = re.sub(r"(?<=\d)\.(?=\s?\d|[^\s.!?])", "", value)
        return _compare(len(SENTENCE_END.findall(value)), self.kwargs["relation"], self.kwargs["num_sentences"])


# closing marks an answer may put around the end phrase it was shown in marks (« », < >, 「」; history 255, H)
CLOSING = "\"'”’»›」』>）)"
FINAL_PUNCTUATION = ".!?。！？।؟…"


class EndCheckerMultilingual(Checker):
    """IFEval's end phrase, with the quotation marks or brackets the request showed the phrase in allowed around it,
    and, for a phrase without final punctuation, the answer's final full stop (zh 期待您的回复。)."""
    description = "End your response with the exact phrase {end_phrase}."

    def check_following(self, value):
        phrase = unicodedata.normalize("NFC", self.kwargs["end_phrase"]).strip().lower()
        text = unicodedata.normalize("NFC", value).strip().lower()
        for _ in range(3):
            text = text.rstrip().rstrip(CLOSING).rstrip()
            if phrase[-1:] not in FINAL_PUNCTUATION:
                text = text.rstrip(FINAL_PUNCTUATION).rstrip()
        phrase = phrase.strip(CLOSING + "\"«‹「『<（(“„‘‚").strip()
        return text.endswith(phrase)


class NumberCharacters(Checker):
    description = "{relation} {num_words} words, as characters."

    def check_following(self, value):
        count = sum(unicodedata.category(c)[0] in "LN" for c in value)
        return _compare(count, self.kwargs["relation"], self.kwargs["num_words"] * CHARACTERS_PER_WORD)


def _paragraph(value, nth):
    """The nth paragraph (IFEval's split at a blank line; empty ones not counted) and the paragraph count."""
    paragraphs = re.split(r"\n\n", value)
    count = sum(bool(p.strip()) for p in paragraphs)
    if nth > count or not paragraphs[nth - 1].strip():
        return None, count
    return paragraphs[nth - 1].strip(), count


def _bare(text):
    """Text without punctuation or symbols at its ends (quotation marks of any language, colons, dashes)."""
    text = unicodedata.normalize("NFC", text)
    while text and unicodedata.category(text[0])[0] in "PSZ":
        text = text[1:]
    while text and unicodedata.category(text[-1])[0] in "PSZ":
        text = text[:-1]
    return without_optional_marks(text).lower()


# Arabic-script short vowels, tanween and tatweel: optional in writing, so مرحباً is مرحبا and ماءً is ماء (history 255, 277)
OPTIONAL_MARKS = re.compile("[\u064b-\u065f\u0670\u0640]")


def without_optional_marks(text):
    return OPTIONAL_MARKS.sub("", text)


class NthParagraphFirstWordMultilingual(Checker):
    """IFEval's paragraph first word, with any punctuation around the word ignored: IFEval cuts the word only at
    . , ? ! ' and \" and so failed „Zaključak" and Fazit: (pilot 3956996)."""
    description = "{num_paragraphs} paragraphs; paragraph {nth_paragraph} starts with the word {first_word}."

    def check_following(self, value):
        paragraph, count = _paragraph(value, self.kwargs["nth_paragraph"])
        if paragraph is None or count != self.kwargs["num_paragraphs"]:
            return False
        return _bare(paragraph.split()[0]) == _bare(self.kwargs["first_word"])


class NthParagraphStartsWith(Checker):
    """The same for scripts written without spaces between words (ja, zh): the paragraph, its leading punctuation
    left out, starts with the word (IFEval's first word is the text up to the first space: the whole sentence)."""
    description = "{num_paragraphs} paragraphs; paragraph {nth_paragraph} starts with {first_word}."

    def check_following(self, value):
        paragraph, count = _paragraph(value, self.kwargs["nth_paragraph"])
        if paragraph is None or count != self.kwargs["num_paragraphs"]:
            return False
        word = _bare(self.kwargs["first_word"])
        start = unicodedata.normalize("NFC", paragraph).lower()
        while start and unicodedata.category(start[0])[0] in "PSZ":
            start = start[1:]
        return bool(word) and start.startswith(word)


class NthParagraphFirstWordPrefix(NthParagraphFirstWordMultilingual):
    """The same, for a language that attaches particles or endings to the word (ko 보안은 for 보안, history 255, E):
    the first word starts with the asked word."""

    def check_following(self, value):
        paragraph, count = _paragraph(value, self.kwargs["nth_paragraph"])
        if paragraph is None or count != self.kwargs["num_paragraphs"]:
            return False
        word = _bare(self.kwargs["first_word"])
        return bool(word) and _bare(paragraph.split()[0]).startswith(word)


class ForbiddenWordsMultilingual(Checker):
    """IFEval's forbidden words, with word boundaries that know combining marks: IFEval's \\b treats a vowel sign,
    nukta or virama as a non-word character, so a Hindi word ending in one (कचरा, गंदगी, पेड़) was never found and the
    ban never enforced (history 277). Whole words by native.checkers.word_spans (letters, digits and marks inside)."""
    description = "Do not include the words {forbidden_words}."

    def check_following(self, value):
        from .checkers import word_spans
        return not word_spans(without_optional_marks(value), [without_optional_marks(w) for w in self.kwargs["forbidden_words"]])


class ForbiddenWordsAtWordStart(Checker):
    """The same for a language that attaches particles or endings to a word (ko 지하철을 for 지하철): the forbidden word
    starts a word, whatever follows it (history 280); not found inside a longer word (도시철도 has no 철도 at its start)."""
    description = "Do not include the words {forbidden_words}."

    def check_following(self, value):
        inside = lambda c: c.isalnum() or c == "_" or unicodedata.category(c).startswith("M")
        text = unicodedata.normalize("NFC", value).lower()
        for word in self.kwargs["forbidden_words"]:
            word = unicodedata.normalize("NFC", word).lower()
            for m in re.finditer(re.escape(word), text) if word else ():
                if m.start() == 0 or not inside(text[m.start() - 1]):
                    return False
        return True


# grammar words a script writes onto a word (history 284): before it (proclitics) and after it (pronoun endings). A use of
# the forbidden word carries only these: Arabic والكتاب, بالكتاب, للكتاب, كتابي for كتاب; not كتابة ("writing"), معلم for علم
CLITICS = {
    "Arab": {"before": re.compile("^(?:[وف])?(?:[بك]?ال|لل|[بكل])?$"),
             "after": re.compile("^(?:ي|ه|ها|هما|هم|هن|نا|ك|كما|كم|كن)?$")},
}


class ForbiddenWordsClitics(Checker):
    """IFEval's forbidden words for a language whose script writes grammar words onto the word (CLITICS of its
    script): the word is used if only clitics stand next to it inside the written word (ar والكتاب). Optional marks
    left out (OPTIONAL_MARKS)."""
    description = "Do not include the words {forbidden_words}."

    def check_following(self, value):
        from . import sanity
        clitics = CLITICS.get(sanity.script_of(self.kwargs.get("code", "")), {})
        inside = lambda c: c.isalnum() or unicodedata.category(c).startswith("M")
        text = without_optional_marks(unicodedata.normalize("NFC", value))
        for word in self.kwargs["forbidden_words"]:
            word = without_optional_marks(unicodedata.normalize("NFC", word))
            core = word[2:] if word.startswith("ال") and len(word) > 3 else word      # الشاشات is used in وبالشاشات
            for m in re.finditer(re.escape(core), text) if core else ():
                start, end = m.span()
                while start > 0 and inside(text[start - 1]):
                    start -= 1
                while end < len(text) and inside(text[end]):
                    end += 1
                before, after = text[start:m.start()], text[m.end():end]
                if clitics.get("before", re.compile("^$")).match(before) and clitics.get("after", re.compile("^$")).match(after):
                    return False
        return True


class ForbiddenWordsAnywhere(Checker):
    """The same for a language written without spaces between words (ja, zh): no word boundary exists, IFEval's \\b
    never fired there; a forbidden word is found anywhere in the text (history 277)."""
    description = "Do not include the words {forbidden_words}."

    def check_following(self, value):
        text = unicodedata.normalize("NFC", value).lower()
        return not any(unicodedata.normalize("NFC", w).lower() in text for w in self.kwargs["forbidden_words"] if w)


REWRITTEN = {"uppercase_any_language": UppercaseAnyLanguage, "lowercase_any_language": LowercaseAnyLanguage,
             "constrained_response_translated": ConstrainedResponseTranslated, "no_comma_multilingual": NoCommaMultilingual,
             "number_sentences_multilingual": NumberSentencesMultilingual, "number_characters": NumberCharacters,
             "nth_paragraph_first_word_multilingual": NthParagraphFirstWordMultilingual,
             "nth_paragraph_starts_with": NthParagraphStartsWith, "nth_paragraph_first_word_prefix": NthParagraphFirstWordPrefix,
             "end_checker_multilingual": EndCheckerMultilingual, "forbidden_words_multilingual": ForbiddenWordsMultilingual,
             "forbidden_words_anywhere": ForbiddenWordsAnywhere, "forbidden_words_at_word_start": ForbiddenWordsAtWordStart,
             "forbidden_words_clitics": ForbiddenWordsClitics}


@functools.lru_cache(maxsize=None)
def _entries():
    """{code: {entry id: {"checker": class name, "type": type id}}} of the language profiles the dataset was made
    with (built from the run's saved profiles: profile.checker_class(entry).__name__, entry type or id)."""
    with open(os.path.join(HERE, "entries.json"), encoding="utf-8") as handle:
        return json.load(handle)


def profile_entry(code, key):
    entries = _entries().get(code, {})
    if key not in entries:
        raise KeyError(f"native:{key}: no entry {key!r} in the {code} profile")
    return entries[key]


def checker(code, instruction_id, kwargs, prompt):
    """A built checker for one instruction of a row. code: language code, "en" for the English original."""
    kwargs = {k: v for k, v in kwargs.items() if v}
    if instruction_id.startswith(NATIVE_PREFIX):
        key = instruction_id[len(NATIVE_PREFIX):]
        if key in DATA_TYPES:                     # native:<type>: rows of the catalog route
            return checkers.build(DATA_TYPES[key]["checker"], key, kwargs)
        entry = profile_entry(code, key)          # native:<entry id>: rows made from a language profile
        return checkers.build(getattr(checkers, entry["checker"]), entry["type"], kwargs)
    if code != "en":
        action = manifest.entry(code, instruction_id)
        if action["action"] == "rewrite_checker" or action.get("checker") in REWRITTEN:
            built = REWRITTEN[action["checker"]](instruction_id)
            built.kwargs = {**kwargs, "prompt": prompt, "code": code}
            return built
    built = _registry()[instruction_id](instruction_id)
    built.build_description(**kwargs)
    args = built.get_instruction_args()
    if args and "prompt" in args:
        built.build_description(prompt=prompt)
    return built


def loose_versions(response):
    """IFEval's loose variants: first / last line removed, asterisks removed."""
    lines = response.split("\n")
    trimmed = ["\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    plain = [response, *trimmed]
    return plain + [text.replace("*", "") for text in plain]


def score(row, response, code):
    """{"strict": [bool per instruction], "loose": [bool per instruction]}."""
    strict, loose = [], []
    for instruction_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
        built = checker(code, instruction_id, kwargs, row["prompt"])
        strict.append(bool(response.strip() and built.check_following(response)))
        loose.append(any(text.strip() and built.check_following(text) for text in loose_versions(response)))
    return {"strict": strict, "loose": loose}
