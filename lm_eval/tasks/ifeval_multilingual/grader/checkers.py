"""
Checkers for the language-specific instruction types (data_types.py).

Same interface as IFEval's checkers (lm_eval/tasks/ifeval/instructions.py): build_description(**kwargs)
sets the kwargs, check_following(response) -> bool. Kept here until they move to lm-eval-harness.
"""

import functools
import json
import os
import re
import unicodedata


class Checker:
    def __init__(self, instruction_id):
        self.id = instruction_id
        self.kwargs = {}

    def build_description(self, **kwargs):
        self.kwargs = kwargs
        return self.description.format(**kwargs)

    def get_instruction_args(self):
        return self.kwargs


def build(checker_class, instruction_id, kwargs, values=None):
    """A checker with its settings and the numbers drawn for one item. The one place a checker of this file is built.
    Example: build(SentenceEndChecker, "punctuation:sentence_end", {"mark": "।", "foreign": "."}, {"min_count": 3})"""
    built = checker_class(instruction_id)
    built.build_description(**kwargs, **(values or {}))
    return built


class NativeQuotesChecker(Checker):
    """Quotes use the language's own marks (e.g. „…“, «…», 「…」), never ASCII double quotes."""

    description = "Put every quotation in {open}…{close}, never in ASCII double quotes."

    def check_following(self, value):
        opening, closing = self.kwargs["open"], self.kwargs["close"]
        if opening not in value or closing not in value or '"' in value:
            return False
        if opening == closing:     # one mark for both sides (Finnish ” ”): a full pair, and not the English opening mark
            return value.count(opening) % 2 == 0 and "“" not in value
        if len(opening) != 1 or len(closing) != 1:
            return True
        # order matters: Arabic ” “ uses the same two characters as English “ ”
        marks = [c for c in value if c in (opening, closing)]
        return marks[0] == opening and marks[-1] == closing


# letters a reader of the script reads as one letter of their own, though stored as several characters: Hindi ज्ञ (gya)
# is ज + virama + ञ, and is neither ज nor ञ to the reader (history 261)
OWN_LETTERS = ("\u091c\u094d\u091e",)


def letter_count(value, letter):
    """Occurrences of a letter in both cases. Normalized first (é may be stored as e + accent); ß has no one-letter
    upper case, and lower() would turn Turkish İ into two characters, so the two cases are counted directly.
    A letter followed by a nukta is another letter (Hindi ढ and ढ़, ड and ड़, ज and ज़: NFC stores ढ़ as ढ + nukta),
    and the letters inside OWN_LETTERS are not counted (history 261)."""
    value = unicodedata.normalize("NFC", value)
    letter = unicodedata.normalize("NFC", letter).lower()
    for own in OWN_LETTERS:
        if own != letter:
            value = value.replace(own, " ")
    upper = letter.upper()
    forms = [letter] + ([upper] if upper != letter and len(upper) == len(letter) else [])
    return sum(len(re.findall(re.escape(form) + NOT_NUKTA, value)) for form in forms)


NUKTAS = "".join(chr(c) for c in range(0x0900, 0x0DFF) if unicodedata.name(chr(c), "").endswith("SIGN NUKTA"))
NOT_NUKTA = f"(?![{NUKTAS}])"


class NativeLetterFrequencyChecker(Checker):
    """A letter English lacks (e.g. ß, ñ, ç) appears at least / less than / exactly N times."""

    description = "Use the letter {letter} {relation} {frequency} times."

    def check_following(self, value):
        count = letter_count(value, self.kwargs["letter"])
        if self.kwargs["relation"] == "exactly":
            return count == self.kwargs["frequency"]
        if self.kwargs["relation"] == "at least":
            return count >= self.kwargs["frequency"]
        return count < self.kwargs["frequency"]


class ForbiddenLettersChecker(Checker):
    """None of the given letters appear (e.g. German umlauts written ae / oe / ue)."""

    description = "Do not use the letters {letters}."

    def check_following(self, value):
        return not any(letter_count(value, letter) for letter in self.kwargs["letters"])


class PatternChecker(Checker):
    """At least min_count matches of the required patterns (regexes, case-insensitive), none of the forbidden ones.
    Base of the number checkers below; also named directly by a hand-written profile entry (checker: pattern), which
    may give only one of the two lists and no min_count (then 1)."""

    description = "At least {min_count} matches of the required patterns, none of the forbidden ones."

    def build_description(self, **kwargs):
        self.kwargs = kwargs
        return self.description.format(**{"min_count": 1, **kwargs})

    def check_following(self, value):
        required, forbidden = self.kwargs.get("required", []), self.kwargs.get("forbidden", [])
        if any(re.search(p, value, re.I) for p in forbidden):
            return False
        return not required or sum(len(list(re.finditer(p, value, re.I))) for p in required) >= self.kwargs.get("min_count", 1)


class LocalFormatChecker(PatternChecker):
    """Large numbers written the local way (regexes), none the foreign way."""

    description = "Write at least {min_count} {what} in the local format, none in another format."


class NativeDigitsChecker(PatternChecker):
    """Numbers in the language's own digits (regexes), none in 0-9. Example (Hindi): required ["[०-९]+"], forbidden ["[0-9]"]"""

    description = "Write at least {min_count} numbers in the language's own digits, none in 0-9."


class ScriptVariantChecker(Checker):
    """
    Text entirely in one script variant (Chinese Traditional: OpenCC s2t leaves it unchanged) and at least
    min_count characters that differ between the variants (text made only of shared characters proves nothing).
    conversion: OpenCC config converting INTO the required variant: "s2tw" Traditional (Taiwan standard), "s2hk"
    Traditional (Hong Kong standard), "t2s" Simplified. Not the generic "s2t": it uses old variants (喫 for 吃).
    """

    description = "Write only in the {conversion} script variant."

    # regional standards: generic s2t uses old variants (吃 -> 喫) and would reject modern Traditional text
    REVERSE = {"s2tw": "tw2s", "s2hk": "hk2s", "t2s": "s2tw", "s2t": "t2s"}

    def check_following(self, value):
        from . import deps
        deps.ensure("opencc")
        from opencc import OpenCC
        into = OpenCC(self.kwargs["conversion"])
        back = OpenCC(self.REVERSE[self.kwargs["conversion"]])
        if into.convert(value) != value:
            return False
        specific = sum(back.convert(c) != c for c in value)
        return specific >= self.kwargs.get("min_count", 1)


class TargetScriptChecker(Checker):
    """
    Written in another script of the language (Hindi in Latin, Serbian in Latin, Kazakh in Arabic): no letter of the
    avoided script, at least one of the target. target / avoid: Unicode name prefixes, e.g. ["LATIN"], ["DEVANAGARI"].
    """

    description = "Write in the {target} script, with no {avoid} letters."

    def check_following(self, value):
        names = [unicodedata.name(c, "") for c in value if c.isalpha()]
        target, avoid = tuple(self.kwargs["target"]), tuple(self.kwargs["avoid"])
        return any(n.startswith(target) for n in names) and not any(n.startswith(avoid) for n in names)


class SentenceEndChecker(Checker):
    """
    Sentences end with the language's own mark (Hindi danda ।, Urdu ۔, Armenian ։, Amharic ።…), not with the
    ASCII one: at least min_count native marks, and the foreign mark never used as a sentence end (after text,
    followed by a space, the end or a non-ASCII character, since Chinese / Japanese put no space after a sentence).
    Decimals (3.5), "..." and list numbers ("1.") are not sentence ends.
    Example (Hindi): mark "।", foreign "."
    """

    description = "End sentences with {mark}, never with {foreign}: at least {min_count} times."

    def check_following(self, value):
        native = value.count(self.kwargs["mark"])
        f = re.escape(self.kwargs.get("foreign", "."))
        misused = re.search(rf"(?<![\d\s.]){f}(?!{f})(?=\s|$|[^\x00-\x7f])", value)
        return native >= self.kwargs.get("min_count", 1) and not misused


# combining marks per named set, and the base letters they attach to
MARK_SETS = {
    # Arabic harakat (fatha… sukun, tanween, shadda) + superscript alef
    "arabic_harakat": (lambda c: "ً" <= c <= "ْ" or c == "ٰ",
                       lambda c: "ء" <= c <= "ي" or "ٱ" <= c <= "ۓ"),
    # Hebrew niqqud (vowel points, dagesh, shin / sin dots)
    "hebrew_niqqud": (lambda c: "ְ" <= c <= "ֽ" or c in "ׇֿׁׂׅׄ",
                      lambda c: "א" <= c <= "ת"),
}


class DiacriticsChecker(Checker):
    """
    Every word of the script carries a vowel mark (Arabic tashkeel, Hebrew niqqud): yes / no, no ratio. Text is
    NFD-normalized first. Words without a letter of the script (digits, foreign names) are ignored.
    """

    description = "Write every word with its {marks}."

    def check_following(self, value):
        is_mark, is_letter = MARK_SETS[self.kwargs["marks"]]
        words = [w for w in unicodedata.normalize("NFD", value).split() if any(map(is_letter, w))]
        return bool(words) and all(any(map(is_mark, w)) for w in words)


def count_form(value, stanza_code, hosts, feature):
    """Words of the given word classes that the Stanza tagger labels with the feature (e.g. Tense=Imp)."""
    from . import register          # the tagger is loaded on first use, once per language
    return sum(has_form(word, hosts, feature)
               for sentence in register.tagger(stanza_code)(value).sentences for word in sentence.words)


def has_form(word, hosts, feature):
    """A tagged word of one of the word classes, labelled with the feature (Tense=Imp)."""
    return word.upos in hosts and feature in (word.feats or "").split("|")


class GrammarFormChecker(Checker):
    """
    At least min_count words in a grammatical form of the language (imperfect, subjunctive, instrumental case…),
    counted by the Stanza tagger. Which forms a language has, and that the tagger recognises them, comes from
    grammar_facts.json (build_grammar_facts.py). Only "at least": a tagger cannot prove that a form is absent.
    Example (Spanish): feature "Tense=Imp", hosts ["VERB", "AUX"]
    """

    description = "Use at least {min_count} {host_name} in {name}."

    def check_following(self, value):
        found = count_form(value, self.kwargs["stanza"], self.kwargs["hosts"], self.kwargs["feature"])
        return found >= self.kwargs.get("min_count", 1)


class NativeMarkChecker(Checker):
    """
    At least min_count questions (or exclamations) written with the language's own marks. role "opens": the native
    mark opens what the Latin mark closes (Spanish ¿…?), and only text between the two counts. role "replaces": the
    native mark is written in place of the Latin one (Arabic ؟, Armenian ՞, which sits inside the sentence).
    A question written with the Latin mark alone does not count.
    Example (Spanish): mark "¿", latin "?", role "opens"
    """

    description = "Include at least {min_count} sentences marked with {mark}."

    def check_following(self, value):
        mark, latin = self.kwargs["mark"], self.kwargs["latin"]
        if self.kwargs["role"] == "opens":
            spans = re.findall(rf"{re.escape(mark)}([^{re.escape(latin)}{re.escape(mark)}\n]*){re.escape(latin)}", value)
            found = sum(any(c.isalpha() for c in span) for span in spans)
        else:
            found = value.count(mark)
        return found >= self.kwargs.get("min_count", 1)


@functools.lru_cache(maxsize=1)
def han_facts():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "han_facts.json"), encoding="utf-8") as handle:
        return json.load(handle)


def han(char):
    """(radical number, strokes) of a Chinese character, from Unicode's Unihan table (han_facts.json); None for a
    character outside the CJK Unified Ideographs block."""
    facts = han_facts()
    first, last = facts["block"]
    if not first <= ord(char) <= last:
        return None
    return facts["radical"][ord(char) - first], facts["strokes"][ord(char) - first]


def with_radical(value, number):
    """The characters of the text that Unicode classifies under the radical (Kangxi number)."""
    return [c for c in value if (han(c) or (0, 0))[0] == number]


class ForbiddenRadicalChecker(Checker):
    """No character of the given radical (e.g. the water radical 水 / 氵: 法, 海, 港). Example: radical "水", number 85"""

    description = "Use no character whose radical is {radical}."

    def check_following(self, value):
        return not with_radical(value, self.kwargs["number"])


class RadicalCountChecker(Checker):
    """At least min_count characters of the given radical. Example: radical "心", number 61, min_count 4"""

    description = "Use at least {min_count} characters whose radical is {radical}."

    def check_following(self, value):
        return len(with_radical(value, self.kwargs["number"])) >= self.kwargs.get("min_count", 1)


class MaxStrokesChecker(Checker):
    """Every Chinese character of the response has at most max_strokes strokes; at least one such character."""

    description = "Use only characters with at most {max_strokes} strokes."

    def check_following(self, value):
        strokes = [han(c)[1] for c in value if han(c)]
        return bool(strokes) and max(strokes) <= self.kwargs["max_strokes"]


class LemmaEndingChecker(Checker):
    """
    At least min_count nouns whose base form (lemma, from the Stanza tagger) ends in one of the given endings. German
    diminutives: endings "chen", "lein". The lemma removes the words that only look the part in an inflected form
    (Wochen -> Woche, Gesprächen -> Gespräch); the word class removes verbs (machen). What remains are nouns whose
    base form really has the ending without being formed with it (Kuchen, Zeichen, das Lachen): not_endings and
    not_words list those. The list is not complete: this check is approximate.
    Example: stanza "de", endings ["chen", "lein"], not_endings ["kuchen", "zeichen", "machen", ...], not_words ["rauchen"]
    """

    description = "Use at least {min_count} nouns ending in {endings}."

    def forms(self, value):
        """The nouns counted, as written."""
        from . import register
        endings, not_endings = tuple(self.kwargs["endings"]), tuple(self.kwargs.get("not_endings", []))
        not_words, out = set(self.kwargs.get("not_words", [])), []
        for sentence in register.tagger(self.kwargs["stanza"], self.kwargs.get("package"), lemma=True)(value).sentences:
            for word in sentence.words:
                lemma = (word.lemma or word.text).lower()
                if word.upos == "NOUN" and lemma.endswith(endings) and not lemma.endswith(not_endings) and lemma not in not_words:
                    out.append(word.text)
        return out

    def check_following(self, value):
        return len(self.forms(value)) >= self.kwargs.get("min_count", 1)


class TaggedFormChecker(Checker):
    """
    At least min_count words that the Stanza tagger labels with every feature given: "Mood=Sub" a feature with its
    value, "Person" the feature with any value. With `after`, a word counts only if one of these words or phrases ends
    at most `window` words before it and the tagger labels its last word with one of after_upos: the Portuguese
    future subjunctive after se / quando, where se the pronoun (PRON) is not a trigger and the regular forms are
    spelled as the infinitive (se ele chegar). after_pronoun: a trigger tagged as a pronoun also counts when the word
    before it is a pronoun too (Stanza tags se after an enclitic as one: ligo-te se puder). prefixes: only words that
    begin so (Indonesian passive di-, not ter-). none: true turns the rule into a ban, the response passes only if no
    such word is found (Indonesian active voice only: no di- verb tagged Voice=Pass); a tagger error then fails it.
    Example (Portuguese): stanza "pt", features ["Mood=Sub", "Tense=Fut"], after ["se", "quando", "assim que"], window 4, after_pronoun true
    Example (Indonesian): stanza "id", features ["Voice=Pass"], prefixes ["di"], none true
    """

    description = "Use at least {min_count} words in the form {features}."

    def build_description(self, **kwargs):
        self.kwargs = kwargs
        return self.description.format(**{"min_count": 1, **kwargs})

    def matches(self, feats):
        have = dict(f.split("=", 1) for f in (feats or "").split("|") if "=" in f)
        return all(have.get(f.split("=", 1)[0]) == f.split("=", 1)[1] if "=" in f else f in have for f in self.kwargs["features"])

    def forms(self, value):
        """The words counted, as written."""
        from . import register
        hosts = set(self.kwargs.get("hosts", ["VERB", "AUX"]))
        after = [tuple(phrase.lower().split()) for phrase in self.kwargs.get("after", [])]
        after_upos, window, out = set(self.kwargs.get("after_upos", ["SCONJ", "ADV"])), self.kwargs.get("window", 4), []
        after_pronoun = self.kwargs.get("after_pronoun", False)
        prefixes = tuple(p.lower() for p in self.kwargs.get("prefixes", []))
        for sentence in register.tagger(self.kwargs["stanza"], self.kwargs.get("package"))(value).sentences:
            words = sentence.words
            lower = [w.text.lower() for w in words]

            def triggered(i):
                for end in range(max(0, i - window), i):
                    for phrase in after:
                        start = end - len(phrase) + 1
                        if start >= 0 and tuple(lower[start:end + 1]) == phrase and (
                                words[end].upos in after_upos
                                # Stanza tags se after an enclitic as a pronoun (ligo-te se puder): a pronoun right before it
                                or (after_pronoun and words[end].upos == "PRON" and end > 0 and words[end - 1].upos == "PRON")):
                            return True
                return False

            out += [w.text for i, w in enumerate(words) if w.upos in hosts and self.matches(w.feats) and (not after or triggered(i))
                    and (not prefixes or w.text.lower().startswith(prefixes))]
        return out

    def check_following(self, value):
        if self.kwargs.get("none"):
            return not self.forms(value)
        return len(self.forms(value)) >= self.kwargs.get("min_count", 1)


SERBIAN_LATIN = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e", "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k",
                 "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "ћ": "ć", "у": "u",
                 "ф": "f", "х": "h", "ц": "c", "ч": "č", "џ": "dž", "ш": "š"}


@functools.lru_cache(maxsize=1)
def yat_forms():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yat_forms.json"), encoding="utf-8") as handle:
        return json.load(handle)


@functools.lru_cache(maxsize=1)
def past_forms():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "past_forms.json"), encoding="utf-8") as handle:
        return {language: {kind: frozenset(forms) for kind, forms in lists.items()} for language, lists in json.load(handle).items()
                if language != "source"}


class SimplePastChecker(Checker):
    """
    Narration in the simple past (Serbian aorist: dođoh, rekoše) and not in the perfect (došao sam, rekli su): at
    least min_count simple-past forms, and no perfect. The forms come from past_forms.json (build_past_forms.py, from
    a morphological lexicon). A perfect is a past participle with one of the given auxiliaries at most `window` words
    away; a participle without one (the conditional bih došao, an adjective) is not a perfect. Cyrillic text is read
    through its Latin transliteration; text between quotation marks is left out (speech quoted inside a narration).
    Not counted: an aorist that is also another form (dođe: present too), so narration in the third person singular
    is undercounted; a perfect whose auxiliary is further away or dropped is not seen.
    Example: facts "sr", auxiliaries ["sam", "si", "je", "smo", "ste", "su"], window 3
    """

    description = "Narrate in the simple past: at least {min_count} such forms, and no perfect."

    def forms(self, value):
        """(simple-past forms, perfects) found in the response."""
        from . import register
        lists = past_forms()[self.kwargs["facts"]]
        text = "".join(SERBIAN_LATIN.get(c, c) for c in register.strip_quoted_spans(value).lower())
        tokens = re.findall(r"[^\W\d_]+", text)
        auxiliaries, window = set(self.kwargs["auxiliaries"]), self.kwargs.get("window", 3)
        perfects = [w for i, w in enumerate(tokens) if w in lists["participles"]
                    and auxiliaries & set(tokens[max(0, i - window):i + window + 1])]
        return [w for w in tokens if w in lists["simple_past"]], perfects

    def check_following(self, value):
        simple, perfects = self.forms(value)
        return not perfects and len(simple) >= self.kwargs.get("min_count", 1)


@functools.lru_cache(maxsize=None)
def morph_analyzer(lang):
    """pymorphy3 with its OpenCorpora dictionary (ru) or the Ukrainian one (uk): the aspect of a verb is in the
    dictionary, not guessed from context (history 322: 0.1 % wrong against 2-3 % for Stanza)."""
    from . import deps
    deps.ensure("pymorphy3")
    import pymorphy3
    return pymorphy3.MorphAnalyzer(lang=lang)


PYMORPHY_VERBAL = {"VERB", "INFN", "GRND", "PRTF", "PRTS"}


class AspectOnlyChecker(Checker):
    """
    Verbs of one aspect only (Russian, Ukrainian): at least min_count verbs of the wanted aspect (impf, perf) and none
    of the other. The tagger (Stanza) finds the verbs (VERB, AUX); the aspect of each comes from pymorphy3's
    dictionary: the set of aspects over its verbal readings. A verb whose readings have both aspects (biaspectual
    использовать, or two verbs with one form) neither counts nor fails; lemmas in `exempt` (быть: the past of the
    narrative and every present form are imperfective) are left out. Text between quotation marks is left out.
    Example: stanza "ru", analyzer "ru", aspect "impf", exempt []
    """

    description = "Use only verbs of the {aspect} aspect: at least {min_count}, and none of the other."

    def verbs(self, value):
        """(wanted, other): verbs of the wanted aspect and of the other one."""
        from . import register
        morph, wanted = morph_analyzer(self.kwargs["analyzer"]), self.kwargs["aspect"]
        exempt = set(self.kwargs.get("exempt", []))
        found, other = [], []
        for sentence in register.tagger(self.kwargs["stanza"])(register.strip_quoted_spans(value)).sentences:
            for word in sentence.words:
                if word.upos not in ("VERB", "AUX"):
                    continue
                readings = [p for p in morph.parse(word.text.lower()) if p.tag.POS in PYMORPHY_VERBAL and p.tag.aspect]
                if exempt & {p.normal_form for p in readings}:
                    continue
                aspects = {p.tag.aspect for p in readings}
                if aspects == {wanted}:
                    found.append(word.text)
                elif len(aspects) == 1:
                    other.append(word.text)
        return found, other

    def check_following(self, value):
        found, other = self.verbs(value)
        return not other and len(found) >= self.kwargs.get("min_count", 1)


def only_case(morph, word, case):
    """The word is known to the dictionary and every reading of it is in the case (pymorphy3 tag, e.g. "voct"):
    Іване, друже, мамо; not лікарю (also dative and locative). Words the dictionary does not know are guessed by
    pymorphy3 (уже, нафто-), so they never count."""
    word = word.lower()
    return morph.word_is_known(word) and all(case in p.tag for p in morph.parse(word))


class OnlyCaseChecker(Checker):
    """
    At least min_count nouns (or names) in a case that their form alone shows: every pymorphy3 reading of the word
    is in that case. Made for the Ukrainian vocative (Іване, Олено, друже, пане), which ordinary text almost never
    uses: on UD_Ukrainian-IU precision 91.5 %, recall 27 % (forms shared with another case, лікарю / синку, are not
    counted), 200-word passages with 2 or more 1.9 % (history 340). The tagger (Stanza) finds the nouns: words with a
    vocative reading that are not nouns in the sentence (уже) are left out. Quoted text counts: an address is often
    direct speech.
    Example: stanza "uk", analyzer "uk", case "voct"
    """

    description = "Use at least {min_count} nouns in the {case} case."

    def forms(self, value):
        from . import register
        morph, case = morph_analyzer(self.kwargs["analyzer"]), self.kwargs["case"]
        return [word.text for sentence in register.tagger(self.kwargs["stanza"])(value).sentences for word in sentence.words
                if word.upos in ("NOUN", "PROPN") and word.text.isalpha() and only_case(morph, word.text, case)]

    def check_following(self, value):
        return len(self.forms(value)) >= self.kwargs.get("min_count", 1)


class VariantFormsChecker(Checker):
    """
    The response is in one of two standards of a language that differ in the form of many words: Serbian Ekavian
    (lepo, mleko, mesto) against Ijekavian (lijepo, mlijeko, mjesto). At least min_count words in the wanted standard,
    and no word in the other. The word pairs come from yat_forms.json (build_yat_forms.py: twin forms of a
    morphological lexicon); also_starts / also_words add frequent words of the unwanted standard that the lexicon
    lacks (vrijeme, gdje, prije). Cyrillic text is read through its Latin transliteration; text between quotation
    marks is left out. The list is not complete: an unlisted form of the other standard passes.
    Example: facts "sr", wanted "ekavian", also_starts ["vrijem", "svijet"], also_words ["gdje", "prije"], also_wanted ["vreme", "gde"]
    """

    description = "Write only in the {wanted} standard: at least {min_count} words in its form, none in the other."

    def words(self, value):
        """(wanted, unwanted): the words of the response in the form of the wanted standard and of the other."""
        from . import register
        ijekavian = yat_forms()[self.kwargs["facts"]]["ijekavian"]
        ekavian = set(ijekavian.values())
        text = "".join(SERBIAN_LATIN.get(c, c) for c in register.strip_quoted_spans(value).lower())
        tokens = re.findall(r"[^\W\d_]+", text)
        starts, exact = tuple(self.kwargs.get("also_starts", [])), set(self.kwargs.get("also_words", []))
        more = set(self.kwargs.get("also_wanted", []))      # frequent words of the wanted standard the lexicon lacks
        listed_other = lambda w: w in exact or (bool(starts) and w.startswith(starts))
        if self.kwargs["wanted"] == "ekavian":
            return [w for w in tokens if w in ekavian or w in more], [w for w in tokens if w in ijekavian or listed_other(w)]
        return [w for w in tokens if w in ijekavian or w in more], [w for w in tokens if w in ekavian or listed_other(w)]

    def check_following(self, value):
        wanted, unwanted = self.words(value)
        return not unwanted and len(wanted) >= self.kwargs.get("min_count", 1)


class AddressNumberChecker(Checker):
    """
    The reader is addressed in one grammatical number throughout: at least min_count words that the Stanza tagger
    labels second person (verbs, pronouns, possessives), and every one of them in the given number. Greek formal
    plural (πληθυντικός ευγενείας): number "Plur", so είστε, μπορείτε, σας count and one είσαι or σου fails. Text
    between quotation marks is left out: someone quoted may speak otherwise. The tagger model matters (register.tagger).
    Example (Greek): stanza "el", package "gud", number "Plur"
    """

    description = "Address the reader with second-person forms in the {number} only, at least {min_count} of them."

    def forms(self, value):
        """[(word, number)] of the words labelled second person."""
        from . import register
        nlp = register.tagger(self.kwargs["stanza"], self.kwargs.get("package"))
        out = []
        for sentence in nlp(register.strip_quoted_spans(value)).sentences:
            for word in sentence.words:
                feats = dict(f.split("=", 1) for f in (word.feats or "").split("|") if "=" in f)
                if feats.get("Person") == "2":
                    out.append((word.text, feats.get("Number")))
        return out

    def check_following(self, value):
        forms = self.forms(value)
        return len(forms) >= self.kwargs.get("min_count", 1) and all(number == self.kwargs["number"] for _, number in forms)


class PastPluralEndingChecker(Checker):
    """
    Plural past-tense verbs in one gender form only, told by their ending: at least min_count verbs the Stanza tagger
    labels past tense and plural whose form ends in one of `wanted`, and none ending in one of `unwanted`. Polish
    non-masculine-personal past: wanted ["ły"] (były, poszły), unwanted ["li"] (byli, poszli). The tagger finds the
    past plural forms (it keeps out mały, jeśli, myśli); the ending decides the gender, since the tagger's own gender
    label is wrong on some forms (umarłyśmy, Byłyśmy labelled masculine personal; history 154). Person and conditional
    endings are cut first (`suffixes`: zrobiłybyśmy -> zrobiły). Text between quotation marks is left out.
    Example (Polish): stanza "pl", wanted ["ły"], unwanted ["li"], suffixes ["by", "śmy", "ście"]
    """

    description = "Plural past-tense verbs only in the form ending in {wanted}, at least {min_count} of them, none ending in {unwanted}."

    def forms(self, value):
        """[(word, "wanted" | "unwanted")] of the past plural verbs whose ending is one of either list."""
        from . import register
        nlp = register.tagger(self.kwargs["stanza"], self.kwargs.get("package"))
        cut = re.compile("(?:" + "|".join(map(re.escape, self.kwargs.get("suffixes", []))) + ")*$") if self.kwargs.get("suffixes") else None
        wanted, unwanted, out = tuple(self.kwargs["wanted"]), tuple(self.kwargs["unwanted"]), []
        for sentence in nlp(register.strip_quoted_spans(value)).sentences:
            for word in sentence.words:
                feats = dict(f.split("=", 1) for f in (word.feats or "").split("|") if "=" in f)
                if word.upos not in ("VERB", "AUX") or feats.get("Tense") != "Past" or feats.get("Number") != "Plur":
                    continue
                stem = word.text.lower()
                stem = cut.sub("", stem) if cut and len(stem) > 4 else stem
                if stem.endswith(wanted):
                    out.append((word.text, "wanted"))
                elif stem.endswith(unwanted):
                    out.append((word.text, "unwanted"))
        return out

    def check_following(self, value):
        kinds = [kind for _, kind in self.forms(value)]
        return "unwanted" not in kinds and kinds.count("wanted") >= self.kwargs.get("min_count", 1)


@functools.lru_cache(maxsize=None)
def camel_disambiguator(model):
    """A CAMeL Tools disambiguator (Arabic morphology), loaded once. model: "calima-msa-r13" (MLE, no context, 89 MB) or
    "bert-unfactored-msa" (BERT, reads the sentence; needs torch and transformers, 445 MB). The MLE tags many
    sentence-initial verbs as nouns (وضع, فكر, نُقل): BERT turned 36 of them into verbs on FLORES+ Arabic (history 153)."""
    from . import deps
    deps.ensure("camel_tools", model)
    if model == "bert-unfactored-msa":
        from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
        return BERTUnfactoredDisambiguator.pretrained()
    from camel_tools.disambig.mle import MLEDisambiguator
    return MLEDisambiguator.pretrained(model)


def camel_words(model, value):
    """[(word, analysis)] of the text outside quotation marks: the disambiguator's first analysis of each word."""
    from . import deps
    deps.ensure("camel_tools", model)
    from camel_tools.tokenizers.word import simple_word_tokenize
    from . import register
    return [(word.word, word.analyses[0].analysis if word.analyses else {})
            for word in camel_disambiguator(model).disambiguate(simple_word_tokenize(register.strip_quoted_spans(value)))]


class IdafaChainChecker(Checker):
    """
    Arabic genitive construct chains (إضافة) of min_length nouns or more: at least min_count of them. A chain is a run
    of nouns with no article, no pronoun suffix and no accusative -ا (the heads), closed by a definite noun or adjective
    (article, pronoun suffix or proper noun): مفتاح باب غرفة البيت. A word with a preposition proclitic (بـ لـ كـ)
    breaks the chain. The construct state CAMeL gives (stt) is not read: on unvocalized text it labels many heads
    indefinite (history 145). FLORES+ Arabic: 7 % of 12-sentence chunks have 3 chains of 4+ nouns, 2 % have 4 (history 146).
    Example: analyzer "calima-msa-r13", min_length 4
    """

    description = "Use at least {min_count} genitive construct chains of {min_length} or more nouns."

    def chains(self, value):
        """[[word, ...]] of the chains found, every length."""
        out, run = [], []
        for word, a in camel_words(self.kwargs["analyzer"], value):
            pos = a.get("pos", "")
            nominal = pos.startswith(("noun", "adj"))
            definite = nominal and (a.get("prc0") == "Al_det" or a.get("enc0") not in ("0", "na", None) or pos == "noun_prop")
            if run and a.get("prc1") not in ("0", "na", None):        # a preposition ends the chain
                run = []
            if run and definite:
                out.append(run + [word])
                run = []
            elif pos.startswith("noun") and not definite and not word.endswith(("ا", "ً")):
                run.append(word)
            else:
                run = []
        return out

    def check_following(self, value):
        long_ = [c for c in self.chains(value) if len(c) >= self.kwargs.get("min_length", 4)]
        return len(long_) >= self.kwargs.get("min_count", 1)


class VerbalSentenceChecker(Checker):
    """
    Arabic verbal sentences (الجملة الفعلية): every sentence begins with its verb, and there are at least min_count
    sentences. Sentences end at . ! ? ؟ or a line break; lines without Arabic letters and text between quotation
    marks are left out. The words of skip (negation, tense and emphasis particles: قد, لم, لن, سوف …, compared
    without vowel marks) and a lone و / ف may stand before the verb; a phrase in front of it (في يوم الاثنين، أعلن)
    fails, as does إنّ (pos verb_pseudo). The verb is the first word CAMeL Tools tags pos "verb" (with its
    proclitics: وقال). FLORES+ Arabic: 55 % of sentences begin with a verb, no 12-sentence chunk has all (history 153).
    Example: analyzer "bert-unfactored-msa" (the MLE misses about 4 % of sentence-initial verbs), skip ["قد", "لقد", "لم", "لن", "ما", "لا", "سوف", "ثم"]
    """

    description = "Begin every sentence with its verb (at least {min_count} sentences)."

    def openings(self, value):
        """[(sentence, first word, its pos)] of the sentences of the text."""
        from . import deps
        deps.ensure("camel_tools", self.kwargs["analyzer"])
        from camel_tools.tokenizers.word import simple_word_tokenize
        from camel_tools.utils.dediac import dediac_ar
        from . import register
        skip = {dediac_ar(w) for w in self.kwargs.get("skip", [])}
        skip |= {"و" + w for w in skip} | {"ف" + w for w in skip} | {"و", "ف"}
        out = []
        for sentence in re.split(r"[.!?؟\n]+", register.strip_quoted_spans(value)):
            words = [w for w in simple_word_tokenize(sentence) if re.search("[ء-ي]", w)]
            if not words:
                continue
            for word in camel_disambiguator(self.kwargs["analyzer"]).disambiguate(words):
                if dediac_ar(word.word) not in skip:
                    out.append((sentence.strip(), word.word, word.analyses[0].analysis.get("pos") if word.analyses else None))
                    break
        return out

    def check_following(self, value):
        openings = self.openings(value)
        return len(openings) >= self.kwargs.get("min_count", 1) and all(pos == "verb" for _, _, pos in openings)


class PluralFormChecker(Checker):
    """
    Arabic broken plurals only (جمع تكسير): at least min_count nouns or adjectives that CAMeL Tools reads as plural
    in meaning and singular in form (كتب, رجال, مدن), and none whose form is a sound plural or a dual (معلمون,
    سيارات, كتابين). Duals are banned with the sound plurals because unvocalized -ين is both, and the
    disambiguator read 5 of 20 oblique sound plurals as duals (history 142). Text between quotation marks is left out.
    Example: analyzer "calima-msa-r13"
    """

    description = "Use at least {min_count} broken plurals and no sound plural or dual nouns or adjectives."

    def plurals(self, value):
        """[(word, kind)] of the nouns and adjectives that are plural or dual; kind: broken, sound or dual."""
        out = []
        for word, analysis in camel_words(self.kwargs["analyzer"], value):
            if not analysis.get("pos", "").startswith(("noun", "adj")):
                continue
            if analysis.get("form_num") in ("p", "d"):
                out.append((word, "dual" if analysis.get("form_num") == "d" else "sound"))
            elif analysis.get("num") == "p":
                out.append((word, "broken"))
        return out

    def check_following(self, value):
        kinds = [kind for _, kind in self.plurals(value)]
        return kinds.count("broken") >= self.kwargs.get("min_count", 1) and not set(kinds) - {"broken"}


class RegisterLevelChecker(Checker):
    """
    Sentences end in one politeness level of the language (Korean 합쇼체 / 해요체 / 해라체, Japanese です・ます体). The
    endings of each level come from register_facts.json (UniMorph, build_register_facts.py) and the test from
    register.py: at least 90 % of the sentences end in the level, none in another level; quoted spans are left out.
    Example (Korean): lang "ko", level "FORM+POL" (sentences ending in -ㅂ니다, -ㅂ니까, -ㅂ시다)
    """

    description = "Write every sentence in the {level} politeness level."

    def check_following(self, value):
        from . import register
        return register.ends_in_level(value, register.facts(self.kwargs["lang"]), self.kwargs["level"])


def word_spans(value, words):
    """(start, end) of every whole-word occurrence of the words, case-insensitive. A side of the word that ends in a
    letter, digit or combining mark must not touch another one: "tu" is not found in "tuile", "क" not in "की" (\b fails
    next to combining marks). A side that ends in punctuation is free: "t'" is found in "t'aime"."""
    inside = lambda c: c.isalnum() or c == "_" or unicodedata.category(c).startswith("M")
    value = unicodedata.normalize("NFC", value)
    words = sorted({unicodedata.normalize("NFC", w) for w in words if w}, key=len, reverse=True)
    if not words:
        return []
    spans = []
    for m in re.finditer("|".join(map(re.escape, words)), value, re.I):
        start, end = m.span()
        left = inside(value[start]) and start > 0 and inside(value[start - 1])
        right = inside(value[end - 1]) and end < len(value) and inside(value[end])
        if not left and not right:
            spans.append((start, end))
    return spans


class WordsChecker(Checker):
    """
    Word lists written by hand in a language profile: at least min_count occurrences of the words in required_any
    (if given), no occurrence of a word in forbidden (if given). Whole words, case-insensitive (word_spans).
    Example (French, informal address): required_any ["tu", "te", "toi"], forbidden ["vous", "votre"], min_count 2
    """

    description = "Use at least {min_count} of the words {required_any}, and none of the words {forbidden}."

    def build_description(self, **kwargs):
        self.kwargs = {"required_any": [], "forbidden": [], "min_count": 1, **kwargs}
        return self.description.format(**self.kwargs)

    def check_following(self, value):
        required, forbidden = self.kwargs.get("required_any", []), self.kwargs.get("forbidden", [])
        if word_spans(value, forbidden):
            return False
        return not required or len(word_spans(value, required)) >= self.kwargs.get("min_count", 1)


# checkers that leave text between quotation marks out of the count (register.strip_quoted_spans): an answer wrapped
# whole in quotes (startend:quotation) would be left with nothing to count, so they cannot share a prompt with it
for _quoted in (SimplePastChecker, AspectOnlyChecker, VariantFormsChecker, AddressNumberChecker, PastPluralEndingChecker, IdafaChainChecker,
                VerbalSentenceChecker, PluralFormChecker, RegisterLevelChecker):
    _quoted.strips_quotes = True


# checkers whose rule holds for every sentence or line of the answer: the planner gives them only tasks written in
# running sentences (generate_prompts.native_tasks, LINE_TASKS; history 276)
VerbalSentenceChecker.every_sentence = True


# what a checker with a minimum count (min_count) counts: the planner keeps room for it when a prompt also has an upper
# length limit (generate_prompts.room_needed). "sentence": one sentence per count; "chain": a run of min_length words;
# "word": one word or form per count. A checker used with a count must have one (test_profile)
for _unit, _classes in (("sentence", (SentenceEndChecker, NativeMarkChecker, VerbalSentenceChecker, RegisterLevelChecker)),
                        ("chain", (IdafaChainChecker,)),
                        ("word", (PatternChecker, WordsChecker, LemmaEndingChecker, AddressNumberChecker, PastPluralEndingChecker,
                                  PluralFormChecker, SimplePastChecker, AspectOnlyChecker, OnlyCaseChecker, VariantFormsChecker, GrammarFormChecker, RadicalCountChecker, TaggedFormChecker,
                                  ScriptVariantChecker))):
    for _counted in _classes:
        _counted.unit = _unit
