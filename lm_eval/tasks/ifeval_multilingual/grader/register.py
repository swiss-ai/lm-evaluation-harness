"""
Detects how a text addresses its reader, from register_facts.json (build_register_facts.py). No language has its
own rule here: word lists, restrictions, endings and the signals to trust all come from the facts of the language.

    informal_present / formal_present    number of informal / formal forms in the text (quoted spans left out)
    is_informal        at least MIN_INFORMAL_FORMS informal forms and no formal form
    has_no_informal    no informal form
    ends_in_level      sentences end in one politeness level (Korean, Japanese)

The Stanza tagger is loaded only for a language whose facts need it (a trusted second-person-verb signal, or a form
that counts only as a pronoun).
"""

import functools
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_INFORMAL_FORMS = 2
LEVEL_SHARE = 0.90

# quotation marks that open a span and the marks that may close it
QUOTES = {'"': '"', "“": "”", "„": "“”", "«": "»", "»": "«", "‹": "›", "「": "」", "『": "』"}
WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*['’]?")
SENTENCE = re.compile(r"[^.!?。！？…\n]+[.!?。！？…]*")


@functools.lru_cache(maxsize=None)
def facts(lang):
    with open(os.path.join(HERE, "register_facts.json"), encoding="utf-8") as handle:
        return json.load(handle)["languages"].get(lang)


@functools.lru_cache(maxsize=None)
def lookup_forms(lang):
    path = os.path.join(HERE, "register_lookup.json")
    if not os.path.exists(path):
        return frozenset()
    with open(path, encoding="utf-8") as handle:
        return frozenset(json.load(handle).get(lang, []))


@functools.lru_cache(maxsize=None)
def tagger(code, package=None, lemma=False):
    """The Stanza tagger of a language. package: a model other than the language's default one. The default is often
    trained on news text, which has no informal address: the default Greek model (gdt) labels none of 40 common
    second-person-singular verbs as such, the gud model 38 of them (history 134, 135)."""
    from . import deps
    deps.ensure("stanza")
    import stanza
    # CPU, as when the dataset was made (and the GPU stays with the model being evaluated)
    options = {"verbose": False, "use_gpu": False, "dir": deps.stanza_dir(), **({"package": package} if package else {})}
    more = ",lemma" if lemma else ""       # the base form of each word (Wochen -> Woche), for checks on a word's ending
    with deps.lock():       # the pipeline downloads missing models: one job at a time
        try:
            pipeline = stanza.Pipeline(code, processors="tokenize,mwt,pos" + more, **options)
        except Exception:       # languages without a multiword-token model
            pipeline = stanza.Pipeline(code, processors="tokenize,pos" + more, **options)
    return lambda text: pipeline(uncapitalized(text))


def uncapitalized(text):
    """Text mostly in capitals, lowercased letter by letter (offsets kept: a letter whose lowercase is longer stays).
    Stanza learned on normally cased text: in capitals it tags German words PROPN and reads Spanish TENDRÍA as
    indicative (counts to 0 %); in lowercase counts stay at 94-124 % (history 324, 334). Other text unchanged."""
    cased = [c for c in text if c.isupper() or c.islower()]
    if sum(c.isupper() for c in cased) <= len(cased) / 2:
        return text
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def strip_quoted_spans(text):
    """The text without what stands between quotation marks: words quoted or cited are not the writer's address."""
    out, index = [], 0
    while index < len(text):
        char = text[index]
        if char in QUOTES:
            close = min((p for p in (text.find(c, index + 1) for c in QUOTES[char]) if p != -1), default=-1)
            if close != -1 and "\n\n" not in text[index:close]:
                out.append(" ")
                index = close + 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


def needs_tagger(record):
    return record["signals"]["tagger"]["enabled"] or any(
        "require_pos" in r["rules"] for r in record.get("restrictions", {}).values())


def tokens(text, record):
    """(word, sentence-initial, word classes or None, second-person-singular verb) per word of the text."""
    if needs_tagger(record):
        out = []
        for sentence in tagger(record["stanza"])(text).sentences:
            first = True
            for token in sentence.tokens:
                classes = {w.upos for w in token.words}
                if classes <= {"PUNCT", "SYM"}:
                    continue
                verb = any(w.upos in ("VERB", "AUX") and {"Person=2", "Number=Sing"} <= set((w.feats or "").split("|"))
                           and "Polite=Form" not in (w.feats or "") for w in token.words)
                out.append((token.text, first, classes, verb))
                first = False
        return out
    out = []
    for sentence in SENTENCE.findall(text):
        for index, match in enumerate(WORD.finditer(sentence)):
            out.append((match.group(), index == 0, None, False))
    return out


def matches(word, first, classes, forms, restrictions):
    low = unicodedata.normalize("NFC", word.lower())
    candidates = {low, low.rstrip("'’") + "'"} | {low[:i + 1] for i, c in enumerate(low) if c in "'’"}
    form = next((f for f in candidates if f in forms or f.replace("’", "'") in forms), None)
    if form is None:
        return False
    rules = restrictions.get(form, {}).get("rules", [])
    if "case_sensitive" in rules and not word[:1].isupper():
        return False
    if "not_sentence_initial" in rules and first:
        return False
    if "require_pos" in rules and (classes is None or not classes & {"PRON", "DET"}):
        return False
    return True


def count(text, record, which):
    forms = set(record[which])
    restrictions = record.get("restrictions", {})
    use_tagger = which == "informal" and record["signals"]["tagger"]["enabled"]
    lookup = lookup_forms(record["lang"]) if which == "informal" and record["signals"]["lookup"]["enabled"] else ()
    hits = 0
    for word, first, classes, verb in tokens(strip_quoted_spans(text), record):
        hits += (matches(word, first, classes, forms, restrictions) or (use_tagger and verb)
                 or unicodedata.normalize("NFC", word.lower()) in lookup)
    return hits


def informal_present(text, record):
    return count(text, record, "informal")


def formal_present(text, record):
    return count(text, record, "formal")


def is_informal(text, record, min_forms=MIN_INFORMAL_FORMS):
    return informal_present(text, record) >= min_forms and formal_present(text, record) == 0


def has_no_informal(text, record, need_address=False):
    """need_address: the text must address its reader at all (a formal form), so that avoiding address is no pass."""
    return informal_present(text, record) == 0 and (not need_address or formal_present(text, record) > 0)


def sentence_levels(text, record):
    """For each sentence, the built politeness levels whose ending closes it (full combinations and single tags)."""
    shape = (lambda s: unicodedata.normalize("NFD", s)) if record.get("decompose") else (lambda s: s)
    built = {name: [shape(e) for e in level["verified"]] for name, level in record["levels"].items() if level["built"]}
    out = []
    # a point between two digits is not a sentence end (3.5, 12.50)
    for sentence in SENTENCE.findall(re.sub(r"(?<=\d)\.(?=\d)", "", strip_quoted_spans(text))):
        core = re.sub(r"[\s\W]+$", "", unicodedata.normalize("NFC", sentence))
        if not any(c.isalpha() for c in core):
            continue
        out.append({name for name, endings in built.items() if any(shape(core).endswith(e) for e in endings)})
    return out


def ends_in_level(text, record, level, share=LEVEL_SHARE):
    """Enough sentences end in the level. For a full combination (FORM+POL) none may end in another combination;
    a single-tag level (any:POL) has no rival: a sentence can carry several tags."""
    sentences = sentence_levels(text, record)
    if not sentences:
        return False
    rivals = set() if level.startswith("any:") else {
        name for name, facts_ in record["levels"].items()
        if facts_["built"] and name != level and not name.startswith("any:")}
    in_level = sum(level in s for s in sentences)
    return in_level / len(sentences) >= share and not any(s & rivals for s in sentences)
