"""
Language-specific instruction types generated programmatically: the rule comes from CLDR, chosen by code with no
per-language decision. The LLM only writes the sentence. Case study of types that can be automated at scale.

Each type has:
    checker:         checker class (checkers.py)
    kwargs:          checker settings filled by code, name -> type
    sampled:         numbers drawn per item, name -> [min, max]; never set by an LLM. An instance of the type may
                     bring its own ranges (instance["sampled"]: the count of a grammatical form)
    hidden_sampled:  sampled numbers the instruction does not state (optional)
    asks / facts / checks:  English fact sheet for the sentence prompt. <name> is filled from CLDR by code; {name} stays
                     in the sentence and is replaced by the number drawn for each item
    task_types:      task types (tasks.py) the instruction fits; absent: any (optional; copied into the language profile)
    validation:      "llm_answers": an LLM writes following answers from the sentence, code breaks them
                     "code_only": code writes the test texts (for rules LLMs cannot follow reliably, e.g. exact counts)
    conflicts:       instructions it cannot be combined with on one prompt
"""

from . import checkers

# id of a language-specific instruction in a dataset row: "native:<type>"
NATIVE_PREFIX = "native:"
# IFEval instructions no language-specific instruction can share a prompt with: the response is constrained already
INCOMPATIBLE_WITH_ANY = {"language:response_language", "detectable_format:constrained_response"}
# rules on the whole text: any string the response must contain (keyword, end phrase, repeated prompt) would break them
WHOLE_TEXT_RULES = {"script:target_script", "script:script_variant", "letters:diacritics", "han:forbidden_radical", "han:max_strokes"}

DATA_TYPES = {
    "punctuation:native_quotes": {
        "checker": checkers.NativeQuotesChecker,
        "kwargs": {"open": str, "close": str},
        "sampled": {},
        # "quotes at least one person" was written as "cite au moins 1 personne", which also reads "name a person" (job 3911885)
        # both halves in "asks" (the marks to use, the mark not to use): with the ban only in "checks", the writer left it
        # out and the judge rejected the request (51 candidates, job 3923684); kept requests stated it or not
        "asks": ("the response contains at least one quotation (the words someone said, reported word for word) in the "
                 "quotation marks <open> <close>, and no straight double quote \" anywhere in the response"),
        "facts": 'the quotation marks <open> <close> , written around the quoted words (<open>word<close>) ; not the straight double quote "',
        "checks": "the text contains <open> and <close>, and no straight double quote",
        "validation": "llm_answers",
        "conflicts": ["startend:quotation", "detectable_format:json_format"],
    },
    "punctuation:sentence_end": {
        "checker": checkers.SentenceEndChecker,
        "kwargs": {"mark": str, "foreign": str},
        "sampled": {"min_count": [2, 5]},
        "asks": "at least {min_count} <what>, each ending with <mark> instead of <foreign>",
        "facts": "the <what> mark is <mark> ; not <foreign>",
        "checks": "at least {min_count} <mark> in the text, and <foreign> never ends a sentence",
        "validation": "llm_answers",
        # the marker P.S. ends in the foreign mark before a space: never passes (hi danda, history 261)
        "conflicts": ["detectable_format:json_format", "startend:end_checker", "detectable_content:postscript"],
    },
    "punctuation:question_exclamation": {
        # questions / exclamations written with the language's own marks (¿…? ¡…! ؟ ՞). The request asks for them in
        # plain words, as M-IFEval does; that they are punctuated the way the language writes them is checked, not
        # asked: it is part of writing the language
        "checker": checkers.NativeMarkChecker,
        "kwargs": {"mark": str, "latin": str, "role": str},
        "sampled": {"min_count": [1, 3]},
        "asks": "at least {min_count} <kind>",
        "facts": "nothing about punctuation is asked: the request does not name or show any punctuation mark",
        "checks": "at least {min_count} <kind> punctuated the way the language writes them (<how>)",
        "validation": "llm_answers",
        "conflicts": ["detectable_format:json_format", "language:response_language", "script:target_script"],
    },
    "numbers:large_numbers": {
        "checker": checkers.LocalFormatChecker,
        "kwargs": {"what": str, "example": str, "required": list[str], "forbidden": list[str]},
        # from 2: a request states "at least 1" without a digit ("write the large numbers...")
        "sampled": {"min_count": [2, 3]},
        # the format is described by what it looks like (<how>: "with a space between the groups of digits"): "the local
        # format" was written word for word in the requests ("format local", job 3911885)
        # the count is in "asks": given only in "checks", the writer left it out of 3 requests in 4 (jobs 3922884, 3922913)
        # so is the ban on the other form (72 candidates rejected by the judge for leaving it out, job 3923684)
        "asks": "at least {min_count} large numbers written <how>, and no large number written like <foreign>",
        "facts": "<local> ; not <foreign><note>",
        "checks": "at least {min_count} large numbers written this way, none written like <foreign>",
        "validation": "llm_answers",
        "conflicts": ["numbers:native_digits", "detectable_format:json_format"],
    },
    # The two types below state a format of the language as an explicit rule (where the symbol goes, which layout), as
    # IFEval states "wrap the response in double quotation marks": the request gives the rule, the model has to follow
    # it. CLDR gives the rule. One rule per type: the currency type does not test the decimal separator.
    "numbers:currency": {
        "checker": checkers.LocalFormatChecker,
        "kwargs": {"what": str, "example": str, "required": list[str], "forbidden": list[str]},
        "sampled": {"min_count": [2, 3]},
        "asks": "at least {min_count} amounts of money, each written <how>, and none with <symbol> before the number (not <foreign>)",
        "facts": "<local> ; not <foreign>",
        "checks": "at least {min_count} amounts written this way, none written like <foreign>",
        "validation": "llm_answers",
        "conflicts": ["detectable_format:json_format"],
        "task_types": ["a product description", "a comparison of two options", "a review", "an email", "a letter",
                       "a news article", "an announcement or invitation", "a plan or schedule"],
    },
    "dates:numeric_date": {
        # tests the layout (separator, digits per field, place of the year, month not above 12). Which of two small
        # numbers is the day cannot be read from the text: languages whose layout is the en-US one get no instruction
        "checker": checkers.LocalFormatChecker,
        "kwargs": {"what": str, "example": str, "required": list[str], "forbidden": list[str]},
        "sampled": {"min_count": [2, 3]},
        "asks": "at least {min_count} dates, each written <how>, and no date in numbers written in another layout",
        "facts": "<local> ; not <foreign>",
        "checks": "at least {min_count} dates in this layout, no date in numbers in another layout (such as <foreign>)",
        "validation": "llm_answers",
        "conflicts": ["detectable_format:json_format"],
        "task_types": ["a plan or schedule", "an announcement or invitation", "a summary of an event", "a news article",
                       "an email", "a letter", "a job or school application text"],
    },
    "numbers:native_digits": {
        "checker": checkers.NativeDigitsChecker,
        "kwargs": {"required": list[str], "forbidden": list[str]},
        "sampled": {"min_count": [2, 3]},
        "asks": "at least {min_count} numbers written in the language's own digits",
        "facts": "the digits are <digits> (for example <example>) ; not 0-9",
        "checks": "at least {min_count} numbers in those digits, and no digit 0-9 anywhere in the text",
        "validation": "llm_answers",
        "conflicts": ["detectable_format:multiple_sections",
                      "numbers:large_numbers"],
    },
    "script:script_variant": {
        "checker": checkers.ScriptVariantChecker,
        "kwargs": {"conversion": str},
        # evidence, not difficulty: a text of shared characters only proves nothing; 3 leaves a margin for the few
        # characters both standards write alike that the reverse conversion still changes (著). A drawn 5-15 failed
        # fully Traditional answers of 40 characters (8-11 distinctive ones, follow test 3950675)
        "sampled": {"min_count": [3, 3]},
        "hidden_sampled": ["min_count"],
        "asks": "the response is written entirely in <target> characters",
        "facts": "write in <target> characters such as <example> ; not <other> characters such as <foreign>",
        "checks": "the text has no <other> characters",
        "validation": "llm_answers",
        "conflicts": ["script:target_script"],
    },
    "script:target_script": {
        "checker": checkers.TargetScriptChecker,
        "kwargs": {"target": list[str], "avoid": list[str]},
        "sampled": {},
        "asks": "the response is written in <script> letters, with no <avoid> letters",
        "facts": "write the language in the <script> script ; not in the <avoid> script",
        "checks": "at least one letter of the <script> script, and no letter of the <avoid> script",
        "validation": "llm_answers",
        "conflicts": ["detectable_format:json_format", "language:response_language",
                      "script:script_variant", "punctuation:sentence_end", "letters:diacritics"],
    },
    "letters:diacritics": {
        "checker": checkers.DiacriticsChecker,
        "kwargs": {"marks": str},
        "sampled": {},
        "asks": "every word is written with its <label>",
        "facts": "write the <label> on every word",
        "checks": "every word contains at least one of these marks",
        "validation": "llm_answers",
        "conflicts": ["detectable_format:json_format", "script:target_script"],
    },
    "letters:native_letter_frequency": {
        "checker": checkers.NativeLetterFrequencyChecker,
        "kwargs": {"letter": str, "relation": str},   # relation is always "exactly"; the letter is drawn per prompt from the pool
        "sampled": {"frequency": [2, 5]},
        "asks": "the letter <letter> appears exactly {frequency} times in the response",
        "facts": "the letter is <letter>",
        "checks": "the text contains <letter> exactly {frequency} times, upper and lower case counted together",
        "validation": "code_only",
        "conflicts": ["detectable_format:json_format", "letters:forbidden_letters", "letters:diacritics"],
    },
    "grammar:form": {
        # one instance per form of grammar_facts.json (build_grammar_facts.py): a tense, mood, case… that English lacks
        # and the Stanza tagger recognises. Broken answer: the sentences with the form removed (perturb.grammar_form)
        "checker": checkers.GrammarFormChecker,
        "kwargs": {"feature": str, "hosts": list[str], "stanza": str, "name": str, "host_name": str},
        "sampled": {"min_count": [1, 3]},      # default; each form brings its own range (instance["sampled"])
        "asks": "at least {min_count} <host_name> in <name>",
        "facts": "the form is <name>, for example: <examples> ; it is counted on <host_name>",
        "checks": "a tagger finds at least {min_count} <host_name> in <name> in the text",
        "validation": "llm_answers",
        "conflicts": ["language:response_language", "script:target_script", "script:script_variant",
                      "detectable_format:json_format"],   # capitals: the tagger lowercases them (history 334)
    },
    # Chinese characters, from Unicode's Unihan table (han_facts.json): the radical and the stroke count of each
    # character. The radicals and the stroke limits offered come from how often they occur in the language's text
    "han:forbidden_radical": {
        "checker": checkers.ForbiddenRadicalChecker,
        "kwargs": {"radical": str, "number": int},
        "sampled": {},
        "asks": "the response contains no character whose radical is <radical>, such as <examples>",
        "facts": "characters with the radical <radical>: <examples> and the others of that radical ; none of them may appear",
        "checks": "no character that Unicode classifies under the radical <radical> (Kangxi radical <number>)",
        "validation": "llm_answers",
        "conflicts": ["language:response_language", "script:target_script"],
    },
    "han:radical_count": {
        "checker": checkers.RadicalCountChecker,
        "kwargs": {"radical": str, "number": int},
        "sampled": {"min_count": [3, 5]},
        "asks": "at least {min_count} characters in the response have the radical <radical>, as <examples> do",
        "facts": "characters with the radical <radical>: <examples> and the others of that radical",
        "checks": "at least {min_count} characters that Unicode classifies under the radical <radical> (Kangxi radical <number>); the same character may count several times",
        "validation": "llm_answers",
        "conflicts": ["language:response_language", "script:target_script"],
    },
    "han:max_strokes": {
        "checker": checkers.MaxStrokesChecker,
        "kwargs": {},
        "sampled": {"max_strokes": [11, 13]},       # default; the resolver gives the range of the language
        "asks": "every Chinese character in the response has at most {max_strokes} strokes",
        "facts": "a character with more than {max_strokes} strokes may not appear (for example <heavy>, with <heavy_strokes> strokes)",
        "checks": "the stroke count of every Chinese character, from Unicode's table, is at most {max_strokes}",
        "validation": "llm_answers",
        "conflicts": ["language:response_language", "script:target_script"],
    },
    "register:level": {
        # one instance per politeness level whose endings register_facts.json could build for the language (UniMorph
        # forms grouped by level, endings verified on treebank sentence ends): Korean, Japanese
        "checker": checkers.RegisterLevelChecker,
        "kwargs": {"lang": str, "level": str},
        "sampled": {},
        "asks": "every sentence of the response is in <name>: each sentence ends in one of the endings of that level",
        "facts": "endings of this level: <endings><others>",
        "checks": ("at least 9 sentences in 10 end in one of these endings, and none ends in an ending of another level; "
                   "sentences between quotation marks are not counted (a quoted span that holds a blank line is counted: an answer of several paragraphs in one pair of quotes is read in full)"),
        "validation": "llm_answers",
        "conflicts": ["language:response_language", "script:target_script", "detectable_format:json_format", "startend:end_checker",
                      "startend:quotation"],   # quoted text is left out of the count
    },
    "letters:forbidden_letters": {
        "checker": checkers.ForbiddenLettersChecker,
        "kwargs": {"letters": list[str]},
        "sampled": {},
        "asks": "the response uses none of these letters: <letters>",
        "facts": "forbidden letters: <letters>",
        "checks": "none of these letters appears in the text: <letters>",
        "validation": "llm_answers",
        "conflicts": ["letters:native_letter_frequency", "letters:diacritics"],
    },
}
