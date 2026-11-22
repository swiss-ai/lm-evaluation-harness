"""Tests for ifeval_multilingual: kwargs decoding and scoring of IFEval, rewritten and native instructions."""

import json

import pytest


# IFEval's checkers need the `ifeval` extra, which the CPU test job doesn't install.
pytest.importorskip("langdetect")
pytest.importorskip("immutabledict")
pytest.importorskip("nltk")

from lm_eval.tasks.ifeval_multilingual import utils
from lm_eval.tasks.ifeval_multilingual.grader import scoring


IFEVAL_FIELDS = ["num_words", "relation", "forbidden_words"]
DATE = r"(?<![0-9])(?:3[01]|[12][0-9]|0[1-9])/(?:1[0-2]|0[1-9])/[0-9]{4}(?![0-9])"


def _kwargs(native=None, **ifeval):
    row = dict.fromkeys(IFEVAL_FIELDS)
    row.update(ifeval)
    row["native"] = json.dumps(native) if native is not None else None
    return row


def _doc(code, ids, kwargs, prompt="…"):
    return {
        "key": 0,
        "id": f"native:{code}:0",
        "prompt": prompt,
        "instruction_id_list": ids,
        "kwargs": kwargs,
        "language": "",
    }


def test_row_kwargs_decodes_native_and_drops_nulls():
    native = {"what": "dates", "required": [DATE], "forbidden": [], "min_count": 2}
    doc = _doc(
        "fr",
        ["native:dates", "length_constraints:number_words"],
        [_kwargs(native), _kwargs(num_words=10, relation="at least")],
    )
    assert utils.row_kwargs(doc) == [native, {"num_words": 10, "relation": "at least"}]


def test_native_pattern_instruction():
    native = {
        "what": "dates",
        "example": "25/11/2026",
        "required": [DATE],
        "forbidden": [],
        "min_count": 2,
    }
    doc = _doc("fr", ["native:dates"], [_kwargs(native)])
    passed = utils.process_results(doc, ["Le 25/11/2026 puis le 02/12/2026."])
    failed = utils.process_results(doc, ["Le 25/11/2026 seulement."])
    assert passed["prompt_level_strict_acc"] and passed["inst_level_strict_acc"] == [
        True
    ]
    assert not failed["prompt_level_strict_acc"] and failed["inst_level_loose_acc"] == [
        False
    ]


def test_forbidden_words_in_another_script():
    # IFEval's \b word match; the multilingual checker matches whole words in any script
    doc = _doc(
        "el", ["keywords:forbidden_words"], [_kwargs(forbidden_words=["θάλασσα"])]
    )
    assert not utils.process_results(doc, ["Η θάλασσα είναι ήρεμη."])[
        "prompt_level_strict_acc"
    ]
    assert utils.process_results(doc, ["Ο ουρανός είναι καθαρός."])[
        "prompt_level_strict_acc"
    ]


def test_every_profile_entry_has_a_checker():
    entries = scoring._entries()
    assert set(entries) == {
        "ar",
        "de",
        "el",
        "es",
        "fr",
        "hi",
        "ja",
        "ko",
        "pl",
        "sr",
        "zh",
    }
    for code, by_id in entries.items():
        for key, entry in by_id.items():
            assert hasattr(scoring.checkers, entry["checker"]), (code, key)
