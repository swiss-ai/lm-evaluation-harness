import pytest

from lm_eval.api.registry import get_filter


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        ([["A was tempting. The answer is (C). B is incorrect."]], [["C"]]),
        ([["The answer is (A). After checking, The answer is (D)."]], [["D"]]),
        ([["(B)", "no selection"], ["C"]], [["B", "[invalid]"], ["C"]]),
        ([[], [""]], [[], ["[invalid]"]]),
    ],
)
def test_ordered_answer_extraction_preserves_priority_and_response_shape(
    responses, expected
):
    extract = get_filter("ordered_regex")(
        regex_patterns=[r"The answer is \(([A-D])\)", r"\(([A-D])\)", r"([A-D])"],
        group_select=-1,
    )
    assert extract.apply(responses, [{}] * len(responses)) == expected


def test_ordered_regex_cleans_capture_and_uses_configured_fallback():
    extract = get_filter("ordered_regex")(
        regex_patterns=[r"USD ([0-9,]+)", r"EUR ([0-9,]+)"],
        strip_extracts=[","],
        fallback="missing",
    )
    assert extract.apply([["USD 1,234", "EUR 42", "unknown"]], [{}]) == [
        ["1234", "42", "missing"]
    ]


def test_ordered_regex_alternative_capture_groups():
    extract = get_filter("ordered_regex")(regex_patterns=[r"answer: (A)|(B)"])
    assert extract.apply([["answer: A", "B"]], [{}]) == [["A", "B"]]
