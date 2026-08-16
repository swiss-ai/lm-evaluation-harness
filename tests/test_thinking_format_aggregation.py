"""Tests for in-repo aggregation of per-response generation info into ResultAcc
(lm_eval/evaluator_utils.py:promote_generation_info_metrics), which routes the
thinking-format flags (always) and response/thinking length (opt-in) through the
standard mean -> group aggregation -> logger path.
"""

import collections
from types import SimpleNamespace

import pytest

from lm_eval.api.group import ConfigurableGroup
from lm_eval.api.metrics import mean
from lm_eval.api.task import Task
from lm_eval.evaluator_utils import (
    _compute_task_aggregations,
    promote_generation_info_metrics,
)


_doc_counter = iter(range(10_000))


def _inst(length_info, repeats=None, doc_id=None):
    # distinct default doc_id so unrelated fake instances are not grouped together
    if doc_id is None:
        doc_id = next(_doc_counter)
    return SimpleNamespace(length_info=length_info, repeats=repeats, doc_id=doc_id)


def _result_acc(instances, output_type="generate_until"):
    task = SimpleNamespace(
        instances=instances,
        OUTPUT_TYPE=output_type,
        task_name="test_task",
        aggregation=dict,
    )
    return {
        "task": task,
        "raw_metrics": collections.defaultdict(list),
        "logged_samples": [],
    }


def test_promotes_format_flags_to_sample_metrics():
    acc = _result_acc(
        [
            _inst(
                [
                    {
                        "thinking_format_has_open": 1,
                        "thinking_format_has_close": 1,
                        "thinking_format_correct": 1,
                        "response_length_chars": 42,  # not a format key
                    }
                ]
            ),
            _inst(
                [
                    {
                        "thinking_format_has_open": 1,
                        "thinking_format_has_close": 0,
                        "thinking_format_correct": 0,
                        "response_length_chars": 10,
                    }
                ]
            ),
        ]
    )
    promote_generation_info_metrics(acc)
    assert acc["raw_metrics"][("thinking_format_correct", "none")] == [1.0, 0.0]
    assert acc["raw_metrics"][("thinking_format_has_open", "none")] == [1.0, 1.0]
    assert acc["raw_metrics"][("thinking_format_has_close", "none")] == [1.0, 0.0]
    # length keys are NOT promoted unless include_length=True
    assert ("response_length_chars", "none") not in acc["raw_metrics"]
    # the per-task mean is the format-correctness rate
    assert mean(acc["raw_metrics"][("thinking_format_correct", "none")]) == 0.5


def test_length_metrics_promoted_only_with_include_length():
    docs = [
        _inst(
            [
                {
                    "response_length_chars": 40,
                    "thinking_length_chars": 10,
                    "thinking_format_correct": 1,
                }
            ]
        ),
        _inst(
            [
                {
                    "response_length_chars": 20,
                    # this doc did not close its thinking block -> no thinking_length
                    "thinking_format_correct": 0,
                }
            ]
        ),
    ]
    # default: length NOT promoted
    off = _result_acc(docs)
    promote_generation_info_metrics(off, include_length=False)
    assert ("response_length_chars", "none") not in off["raw_metrics"]
    # opt-in: response_length over ALL responses; thinking_length only over the
    # responses that carry it (the well-formed ones; the malformed doc is absent, NOT
    # zero-filled).
    on = _result_acc(docs)
    promote_generation_info_metrics(on, include_length=True)
    assert on["raw_metrics"][("response_length_chars", "none")] == [40.0, 20.0]
    assert on["raw_metrics"][("thinking_length_chars", "none")] == [10.0]
    # thinking-format flags still promoted alongside, over all responses
    assert on["raw_metrics"][("thinking_format_correct", "none")] == [1.0, 0.0]


def test_multi_unit_promoted_independently():
    # Each unit (chars, tokens) for both response and thinking length is promoted as
    # its own independent key.
    acc = _result_acc(
        [
            _inst(
                [
                    {
                        "response_length_chars": 100,
                        "thinking_length_chars": 90,
                        "response_length_tokens": 30,
                        "thinking_length_tokens": 25,
                    }
                ]
            )
        ]
    )
    promote_generation_info_metrics(acc, include_length=True)
    assert acc["raw_metrics"][("response_length_chars", "none")] == [100.0]
    assert acc["raw_metrics"][("thinking_length_chars", "none")] == [90.0]
    assert acc["raw_metrics"][("response_length_tokens", "none")] == [30.0]
    assert acc["raw_metrics"][("thinking_length_tokens", "none")] == [25.0]


def test_bool_excluded_on_length_path():
    # bool guard (int subclass) must also apply to length values.
    acc = _result_acc(
        [_inst([{"response_length_chars": True, "thinking_length_chars": False}])]
    )
    promote_generation_info_metrics(acc, include_length=True)
    assert ("response_length_chars", "none") not in acc["raw_metrics"]
    assert ("thinking_length_chars", "none") not in acc["raw_metrics"]


def test_multiple_responses_aggregated_per_doc():
    # repeats > 1: one length_info dict per response -> ONE per-doc value (the mean),
    # so the appended count matches num docs (not num responses).
    acc = _result_acc(
        [_inst([{"thinking_format_correct": 1}, {"thinking_format_correct": 0}])]
    )
    promote_generation_info_metrics(acc)
    assert acc["raw_metrics"][("thinking_format_correct", "none")] == [0.5]


def test_padding_clones_capped_at_repeats():
    # data-parallel padding can leave extra length_info entries on the last doc;
    # only the first `repeats` are real and should count.
    acc = _result_acc(
        [
            _inst(
                [
                    {"thinking_format_correct": 1},
                    {"thinking_format_correct": 1},  # real (repeats=2)
                    {"thinking_format_correct": 0},  # padding clone -> ignored
                ],
                repeats=2,
            )
        ]
    )
    promote_generation_info_metrics(acc)
    assert acc["raw_metrics"][("thinking_format_correct", "none")] == [1.0]


def test_multi_turn_instances_grouped_per_doc():
    # multi_turn_generate produces one Instance per turn; all turns of a doc share
    # doc_id and must collapse to ONE per-doc value (not one per turn).
    acc = _result_acc(
        [
            _inst([{"thinking_format_correct": 1}], doc_id=0),  # turn 1
            _inst([{"thinking_format_correct": 0}], doc_id=0),  # turn 2 (same doc)
            _inst([{"thinking_format_correct": 1}], doc_id=1),  # other doc
        ],
        output_type="multi_turn_generate",
    )
    promote_generation_info_metrics(acc)
    # doc 0 -> mean(1,0)=0.5 ; doc 1 -> 1.0  => two values, not three
    assert sorted(acc["raw_metrics"][("thinking_format_correct", "none")]) == [
        0.5,
        1.0,
    ]


def test_multi_turn_length_only_well_formed_turns_count():
    # multi_turn_generate: each turn is a SEPARATE Instance sharing doc_id. Turn 1 is
    # well-formed (carries thinking_length); turn 2 is not (response only). The per-doc
    # response length averages BOTH turns; thinking length counts ONLY the well-formed
    # turn (denominator = correct turns, no zero-fill from the malformed one).
    acc = _result_acc(
        [
            _inst(
                [{"response_length_chars": 100, "thinking_length_chars": 80}], doc_id=0
            ),  # turn 1, well-formed
            _inst([{"response_length_chars": 40}], doc_id=0),  # turn 2, not well-formed
        ],
        output_type="multi_turn_generate",
    )
    promote_generation_info_metrics(acc, include_length=True)
    assert acc["raw_metrics"][("response_length_chars", "none")] == [
        70.0
    ]  # mean(100,40)
    assert acc["raw_metrics"][("thinking_length_chars", "none")] == [
        80.0
    ]  # turn 1 only


def test_no_length_info_yields_no_metrics():
    acc = _result_acc([_inst([]), _inst(None)])
    promote_generation_info_metrics(acc)
    assert len(acc["raw_metrics"]) == 0


def test_bool_values_excluded():
    # bool is an int subclass; flags are emitted as plain ints, so a stray bool
    # must not be promoted as 0/1.
    acc = _result_acc([_inst([{"thinking_format_correct": True}])])
    promote_generation_info_metrics(acc)
    assert ("thinking_format_correct", "none") not in acc["raw_metrics"]


def test_thinking_length_denominator_excludes_incorrect():
    # Unbiased average: thinking_length is present only on well-formed responses, so
    # its mean's denominator is the count of CORRECT responses — never inflated by the
    # malformed one. response_length stays over all responses.
    docs = [
        _inst([{"response_length_chars": 100, "thinking_length_chars": 80}]),  # correct
        _inst([{"response_length_chars": 120, "thinking_length_chars": 90}]),  # correct
        _inst([{"response_length_chars": 10}]),  # malformed -> no thinking_length
    ]
    acc = _result_acc(docs)
    promote_generation_info_metrics(acc, include_length=True)
    assert acc["raw_metrics"][("response_length_chars", "none")] == [
        100.0,
        120.0,
        10.0,
    ]
    # thinking over the 2 correct only -> mean(80,90)=85, NOT divided by 3
    assert acc["raw_metrics"][("thinking_length_chars", "none")] == [80.0, 90.0]
    assert mean(acc["raw_metrics"][("thinking_length_chars", "none")]) == 85.0


def test_aggregate_thinking_may_exceed_aggregate_response():
    # Intended consequence of "response over all, thinking over correct-only": the
    # aggregate thinking mean can exceed the aggregate response mean (different bases),
    # even though per-sample thinking <= response always holds.
    docs = [
        _inst([{"response_length_chars": 100, "thinking_length_chars": 90}]),  # correct
        _inst([{"response_length_chars": 10}]),  # malformed -> no thinking_length
    ]
    acc = _result_acc(docs)
    promote_generation_info_metrics(acc, include_length=True)
    assert mean(acc["raw_metrics"][("response_length_chars", "none")]) == 55.0
    assert mean(acc["raw_metrics"][("thinking_length_chars", "none")]) == 90.0


def test_no_correct_response_omits_thinking_length():
    # When no response is well-formed, thinking_length has no values and the key is
    # simply absent (the gather unions keys, so this is multi-rank safe). response
    # length is still aggregated over all responses.
    docs = [
        _inst([{"response_length_chars": 30}]),
        _inst([{"response_length_chars": 40}]),
    ]
    acc = _result_acc(docs)
    promote_generation_info_metrics(acc, include_length=True)
    assert acc["raw_metrics"][("response_length_chars", "none")] == [30.0, 40.0]
    assert ("thinking_length_chars", "none") not in acc["raw_metrics"]


def test_orphan_thinking_unit_promoted_independently():
    # Each length key is now independent: a thinking_length_<unit> with no matching
    # response_length_<unit> (e.g. vLLM token_ids=None -> no response_length_tokens but
    # a tokenizer-derived thinking_length_tokens) is promoted on its own.
    acc = _result_acc(
        [
            _inst(
                [
                    {
                        "response_length_chars": 100,
                        "thinking_length_chars": 90,
                        "thinking_length_tokens": 50,  # no response_length_tokens
                    }
                ]
            )
        ]
    )
    promote_generation_info_metrics(acc, include_length=True)
    assert acc["raw_metrics"][("thinking_length_chars", "none")] == [90.0]
    assert acc["raw_metrics"][("thinking_length_tokens", "none")] == [50.0]
    assert ("response_length_tokens", "none") not in acc["raw_metrics"]


def test_sample_len_is_max_over_sparse_metrics():
    # A sparse promoted metric (e.g. thinking_length only on docs that reasoned)
    # must NOT shrink the reported per-task sample count: sample_len = max, not
    # last-wins. Inserted after the dense metric so a last-wins bug would show.
    task = SimpleNamespace(task_name="t", aggregation=dict)  # unknown metrics -> mean
    raw_metrics = {
        ("exact_match", "none"): [1.0, 0.0, 1.0],  # 3 docs (dense)
        ("thinking_length_chars", "none"): [10.0],  # sparse: 1 doc
    }
    _, sample_len = _compute_task_aggregations(task, raw_metrics, bootstrap_iters=0)
    assert sample_len == 3  # max(3, 1); a last-wins bug would give 1


class _StubTask(Task):
    """Minimal concrete Task used as a child of the refactored Group object."""

    def __init__(self, name):
        self._config = SimpleNamespace(task=name)

    def has_training_docs(self): ...
    def has_validation_docs(self): ...
    def has_test_docs(self): ...
    def doc_to_text(self, doc): ...
    def doc_to_target(self, doc): ...
    def construct_requests(self, doc, ctx, **kw): ...
    def process_results(self, doc, results): ...
    def aggregation(self): ...
    def higher_is_better(self): ...


def test_group_sample_len_independent_of_sparse_metrics():
    """A group's sample count is the sum over subtasks, not a sparse metric count."""
    results = {
        "taskA": {  # dense only
            "name": "taskA",
            "alias": "taskA",
            "exact_match,none": 0.5,
            "exact_match_stderr,none": 0.1,
            "sample_len": 100,
        },
        "taskB": {  # dense + sparse promoted metric
            "name": "taskB",
            "alias": "taskB",
            "exact_match,none": 0.3,
            "exact_match_stderr,none": 0.1,
            "thinking_format_correct,none": 0.9,
            "thinking_format_correct_stderr,none": 0.01,
            "sample_len": 50,
        },
    }
    group = ConfigurableGroup(
        config={
            "group": "mygroup",
            "task": ["taskA", "taskB"],
            "aggregate_metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "weight_by_size": True,
                    "filter_list": ["none"],
                }
            ],
        }
    )
    group.add(_StubTask("taskA"))
    group.add(_StubTask("taskB"))

    aggregated = group.aggregate(results)
    assert aggregated["sample_len"] == 150
    assert aggregated["exact_match,none"] == pytest.approx((0.5 * 100 + 0.3 * 50) / 150)
