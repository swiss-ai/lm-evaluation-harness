"""Tests for in-repo aggregation of per-response generation info into sample_metrics
(lm_eval/evaluator_utils.py:promote_generation_info_metrics), which routes the
thinking-format flags (always) and response/thinking length (opt-in) through the
standard mean -> consolidate -> W&B path.
"""

import collections
from types import SimpleNamespace

from lm_eval.api.metrics import mean
from lm_eval.evaluator_utils import TaskOutput, promote_generation_info_metrics


_doc_counter = iter(range(10_000))


def _inst(length_info, repeats=None, doc_id=None):
    # distinct default doc_id so unrelated fake instances are not grouped together
    if doc_id is None:
        doc_id = next(_doc_counter)
    return SimpleNamespace(length_info=length_info, repeats=repeats, doc_id=doc_id)


def _task_output(instances):
    return SimpleNamespace(
        task=SimpleNamespace(instances=instances),
        sample_metrics=collections.defaultdict(list),
    )


def test_promotes_format_flags_to_sample_metrics():
    to = _task_output(
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
    promote_generation_info_metrics(to)
    assert to.sample_metrics[("thinking_format_correct", "none")] == [1.0, 0.0]
    assert to.sample_metrics[("thinking_format_has_open", "none")] == [1.0, 1.0]
    assert to.sample_metrics[("thinking_format_has_close", "none")] == [1.0, 0.0]
    # length keys are NOT promoted unless include_length=True
    assert ("response_length_chars", "none") not in to.sample_metrics
    # the per-task mean is the format-correctness rate
    assert mean(to.sample_metrics[("thinking_format_correct", "none")]) == 0.5


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
    off = _task_output(docs)
    promote_generation_info_metrics(off, include_length=False)
    assert ("response_length_chars", "none") not in off.sample_metrics
    # opt-in: length promoted as a per-task mean. thinking_length shares response
    # length's basis: the unit occurs on at least one response, so the doc without
    # it counts as 0 (not skipped) -> averaged over the same docs as response length.
    on = _task_output(docs)
    promote_generation_info_metrics(on, include_length=True)
    assert on.sample_metrics[("response_length_chars", "none")] == [40.0, 20.0]
    assert on.sample_metrics[("thinking_length_chars", "none")] == [10.0, 0.0]
    # thinking-format flags still promoted alongside
    assert on.sample_metrics[("thinking_format_correct", "none")] == [1.0, 0.0]


def test_multi_unit_pairing_independent():
    # Both unit pairs (chars and tokens) present on one response must be promoted
    # independently and paired correctly per unit (guards the suffix-keying).
    to = _task_output(
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
    promote_generation_info_metrics(to, include_length=True)
    assert to.sample_metrics[("response_length_chars", "none")] == [100.0]
    assert to.sample_metrics[("thinking_length_chars", "none")] == [90.0]
    assert to.sample_metrics[("response_length_tokens", "none")] == [30.0]
    assert to.sample_metrics[("thinking_length_tokens", "none")] == [25.0]


def test_bool_excluded_on_length_path():
    # bool guard must also apply to length values (the pairing reads rval/tval).
    to = _task_output(
        [_inst([{"response_length_chars": True, "thinking_length_chars": False}])]
    )
    promote_generation_info_metrics(to, include_length=True)
    assert ("response_length_chars", "none") not in to.sample_metrics
    assert ("thinking_length_chars", "none") not in to.sample_metrics


def test_multiple_responses_aggregated_per_doc():
    # repeats > 1: one length_info dict per response -> ONE per-doc value (the mean),
    # so the appended count matches num docs (not num responses).
    to = _task_output(
        [_inst([{"thinking_format_correct": 1}, {"thinking_format_correct": 0}])]
    )
    promote_generation_info_metrics(to)
    assert to.sample_metrics[("thinking_format_correct", "none")] == [0.5]


def test_padding_clones_capped_at_repeats():
    # data-parallel padding can leave extra length_info entries on the last doc;
    # only the first `repeats` are real and should count.
    to = _task_output(
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
    promote_generation_info_metrics(to)
    assert to.sample_metrics[("thinking_format_correct", "none")] == [1.0]


def test_multi_turn_instances_grouped_per_doc():
    # multi_turn_generate produces one Instance per turn; all turns of a doc share
    # doc_id and must collapse to ONE per-doc value (not one per turn).
    to = _task_output(
        [
            _inst([{"thinking_format_correct": 1}], doc_id=0),  # turn 1
            _inst([{"thinking_format_correct": 0}], doc_id=0),  # turn 2 (same doc)
            _inst([{"thinking_format_correct": 1}], doc_id=1),  # other doc
        ]
    )
    promote_generation_info_metrics(to)
    # doc 0 -> mean(1,0)=0.5 ; doc 1 -> 1.0  => two values, not three
    assert sorted(to.sample_metrics[("thinking_format_correct", "none")]) == [0.5, 1.0]


def test_no_length_info_yields_no_metrics():
    to = _task_output([_inst([]), _inst(None)])
    promote_generation_info_metrics(to)
    assert len(to.sample_metrics) == 0


def test_group_placeholder_task_none_is_noop():
    to = SimpleNamespace(task=None, sample_metrics=collections.defaultdict(list))
    promote_generation_info_metrics(to)  # must not raise
    assert len(to.sample_metrics) == 0


def test_bool_values_excluded():
    # bool is an int subclass; flags are emitted as plain ints, so a stray bool
    # must not be promoted as 0/1.
    to = _task_output([_inst([{"thinking_format_correct": True}])])
    promote_generation_info_metrics(to)
    assert ("thinking_format_correct", "none") not in to.sample_metrics


def test_thinking_length_shares_response_basis():
    # Mixed task: some responses close their thinking block, some don't. thinking
    # length must be averaged over the SAME responses as response length (missing
    # = 0), so it can never exceed response length. A skip-missing bug would give
    # mean(thinking) = 90 > mean(response) = 55.
    docs = [
        _inst([{"response_length_chars": 100, "thinking_length_chars": 90}]),
        _inst([{"response_length_chars": 10}]),  # no closed block -> thinking 0
    ]
    to = _task_output(docs)
    promote_generation_info_metrics(to, include_length=True)
    assert to.sample_metrics[("response_length_chars", "none")] == [100.0, 10.0]
    assert to.sample_metrics[("thinking_length_chars", "none")] == [90.0, 0.0]
    assert mean(to.sample_metrics[("thinking_length_chars", "none")]) <= mean(
        to.sample_metrics[("response_length_chars", "none")]
    )


def test_zero_fill_within_doc_partial_close():
    # repeats>1: within ONE doc, some responses close their thinking block and some
    # don't. The non-closing responses count as 0, so the per-doc thinking mean is
    # over all responses (mean(90, 0) = 45), same basis as response_length.
    to = _task_output(
        [
            _inst(
                [
                    {"response_length_chars": 100, "thinking_length_chars": 90},
                    {"response_length_chars": 100},
                ],
                repeats=2,
            )
        ]
    )
    promote_generation_info_metrics(to, include_length=True)
    assert to.sample_metrics[("response_length_chars", "none")] == [100.0]
    assert to.sample_metrics[("thinking_length_chars", "none")] == [45.0]


def test_no_thinking_anywhere_zero_thinking_length():
    # thinking_length_<unit> is paired per-unit with response_length_<unit>: when no
    # response closed a block it is emitted as 0 for every response that has the
    # response_length unit (same basis/count, and consistent across ranks so a
    # multi-GPU shard with no thinking can't desync the key set).
    docs = [
        _inst([{"response_length_chars": 30}]),
        _inst([{"response_length_chars": 40}]),
    ]
    to = _task_output(docs)
    promote_generation_info_metrics(to, include_length=True)
    assert to.sample_metrics[("response_length_chars", "none")] == [30.0, 40.0]
    assert to.sample_metrics[("thinking_length_chars", "none")] == [0.0, 0.0]


def test_length_less_response_not_counted():
    # A response dict carrying no response_length_* (e.g. an empty {} from a
    # suppressed error, or a format-only dict) must contribute to NEITHER response
    # nor thinking length, so the two stay on an identical count/basis.
    docs = [
        _inst([{"response_length_chars": 100, "thinking_length_chars": 90}]),
        _inst([{"thinking_format_correct": 0}]),  # format-only, no response_length
        _inst([{}]),  # empty (suppressed-error) dict
    ]
    to = _task_output(docs)
    promote_generation_info_metrics(to, include_length=True)
    rc = to.sample_metrics[("response_length_chars", "none")]
    tc = to.sample_metrics[("thinking_length_chars", "none")]
    assert rc == [100.0]  # only the doc that had response_length
    assert tc == [90.0]  # same single basis, NOT [90.0, 0.0, 0.0]
    assert len(rc) == len(tc)


def test_orphan_thinking_unit_without_response_dropped():
    # thinking_length_<unit> with no matching response_length_<unit> (e.g. vLLM
    # token_ids=None -> no response_length_tokens, but tokenizer set -> a
    # thinking_length_tokens) is dropped, so it can't exceed / outnumber response.
    to = _task_output(
        [
            _inst(
                [
                    {
                        "response_length_chars": 100,
                        "thinking_length_chars": 90,
                        "thinking_length_tokens": 50,  # orphan: no response_length_tokens
                    }
                ]
            )
        ]
    )
    promote_generation_info_metrics(to, include_length=True)
    assert to.sample_metrics[("thinking_length_chars", "none")] == [90.0]
    assert ("thinking_length_tokens", "none") not in to.sample_metrics
    assert ("response_length_tokens", "none") not in to.sample_metrics


def test_sample_len_is_max_over_sparse_metrics():
    # A sparse promoted metric (e.g. thinking_length only on docs that reasoned)
    # must NOT shrink the reported per-task sample count: sample_len = max, not
    # last-wins. Inserted after the dense metric so a last-wins bug would show.
    task = SimpleNamespace(aggregation=lambda: {})  # unknown metrics -> mean
    to = TaskOutput(task=task, task_name="t")
    to.sample_metrics[("exact_match", "none")] = [1.0, 0.0, 1.0]  # 3 docs (dense)
    to.sample_metrics[("thinking_length_chars", "none")] = [10.0]  # sparse: 1 doc
    to.calculate_aggregate_metric(bootstrap_iters=0)
    assert to.sample_len == 3  # max(3, 1); a last-wins bug would give 1
