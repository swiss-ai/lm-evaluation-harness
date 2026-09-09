from pathlib import Path

import pytest

from lm_eval.api.group import AggMetricConfig, Group
from lm_eval.tasks._yaml_loader import load_yaml
from tests.test_group import MockTask


def test_mmlu_flan_cot_groups_aggregate_generated_answer_metrics():
    config = load_yaml(
        Path(__file__).resolve().parents[1]
        / "lm_eval/tasks/mmlu/flan_cot_zeroshot/_mmlu.yaml",
        resolve_func=False,
    )
    # The tasks emit exact_match for three extraction filters, never acc.
    results = {
        "subject_a": {
            "sample_len": 10,
            "exact_match,strict-match": 0.2,
            "exact_match,flexible-extract": 0.5,
            "exact_match,ordered-extract": 0.6,
        },
        "subject_b": {
            "sample_len": 30,
            "exact_match,strict-match": 0.6,
            "exact_match,flexible-extract": 0.9,
            "exact_match,ordered-extract": 1.0,
        },
    }
    for group_config in [config, *config["task"]]:
        group = Group(
            name=group_config["group"],
            aggregate_metric_list=[
                AggMetricConfig(**item)
                for item in group_config["aggregate_metric_list"]
            ],
        )
        group.add(MockTask("subject_a"))
        group.add(MockTask("subject_b"))
        actual = group.aggregate(results)
        assert actual.get("exact_match,strict-match") == pytest.approx(0.5)
        assert actual.get("exact_match,flexible-extract") == pytest.approx(0.8)
        assert actual.get("exact_match,ordered-extract") == pytest.approx(0.9)
