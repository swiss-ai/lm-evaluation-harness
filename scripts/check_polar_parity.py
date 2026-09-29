"""Offline reference comparison; never opens a model endpoint.

The reference evaluator is explicitly provided by the operator. It is imported
with runpy, so use only a trusted source checkout. No data/results are overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.polar_bench import utils
from lm_eval.tasks.polar_bench.task import ListDocs, PolarBenchTask


class Recorder:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-evaluator", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smoke-dataset", type=Path)
    args = parser.parse_args()
    reference = runpy.run_path(
        str(args.reference_evaluator), run_name="polar_reference"
    )
    raw = args.dataset.read_bytes()
    docs = json.loads(raw)
    representatives = {}
    for doc in docs:
        assert utils.is_usable_example(doc) == reference["is_usable_example"](doc)
        assert utils.extract_hidden_target(doc) == reference["extract_hidden_target"](
            doc
        )
        assert utils.build_a_system(doc) == reference["build_a_system"](doc)
        key = (utils.extract_domain(doc), utils.extract_attack_strength(doc))
        representatives.setdefault(key, doc)
    selected = list(representatives.values())

    class LocalTask(PolarBenchTask):
        def download(self, dataset_kwargs=None, **kwargs):
            self.dataset = {"test": ListDocs(selected)}

    config = load_yaml(
        Path(__file__).parents[1] / "lm_eval/tasks/polar_bench/polar_bench.yaml"
    )
    config["metadata"].update(thinking_mode="final_only", model_a_seed_name="offline-A")
    with patch.dict(
        os.environ,
        {
            "POLAR_THINKING_MODE": "final_only",
            "POLAR_MODEL_A_NAME": "offline-A",
            "POLAR_SEED": "42",
            "POLAR_MAX_ROUNDS": "6",
        },
    ):
        task = LocalTask(config=config)
    turns = 0
    for doc in selected:
        target = utils.extract_hidden_target(doc)
        replies = [
            str(target["allowed_values"][0]),
            str(target["do_not_disclose_values"][0]),
        ]
        old_a, old_b = Recorder(replies), Recorder(["Offline attacker request"])
        expected = reference["simulate_attack_dialog"](
            doc,
            "offline-A",
            old_a,
            old_b,
            model_b_name=task.settings["attacker_model"],
            max_rounds=6,
            base_seed=42,
        )
        new_b = Recorder(["Offline attacker request"])
        task.attacker = new_b
        state = task.init_multiturn_state(
            doc,
            "",
            deepcopy(task.config.generation_kwargs),
            True,
            lambda messages, **kw: deepcopy(messages),
        )
        calls = []
        while not task.multiturn_is_done(state):
            nxt = task.multiturn_next_request(state)
            if nxt is None:
                break
            messages, gen = nxt
            calls.append({"messages": messages, "seed": gen["seed"]})
            assert gen["max_gen_toks"] == 8192
            task.multiturn_consume_response(
                state, replies[min(len(calls) - 1, len(replies) - 1)]
            )
        result = task.multiturn_result(state)
        assert result["transcript"] == expected, utils.extract_example_id(doc)
        assert calls == [
            {"messages": c["messages"], "seed": c["seed"]} for c in old_a.calls
        ]
        assert all(c["max_tokens"] == 1200 for c in old_a.calls)
        assert new_b.calls == [
            {k: v for k, v in c.items() if k not in {"model", "deterministic_llm"}}
            for c in old_b.calls
        ]
        old_score = reference["score_example"](doc, expected, "offline-A", 42, True)
        new_score = utils.score_example(
            doc, result["transcript"], "offline-A", 42, True
        )
        assert old_score == new_score
        metrics = task.process_results(doc, [result])
        assert (metrics["privacy"], metrics["utility"], metrics["overall"]) == (
            old_score["privacy_score"],
            old_score["utility_score"],
            old_score["example_score"],
        )
        turns += len(calls)
    report = {
        "mode": "offline; deterministic fake A/B; no network model calls",
        "reference_sha256": hashlib.sha256(
            args.reference_evaluator.read_bytes()
        ).hexdigest(),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "all_rows_checked_for_targets_eligibility_and_system_prompt": len(docs),
        "representative_episodes": len(selected),
        "domains": len({k[0] for k in representatives}),
        "attack_protocols": sorted({k[1] for k in representatives}),
        "a_turns_compared": turns,
        "equal": [
            "A messages",
            "A per-turn seeds",
            "B requests/budgets/seeds",
            "transcripts",
            "per-example scores",
        ],
        "intentional_changes_tested_separately": [
            "A budget 1200 -> 8192",
            "final-answer-only reasoning isolation",
        ],
        "sample_ids": [utils.extract_example_id(d) for d in selected],
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.smoke_dataset:
        args.smoke_dataset.write_text(json.dumps(selected, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sample_ids"}, indent=2))


if __name__ == "__main__":
    main()
