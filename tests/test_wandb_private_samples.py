from unittest.mock import Mock

import pandas as pd

from lm_eval.loggers.wandb_logger import WandbLogger


def _make_logger(task_configs, task_names):
    logger = object.__new__(WandbLogger)
    logger.task_configs = task_configs
    logger.task_names = task_names
    logger.group_names = []
    logger.step_metrics = {"OptimizerStep": 1}
    logger.run = Mock()
    logger._generate_dataset = Mock(return_value=pd.DataFrame([{"id": 1}]))
    logger._log_samples_as_artifact = Mock()
    return logger


def test_log_eval_samples_skips_private_dataset_tasks():
    logger = _make_logger(
        task_configs={
            "private_task": {"dataset_path": "swiss-ai/switzerland_qa"},
            "public_task": {"dataset_path": "hellaswag"},
        },
        task_names=["private_task", "public_task"],
    )

    samples = {
        "private_task": [
            {
                "doc_id": 0,
                "target": "A",
                "arguments": [["q"]],
                "resps": [[["a"]]],
                "filtered_resps": [["a"]],
                "acc": 1.0,
            }
        ],
        "public_task": [
            {
                "doc_id": 1,
                "target": "B",
                "arguments": [["q"]],
                "resps": [[["b"]]],
                "filtered_resps": [["b"]],
                "acc": 1.0,
            }
        ],
    }

    logger.log_eval_samples(samples)

    assert logger._generate_dataset.call_count == 1
    assert logger._generate_dataset.call_args.args[0] == samples["public_task"]
    logger._log_samples_as_artifact.assert_called_once_with(
        samples["public_task"], "public_task"
    )
    assert logger.run.log.call_count == 1
    logged_payload = logger.run.log.call_args.kwargs
    assert logged_payload["commit"] is True
    assert "public_task_eval_results" in logger.run.log.call_args.args[0]
    assert "private_task_eval_results" not in logger.run.log.call_args.args[0]


def test_log_eval_samples_grouped_tasks_skip_private_and_keep_public():
    logger = _make_logger(
        task_configs={
            "private_group_task": {
                "dataset_path": "swiss-ai/include-base-new-45",
                "group": "my_group",
            },
            "public_group_task": {"dataset_path": "hellaswag", "group": "my_group"},
        },
        task_names=["private_group_task", "public_group_task"],
    )

    samples = {
        "private_group_task": [
            {
                "doc_id": 0,
                "target": "A",
                "arguments": [["q"]],
                "resps": [[["a"]]],
                "filtered_resps": [["a"]],
                "acc": 1.0,
            }
        ],
        "public_group_task": [
            {
                "doc_id": 1,
                "target": "B",
                "arguments": [["q"]],
                "resps": [[["b"]]],
                "filtered_resps": [["b"]],
                "acc": 1.0,
            }
        ],
    }

    logger.log_eval_samples(samples)

    assert logger._generate_dataset.call_count == 1
    assert logger._generate_dataset.call_args.args[0] == samples["public_group_task"]
    logger._log_samples_as_artifact.assert_called_once_with(
        samples["public_group_task"], "public_group_task"
    )
    assert logger.run.log.call_count == 1
    assert "my_group_eval_results" in logger.run.log.call_args.args[0]
