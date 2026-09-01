from pathlib import Path

from lm_eval.tasks._yaml_loader import load_yaml


TASKS_ROOT = Path(__file__).parents[1] / "lm_eval" / "tasks"


def test_blend_country_processors_resolve_from_templates():
    root = TASKS_ROOT / "blend"
    task_configs = sorted(root.glob("*/blend_*.yaml"))

    assert len(task_configs) == 32
    for task_config in task_configs:
        config = load_yaml(task_config)
        assert callable(config["process_docs"]), task_config
        assert "!function ../" not in task_config.read_text()


def test_switzerland_qa_category_processors_resolve_from_templates():
    root = TASKS_ROOT / "switzerland_qa"
    task_configs = sorted(root.glob("*/[a-z][a-z]/switzerland_qa_*.yaml"))

    assert len(task_configs) == 75
    for task_config in task_configs:
        config = load_yaml(task_config)
        assert callable(config["process_docs"]), task_config
        assert "!function ../" not in task_config.read_text()
