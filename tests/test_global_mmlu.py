from pathlib import Path

from lm_eval.tasks._yaml_loader import load_yaml


GEN_0SHOT_ROOT = (
    Path(__file__).parents[1] / "lm_eval" / "tasks" / "global_mmlu" / "gen_0shot"
)


def test_gen_0shot_category_processors_resolve_from_templates():
    task_configs = sorted(GEN_0SHOT_ROOT.glob("[a-z][a-z]/global_mmlu_*.yaml"))

    assert len(task_configs) == 90
    for task_config in task_configs:
        config = load_yaml(task_config)
        assert callable(config["process_docs"]), task_config
