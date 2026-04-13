from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.api.task import ConfigurableTask


def _make_task(dataset_path: str, dataset_name: str = "English") -> ConfigurableTask:
    task = object.__new__(ConfigurableTask)
    task.DATASET_PATH = dataset_path
    task.DATASET_NAME = dataset_name
    task._config = SimpleNamespace(
        custom_dataset=None,
        metadata=None,
        dataset_kwargs=None,
        task="dummy_task",
    )
    return task


@pytest.mark.parametrize(
    ("dataset_path", "private_dir_name"),
    [
        ("swiss-ai/switzerland_qa", "swiss-ai__switzerland_qa"),
        ("swiss-ai/include-base-new-45", "swiss-ai__include-base-new-45"),
    ],
)
def test_private_dataset_uses_local_path_when_present(
    monkeypatch, tmp_path, dataset_path: str, private_dir_name: str
):
    local_root = tmp_path
    local_path = local_root / private_dir_name
    local_path.mkdir(parents=True)

    loader = Mock(return_value={"ok": True})
    monkeypatch.setenv("LM_EVAL_PRIVATE_DATASET_ROOT", str(local_root))
    monkeypatch.setattr("lm_eval.api.task.datasets.load_dataset", loader)

    task = _make_task(dataset_path=dataset_path)
    task.download({})

    assert task.dataset == {"ok": True}
    assert loader.call_count == 1
    assert loader.call_args.kwargs["path"] == str(local_path)
    assert loader.call_args.kwargs["name"] == "English"


def test_private_dataset_falls_back_to_hf_when_missing(monkeypatch, tmp_path):
    loader = Mock(return_value={"ok": True})
    monkeypatch.setenv("LM_EVAL_PRIVATE_DATASET_ROOT", str(tmp_path))
    monkeypatch.setattr("lm_eval.api.task.datasets.load_dataset", loader)

    task = _make_task(dataset_path="swiss-ai/switzerland_qa")
    task.download({})

    assert task.dataset == {"ok": True}
    assert loader.call_count == 1
    assert loader.call_args.kwargs["path"] == "swiss-ai/switzerland_qa"


def test_private_dataset_falls_back_to_hf_when_local_load_fails(monkeypatch, tmp_path):
    local_path = tmp_path / "swiss-ai__switzerland_qa"
    local_path.mkdir(parents=True)

    loader = Mock(side_effect=[RuntimeError("bad local dataset"), {"ok": True}])
    monkeypatch.setenv("LM_EVAL_PRIVATE_DATASET_ROOT", str(tmp_path))
    monkeypatch.setattr("lm_eval.api.task.datasets.load_dataset", loader)

    task = _make_task(dataset_path="swiss-ai/switzerland_qa")
    task.download({})

    assert task.dataset == {"ok": True}
    assert loader.call_count == 2
    assert loader.call_args_list[0].kwargs["path"] == str(local_path)
    assert loader.call_args_list[1].kwargs["path"] == "swiss-ai/switzerland_qa"


def test_private_dataset_loads_from_saved_dataset_config_dir(monkeypatch, tmp_path):
    local_path = tmp_path / "swiss-ai__switzerland_qa"
    config_path = local_path / "Romansh"
    config_path.mkdir(parents=True)
    (config_path / "dataset_dict.json").write_text("{}", encoding="utf-8")

    load_from_disk = Mock(return_value={"ok": "from-disk"})
    load_dataset = Mock()
    monkeypatch.setenv("LM_EVAL_PRIVATE_DATASET_ROOT", str(tmp_path))
    monkeypatch.setattr("lm_eval.api.task.datasets.load_from_disk", load_from_disk)
    monkeypatch.setattr("lm_eval.api.task.datasets.load_dataset", load_dataset)

    task = _make_task(dataset_path="swiss-ai/switzerland_qa", dataset_name="Romansh")
    task.download({})

    assert task.dataset == {"ok": "from-disk"}
    load_from_disk.assert_called_once_with(str(config_path))
    load_dataset.assert_not_called()
