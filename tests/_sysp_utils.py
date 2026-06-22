"""Test helpers for the system-prompt-adherence tasks.

The new tasks live under ``lm_eval/tasks/realguardrails/`` and their
``utils.py`` modules import from ``lm_eval.tasks.ifeval``. Importing
``lm_eval.tasks`` triggers ``lm_eval/tasks/__init__.py``, which pulls
``lm_eval.api.task`` → ``lm_eval.api.metrics`` → ``sacrebleu`` and the
full backend dependency chain. The repo's other tests (e.g.
``test_private_dataset_overrides``) accept that cost because they
actually exercise ``ConfigurableTask``; the rule-based scoring tests do
not need any of it.

To keep these tests runnable under a lightweight env (and to keep their
runtime in milliseconds), we load each task's ``utils.py`` directly from
its file path. This helper centralizes the boilerplate so the pattern
isn't duplicated across test modules.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_task_utils(rel_path: str, *, module_name: str | None = None) -> ModuleType:
    """Load a task ``utils.py`` by its repo-relative path.

    Example::

        utils = load_task_utils(
            "lm_eval/tasks/realguardrails/tensortrust/utils.py"
        )

    Args:
        rel_path: path relative to the repo root.
        module_name: optional name to register under; defaults to the path
            with separators replaced by underscores.

    Returns:
        The freshly loaded module.
    """
    path = _REPO_ROOT / rel_path
    name = module_name or rel_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    """Minimal ``datasets.Dataset``-like stub exposing only ``.filter()``.

    Tasks under test call ``dataset.filter(predicate)`` in their
    ``process_docs`` hook. The full ``datasets`` library is heavyweight to
    spin up just to verify a one-line predicate — this stub lets the unit
    tests stay in pure Python.
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def filter(self, fn: Callable[[dict[str, Any]], bool]) -> FakeDataset:
        return FakeDataset([r for r in self.rows if fn(r)])
