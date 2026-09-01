from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from lm_eval.tasks._yaml_loader import load_yaml


if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
INDEX_FILENAME = "_task_index.json"
INDEX_SCHEMA_VERSION = 1


class Kind(Enum):
    TASK = auto()
    PY_TASK = auto()  # Python-defined, via "class"
    GROUP = auto()
    TAG = auto()
    UNKNOWN = auto()


@dataclass
class Entry:
    name: str
    kind: Kind
    yaml_path: Path | None  # None for generated / py-only entries
    cfg: dict[str, Any] | None = None
    tags: set[str] = field(default_factory=set)


class TaskIndex:
    """Build, persist, and load task discovery metadata."""

    def __init__(self, *, meta: dict[str, str] | None = None) -> None:
        pass

    @staticmethod
    def build(
        paths: Iterable[Path],
        *,
        resolve_includes=True,
    ) -> dict[str, Entry]:
        index: dict[str, Entry] = {}
        log.debug("Building task index from %s", paths)
        for root in paths:
            # Build this path's entries (skip duplicates within same path)
            path_index: dict[str, Entry] = {}
            for yaml_path in TaskIndex._iter_yaml_files(root):
                try:
                    cfg = load_yaml(
                        yaml_path,
                        resolve_func=False,
                        recursive=resolve_includes,
                    )
                    TaskIndex.process_cfg(cfg, yaml_path, path_index)
                except Exception as err:
                    log.debug("Skip %s (%s)", yaml_path, err)
                    continue

            # Merge: later paths overwrite earlier, with warning
            TaskIndex.merge(index, path_index)

        log.debug("Built task index with %d entries", len(index))
        return index

    @staticmethod
    def merge(index: dict[str, Entry], additions: dict[str, Entry]) -> None:
        """Merge entries in place, preserving include-path override semantics."""
        for name, entry in additions.items():
            if name in index:
                log.warning(
                    "Task '%s' from %s overrides existing task from %s",
                    name,
                    entry.yaml_path,
                    index[name].yaml_path,
                )
            index[name] = entry

    @staticmethod
    def read(path: Path, *, root: Path) -> dict[str, Entry]:
        """Load a persistent index without touching the indexed YAML files."""
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise TypeError("Task index root must be an object")
        if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported task index schema version "
                f"{payload.get('schema_version')!r}; expected {INDEX_SCHEMA_VERSION}"
            )
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            raise TypeError("Task index 'entries' must be an object")

        index: dict[str, Entry] = {}
        for name, raw_entry in raw_entries.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Task index entry names must be non-empty strings")
            if not isinstance(raw_entry, dict):
                raise TypeError(f"Task index entry {name!r} must be an object")
            try:
                kind = Kind[raw_entry["kind"]]
            except (KeyError, TypeError) as err:
                raise ValueError(
                    f"Task index entry {name!r} has an invalid kind"
                ) from err

            relative_path = raw_entry.get("yaml_path")
            yaml_path = None
            if relative_path is not None:
                if not isinstance(relative_path, str):
                    raise ValueError(
                        f"Task index entry {name!r} has a non-string YAML path"
                    )
                posix_path = PurePosixPath(relative_path)
                if posix_path.is_absolute() or ".." in posix_path.parts:
                    raise ValueError(
                        f"Task index entry {name!r} has an unsafe YAML path"
                    )
                yaml_path = root.joinpath(*posix_path.parts)

            tags = raw_entry.get("tags", [])
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                raise ValueError(f"Task index entry {name!r} has invalid tags")
            index[name] = Entry(
                name=name,
                kind=kind,
                yaml_path=yaml_path,
                cfg=None,
                tags=set(tags),
            )

        expected_count = payload.get("entry_count")
        if expected_count is not None and expected_count != len(index):
            raise ValueError(
                f"Task index entry count mismatch: {expected_count!r} != {len(index)}"
            )
        return index

    @staticmethod
    def write(index: dict[str, Entry], path: Path, *, root: Path) -> None:
        """Write discovery metadata with YAML paths relative to ``root``."""
        root = root.resolve()
        entries: dict[str, dict[str, Any]] = {}
        for name, entry in sorted(index.items()):
            relative_path = None
            if entry.yaml_path is not None:
                try:
                    relative_path = (
                        entry.yaml_path.resolve().relative_to(root).as_posix()
                    )
                except ValueError as err:
                    raise ValueError(
                        f"Cannot index YAML path outside task root: {entry.yaml_path}"
                    ) from err
            entries[name] = {
                "kind": entry.kind.name,
                "yaml_path": relative_path,
                "tags": sorted(entry.tags),
            }

        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "entry_count": len(entries),
            "entries": entries,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _iter_yaml_files(root: Path):
        # Sort for deterministic traversal order across filesystems
        yield from sorted(
            (
                p
                for p in root.glob("**/*.yaml")
                if not any(part in _IGNORE_DIRS for part in p.parts)
            ),
            key=lambda p: p.as_posix(),
        )

    @staticmethod
    def process_cfg(
        cfg: dict[str, Any],
        path: Path,
        index_: dict[str, Entry],
    ) -> None:
        kind = TaskIndex._kind_of(cfg)
        match kind:
            case Kind.GROUP:
                grp_name = cfg["group"]

                if grp_name in index_:
                    log.debug(
                        f"Duplicate group name '{grp_name}' found. "
                        f"Already registered from: {index_[grp_name].yaml_path}. "
                        f"Skipping duplicate from: {path}"
                    )
                    return
                index_[grp_name] = Entry(
                    name=grp_name,
                    kind=Kind.GROUP,
                    yaml_path=path,
                    tags=TaskIndex._str_to_set(cfg.get("tag")),
                    cfg=cfg,
                )

            case Kind.TASK | Kind.PY_TASK:
                name = cfg["task"]
                if name in index_:
                    log.warning(
                        f"Duplicate task name '{name}' found. "
                        f"Already registered from: {index_[name].yaml_path}. "
                        f"Skipping duplicate from: {path}"
                    )
                    return
                index_[name] = Entry(
                    name=name,
                    kind=kind,
                    yaml_path=path,
                    tags=TaskIndex._str_to_set(cfg.get("tag")),
                    cfg=cfg,
                )
                TaskIndex._register_tags(name, cfg.get("tag"), index_)
        return

    @staticmethod
    def _register_tags(
        task: str,
        tags: str | list[str] | None,
        index_: dict[str, Entry],
    ) -> None:
        if not tags:
            return
        for tag in tags if isinstance(tags, list) else [tags]:
            entry = index_.setdefault(
                tag,
                Entry(name=tag, kind=Kind.TAG, yaml_path=None, tags=set()),
            )
            entry.tags.add(task)

    @staticmethod
    def _kind_of(cfg: dict) -> Kind:
        match cfg:
            # Python task: has 'class' key (check before 'task' since PY_TASK may have both)
            case {"class": _}:
                return Kind.PY_TASK
            # Group configs have task: list[str | dict]
            case {"group": _}:
                return Kind.GROUP
            case {"task": _}:
                return Kind.TASK
            case _:
                raise ValueError(f"Unknown config shape: keys={list(cfg.keys())}")

    @staticmethod
    def entry_from_path(path: Path) -> Entry | None:
        """Create an Entry from a YAML file path (not in the index)."""
        path = path.expanduser().resolve()
        if not path.is_file():
            return None
        cfg = load_yaml(path, resolve_func=False)
        kind = TaskIndex._kind_of(cfg)
        name: str | None = cfg.get("task") or cfg.get("group")
        return Entry(name=name, kind=kind, yaml_path=path) if name else None

    @staticmethod
    def entry_from_config(cfg: dict[str, Any]) -> Entry | None:
        """Create an Entry from a raw config dict (not in the index)."""
        _kind = TaskIndex._kind_of(cfg)
        match _kind:
            case Kind.GROUP:
                if "group" in cfg:
                    return Entry(name=cfg["group"], kind=_kind, yaml_path=None, cfg=cfg)
            case Kind.TASK | Kind.PY_TASK:
                if "task" in cfg:
                    return Entry(name=cfg["task"], kind=_kind, yaml_path=None, cfg=cfg)
        return None

    @staticmethod
    def _str_to_set(*args) -> set[str]:
        """Convert a string or list of strings to a set of strings."""
        result = set()
        for t in args:
            if t is None:
                continue
            result.update([t] if isinstance(t, str) else t)
        return result
