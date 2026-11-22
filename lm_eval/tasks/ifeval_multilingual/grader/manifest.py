"""
IFEval manifest: how each instruction is handled in each language (manifest.yaml).

entry(lang, instruction_id):  the instruction's action in the language, after group and language overrides
translated_keys(lang, ...):   the string kwargs to translate with the prompt ([] unless translate_kwarg)
check(manifest):              consistency of manifest.yaml; raises ValueError on the first problem
"""

import os
from functools import lru_cache

import yaml

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.yaml")
ACTIONS = {"keep", "translate_kwarg", "rewrite_checker", "drop"}


@lru_cache(maxsize=None)
def load(path=PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def entry(lang, instruction_id, manifest=None):
    """
    Action of an instruction in a language: defaults, then the overrides of every group
    containing the language, then the language's own overrides.

    Example:
        entry("de", "change_case:english_capital") -> {"action": "rewrite_checker", "checker": "uppercase_any_language"}
        entry("zh", "change_case:english_capital") -> {"action": "drop", "checker": "uppercase_any_language"}
    """
    manifest = manifest or load()
    if instruction_id not in manifest["instructions"]:
        raise KeyError(f"{instruction_id}: not in the manifest")
    result = dict(manifest["instructions"][instruction_id])
    for group, langs in manifest["groups"].items():
        if lang in langs:
            result.update(manifest["overrides"].get(group, {}).get(instruction_id, {}))
    result.update((manifest.get("languages") or {}).get(lang, {}).get(instruction_id, {}))
    return result


def translated_keys(lang, instruction_id, manifest=None):
    result = entry(lang, instruction_id, manifest)
    return result.get("translate", []) if result["action"] == "translate_kwarg" else []


def check(manifest=None):
    manifest = manifest or load()
    instructions = manifest["instructions"]
    overrides = [(f"group {g}", o) for g, o in manifest["overrides"].items()]
    overrides += [(f"language {l}", o) for l, o in (manifest.get("languages") or {}).items()]
    for group in manifest["overrides"]:
        if group not in manifest["groups"]:
            raise ValueError(f"overrides for unknown group {group!r}")
    for where, entries in [("instructions", instructions)] + overrides:
        for instruction_id, e in entries.items():
            if instruction_id not in instructions:
                raise ValueError(f"{where}: unknown instruction {instruction_id!r}")
            if "action" in e and e["action"] not in ACTIONS:
                raise ValueError(f"{where}: {instruction_id}: unknown action {e['action']!r}")
            if e.get("action") == "translate_kwarg" and not e.get("translate", instructions[instruction_id].get("translate")):
                raise ValueError(f"{where}: {instruction_id}: translate_kwarg without kwargs to translate")
            if e.get("action") == "rewrite_checker" and not e.get("checker"):
                raise ValueError(f"{where}: {instruction_id}: rewrite_checker without checker")
