from __future__ import annotations


""" 
multi_if.py
#
# Multi-IF task for lm-eval-harness with:
#  - Iterative multi-turn conversation evaluation (3 turns default)
#  - Instruction-level strict + loose checking (mirrors upstream metrics.py logic)
#  - Prompt-level (all instructions satisfied) strict + loose accuracies
#  - Aggregated "overall" scores per (turn, language)
#  - Confidence intervals (bootstrap) per (turn, language) for strict & loose (if scipy/numpy available)
#  - Additional cross-turn diagnostics: overall_accuracy (strict instruction-level),
#    conversation_all_turn_success, turn1_to_turn3_relative_drop, per-turn strict accs, per-language strict accs.
#
# NOTE: Actual model generation loop must call `process_responses(doc, responses)`
#       where `responses` is a list of strings (one per turn in doc["turns"]).
#
# License: This file may incorporate logic conceptually derived from the
# upstream Multi-IF repository (Apache-2.0). Keep attribution as needed.

"""

# lm_eval/tasks/multi_if/multi_if.py
"""
Multi-IF evaluation task integration (lm-eval-harness 0.4.8 style).

Registers a factory function `build_multi_if` with the global TASK_REGISTRY
via @register_task("multi_if").

Key features:
  - Loads facebook/Multi-IF dataset.
  - Multi-turn (default 3) conversation handling; external loop must generate one
    response per turn in order.
  - STRICT + LOOSE instruction checking (loose = normalization variants).
  - Aggregates per-turn & per-language strict/loose prompt & instruction accuracies,
    plus overall diagnostics (overall_accuracy, conversation_all_turn_success, etc.).
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional

from datasets import load_dataset

from lm_eval.api.task import Task
from lm_eval.api.registry import register_task, TASK_REGISTRY

from lm_eval.tasks.multi_if.instruction_checks import INSTRUCTION_REGISTRY


# -------- Helpers for loose normalization --------
def _gen_variants(response: str):
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    variants = [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def check_instruction_strict(checker, resp: str, kw: Dict[str, Any]) -> bool:
    return bool(checker(resp, **kw))


def check_instruction_loose(checker, resp: str, kw: Dict[str, Any]) -> bool:
    for variant in _gen_variants(resp):
        if variant.strip() and checker(variant, **kw):
            return True
    return False


class TurnSpec:
    prompt: str
    instruction_ids: List[str]
    kwargs_list: List[Dict[str, Any]]


class MultiIF(Task):
    subset_size = None
    languages = None
    max_turns = 3
    dataset = None
    VERSION = 1
    DATASET_PATH = "facebook/Multi-IF"

    def __init__(
        self,
        subset_size: Optional[int] = None,
        languages: Optional[List[str]] = None,
        max_turns: int = 3,
    ):
        super().__init__()
        self.subset_size = subset_size
        self.languages = set(languages) if languages else None
        self.max_turns = max_turns
        self.dataset = None

    # --- Harness flags ---
    def has_training_docs(self): return False
    def has_validation_docs(self): return True
    def has_test_docs(self): return False
    def fewshot_examples(self): return []
    def should_decontaminate(self): return False

    # --- Data loading ---
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        ds = load_dataset(self.DATASET_PATH, split="train")
        if self.subset_size is not None:
            ds = ds.select(range(min(self.subset_size, len(ds))))
        self.dataset = ds
        return ds

    # --- Iterator over evaluation docs ---
    def validation_docs(self):
        for idx, row in enumerate(self.dataset):
            lang = row.get("language") or row.get("turn_1_language")
            if self.languages and lang not in self.languages:
                continue
            turns = []
            for t in (1, 2, 3):
                if t > self.max_turns:
                    break
                pkey = f"turn_{t}_prompt"
                if not row.get(pkey):
                    continue
                id_key = f"turn_{t}_instruction_id_list"
                kw_key = f"turn_{t}_kwargs"
                inst_ids = row[id_key]
                kwargs_list = row[kw_key]
                if isinstance(inst_ids, str):
                    inst_ids = json.loads(inst_ids)
                if isinstance(kwargs_list, str):
                    kwargs_list = json.loads(kwargs_list)
                turns.append({
                    "prompt": row[pkey],
                    "instruction_ids": inst_ids,
                    "kwargs_list": kwargs_list
                })
            if turns:
                yield {
                    "conversation_id": idx,
                    "language": lang,
                    "turns": turns
                }

    # --- Single-turn prompt (first turn only; real multi-turn handled externally) ---
    def doc_to_text(self, doc):
        return doc["turns"][0]["prompt"]
    
    def fewshot_context(self, *args, **kwargs):
        # We don't do few-shot; return empty prefix.
        return ""

    def construct_requests(self, doc, ctx):
        # We bypass harness request construction; multi-turn generation is custom.
        return []

    # --- Processing model outputs ---
    def process_responses(self, doc, responses: List[str]):
        """
        responses: list of model outputs, one per turn (same length as doc["turns"])
        """
        per_instruction = []
        per_turn_lang_bucket = []
        language = doc["language"]

        for t, (turn, resp) in enumerate(zip(doc["turns"], responses)):
            strict_list, loose_list = [], []
            for inst_id, kw in zip(turn["instruction_ids"], turn["kwargs_list"]):
                checker = INSTRUCTION_REGISTRY.get(inst_id)
                if checker is None:
                    strict_ok = loose_ok = False
                else:
                    strict_ok = check_instruction_strict(checker, resp, kw)
                    loose_ok = check_instruction_loose(checker, resp, kw)
                strict_list.append(strict_ok)
                loose_list.append(loose_ok)
                per_instruction.append({
                    "conversation_id": doc["conversation_id"],
                    "language": language,
                    "turn_index": t,
                    "instruction_id": inst_id,
                    "passed": int(strict_ok)
                })
            per_turn_lang_bucket.append({
                "conversation_id": doc["conversation_id"],
                "turn_index": t,
                "language": language,
                "strict_follow_list": strict_list,
                "loose_follow_list": loose_list
            })

        return {
            "per_instruction": per_instruction,
            "per_turn_lang_bucket": per_turn_lang_bucket
        }

    # --- Aggregation registration ---
    def aggregation(self):
        return {"multi_if_metrics": self._aggregate}

    def higher_is_better(self):
        return {"multi_if_metrics": True}

    # --- Aggregation logic ---
    def _aggregate(self, results: Iterable[Dict[str, Any]]):
        try:
            import numpy as np
            from scipy.stats import bootstrap
        except Exception:
            np = None
            bootstrap = None

        turn_counts = defaultdict(lambda: [0, 0])
        lang_counts = defaultdict(lambda: [0, 0])
        conv_fail_flag = {}
        metrics_data = defaultdict(lambda: defaultdict(list))

        for r in results:
            for e in r["per_instruction"]:
                t = e["turn_index"]; lang = e["language"]
                turn_counts[t][0] += e["passed"]; turn_counts[t][1] += 1
                lang_counts[lang][0] += e["passed"]; lang_counts[lang][1] += 1
                cid = e["conversation_id"]
                if cid not in conv_fail_flag:
                    conv_fail_flag[cid] = False
                if e["passed"] == 0:
                    conv_fail_flag[cid] = True
            for entry in r["per_turn_lang_bucket"]:
                t = entry["turn_index"]; lang = entry["language"]
                metrics_data[t][lang].append(entry)
                metrics_data[t]["all_languages"].append(entry)

        metrics: Dict[str, Any] = {}
        overall_pass = sum(v[0] for v in turn_counts.values())
        overall_total = sum(v[1] for v in turn_counts.values())

        for t,(p,tot) in turn_counts.items():
            metrics[f"turn_{t+1}_acc"] = p / tot if tot else 0.0
        for l,(p,tot) in lang_counts.items():
            metrics[f"lang_{l}_acc"] = p / tot if tot else 0.0
        metrics["overall_accuracy"] = overall_pass / overall_total if overall_total else 0.0
        if "turn_1_acc" in metrics and "turn_3_acc" in metrics and metrics["turn_1_acc"] > 0:
            metrics["turn1_to_turn3_relative_drop"] = 1 - (metrics["turn_3_acc"] / metrics["turn_1_acc"])
        else:
            metrics["turn1_to_turn3_relative_drop"] = 0.0

        conv_total = len(conv_fail_flag)
        conv_success = sum(1 for v in conv_fail_flag.values() if not v)
        metrics["conversation_all_turn_success"] = conv_success / conv_total if conv_total else 0.0

        # Rich strict/loose per-turn/lang
        for t, lang_dict in metrics_data.items():
            metrics[f"turn_{t+1}_prompts_number"] = len(lang_dict.get("all_languages", []))
            for lang, entries in lang_dict.items():
                if not entries:
                    continue
                sp_hits, si_hits, lp_hits, li_hits = [], [], [], []
                for e in entries:
                    s_list = e["strict_follow_list"]
                    l_list = e["loose_follow_list"]
                    sp_hits.append(1 if all(s_list) else 0)
                    si_hits.extend(int(x) for x in s_list)
                    lp_hits.append(1 if all(l_list) else 0)
                    li_hits.extend(int(x) for x in l_list)
                def mean(a): return sum(a)/len(a) if a else 0.0
                sp = mean(sp_hits); si = mean(si_hits); lp = mean(lp_hits); li = mean(li_hits)
                overall_avg = (sp + si + lp + li) / 4.0
                prefix = f"turn_{t+1}_{lang}"
                metrics[f"{prefix}_overall"] = overall_avg
                metrics[f"{prefix}_strict_prompt_acc"] = sp
                metrics[f"{prefix}_strict_instruction_acc"] = si
                metrics[f"{prefix}_loose_prompt_acc"] = lp
                metrics[f"{prefix}_loose_instruction_acc"] = li
                if np is not None and bootstrap is not None and len(sp_hits) > 1:
                    try:
                        sp_ci = bootstrap((np.array(sp_hits),), np.mean).confidence_interval
                        si_ci = bootstrap((np.array(si_hits),), np.mean).confidence_interval
                        lp_ci = bootstrap((np.array(lp_hits),), np.mean).confidence_interval
                        li_ci = bootstrap((np.array(li_hits),), np.mean).confidence_interval
                        metrics[f"{prefix}_cis_strict_prompt"] = (sp_ci.low, sp_ci.high)
                        metrics[f"{prefix}_cis_strict_instruction"] = (si_ci.low, si_ci.high)
                        metrics[f"{prefix}_cis_loose_prompt"] = (lp_ci.low, lp_ci.high)
                        metrics[f"{prefix}_cis_loose_instruction"] = (li_ci.low, li_ci.high)
                    except Exception:
                        pass

        return metrics
    
    def doc_to_target(self, doc):
        """
        Multi-IF evaluation is custom; we don’t have a single “gold target”.
        Returning empty string satisfies abstract interface.
        """
        return ""

    def process_results(self, doc, results):
        """
        Harness calls this after construct_requests().
        We aren’t using the standard request pipeline (construct_requests returns []),
        so there are no per-doc standard metrics to emit here.
        We return an empty dict; all real metrics are computed later via the
        custom multi-turn path (once you wire it) or remain empty for now.
        """
        return {}



# -------- Factory function registration (preferred) --------
# @register_task("multi-if")
def build_multi_if(**kwargs):
    t = MultiIF(**kwargs)
    t.task_name = "multi-if"   # ensure it’s set
    t.download()
    return t


