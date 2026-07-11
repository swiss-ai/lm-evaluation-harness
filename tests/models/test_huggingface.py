from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tokenizers
import torch
from packaging.version import parse as parse_version

from lm_eval import tasks
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"
task_manager = tasks.TaskManager()

TEST_STRING = "foo bar"

# Bound to names (not passed as string literals) so the *_token kwargs don't trip
# ruff's flake8-bandit S105/S106 hardcoded-password false positive.
OPEN_T = "<think>"
CLOSE_T = "</think>"


class Test_HFLM:
    torch.use_deterministic_algorithms(True)
    task_list = task_manager.load_task_or_group(["arc_easy", "gsm8k", "wikitext"])
    version_minor = sys.version_info.minor
    multiple_choice_task = task_list["arc_easy"]  # type: ignore
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: list[Instance] = multiple_choice_task.instances
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.set_fewshot_seed(1234)  # fewshot random generator seed
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until: list[Instance] = generate_until_task.instances
    rolling_task = task_list["wikitext"]  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: list[Instance] = rolling_task.instances

    MULTIPLE_CH_RES = [
        -41.902435302734375,
        -42.939308166503906,
        -33.914180755615234,
        -37.07139205932617,
        -22.95258331298828,
        -20.342208862304688,
        -14.818366050720215,
        -27.942853927612305,
        -15.80704116821289,
        -15.936427116394043,
        -13.052018165588379,
        -18.04828453063965,
        -13.345029830932617,
        -13.366025924682617,
        -12.127134323120117,
        -11.872495651245117,
        -47.10598373413086,
        -47.76410675048828,
        -36.4406852722168,
        -50.0289421081543,
        -16.72093963623047,
        -18.535587310791016,
        -26.46993637084961,
        -20.355995178222656,
        -17.757919311523438,
        -21.80595588684082,
        -33.1990852355957,
        -39.28636932373047,
        -14.759679794311523,
        -16.753942489624023,
        -11.486852645874023,
        -15.42177677154541,
        -13.15798282623291,
        -15.887393951416016,
        -15.28614616394043,
        -12.339089393615723,
        -44.59441375732422,
        -55.40888214111328,
        -52.70050811767578,
        -56.25089645385742,
    ]
    generate_until_RES = [
        " The average of $2.50 each is $",
        " A robe takes 2 bolts of blue fiber and half",
        " $50,000 in repairs.\n\nQuestion",
        " He runs 1 sprint 3 times a week.",
        " They feed each of her chickens three cups of mixed",
        " The price of the glasses is $5, but",
        " The total percentage of students who said they like to",
        " Carla is downloading a 200 GB file. Normally",
        " John drives for 3 hours at a speed of 60",
        " Eliza sells 4 tickets to 5 friends so she",
    ]
    ROLLING_RES = [
        -3603.6328125,
        -19779.23974609375,
        -8834.16455078125,
        -27967.591796875,
        -7636.794982910156,
        -9491.93505859375,
        -41043.4248046875,
        -8397.689819335938,
        -45969.47155761719,
        -7158.90625,
    ]
    LM = HFLM(pretrained="EleutherAI/pythia-70m", device="cpu", dtype="float32")

    def test_logliklihood(self) -> None:
        res = self.LM.loglikelihood(self.MULTIPLE_CH)
        _RES, _res = self.MULTIPLE_CH_RES, [r[0] for r in res]
        # log samples to CI
        dir_path = Path("test_logs")
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"outputs_log_{self.version_minor}.txt"
        file_path = file_path.resolve()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in _res))
        assert np.allclose(_res, _RES, atol=1e-2)
        # check indices for Multiple Choice
        argmax_RES, argmax_res = (
            np.argmax(np.array(_RES).reshape(-1, 4), axis=1),
            np.argmax(np.array(_res).reshape(-1, 4), axis=1),
        )
        assert (argmax_RES == argmax_res).all()

    def test_generate_until(self) -> None:
        res = self.LM.generate_until(self.generate_until)
        assert res == self.generate_until_RES

    def test_logliklihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        assert np.allclose(res, self.ROLLING_RES, atol=1e-1)

    def test_toc_encode(self) -> None:
        res = self.LM.tok_encode(TEST_STRING)
        assert res == [12110, 2534]

    def test_toc_decode(self) -> None:
        res = self.LM.tok_decode([12110, 2534])
        assert res == TEST_STRING

    def test_batch_encode(self) -> None:
        res = self.LM.tok_batch_encode([TEST_STRING, "bar foo"])[0].tolist()
        assert res == [[12110, 2534], [2009, 17374]]

    def test_model_generate(self) -> None:
        context = self.LM.tok_batch_encode([TEST_STRING])[0]
        res = self.LM._model_generate(context, max_length=10, stop=["\n\n"])
        res = self.LM.tok_decode(res[0])
        if parse_version(tokenizers.__version__) >= parse_version("0.20.0"):
            assert res == "foo bar\n<bazhang> !info bar"
        else:
            assert res == "foo bar\n<bazhang>!info bar"

    def test_generate_until_length_info_bounds_batch_overflow(self) -> None:
        """Regression: HF batched generation runs until ALL sequences in the batch
        stop, so a short sequence over-generates past its own stop up to the batch max.
        Per-response length/thinking-format must be measured on a bounded view (that
        overflow trimmed), and the scored output must be unchanged.
        """
        from lm_eval.models.utils import postprocess_generated_text

        eos = self.LM.tokenizer.eos_token_id
        content = [t for t in self.LM.tok_encode("hello world foo bar") if t != eos][:3]

        def _reqs(until):
            return [
                Instance(
                    request_type="generate_until",
                    doc={},
                    arguments=("prompt", {"until": list(until), "max_gen_toks": 16}),
                    idx=i,
                    metadata=("t", i, 1),
                )
                for i in range(2)
            ]

        def _patch(rows):
            def fake(context, attention_mask=None, stop=None, max_length=None, **kw):
                gen = [rows for _ in range(context.shape[0])]
                return torch.cat(
                    [context, torch.tensor(gen, dtype=context.dtype)], dim=1
                )

            self.LM._model_generate = fake

        _think_attrs = (
            "think_end_token",
            "think_start_token",
            "think_end_token_str",
            "track_thinking_metrics",
            "think_open_prefilled",
        )
        saved = {k: getattr(self.LM, k, None) for k in _think_attrs}
        try:
            # (A) trailing-eos overflow: 3 real tokens + 7 eos -> length must be 3, not 10
            _patch(content + [eos] * 7)
            reqs = _reqs(["\n\n"])
            self.LM.generate_until(reqs)
            for r in reqs:
                assert r.length_info[0]["response_length_tokens"] == 3, r.length_info

            # (B) string-stop overflow: the row keeps emitting REAL tokens past its own
            # stop (whole-batch stopping), so measurement must bound at STOP -> exactly
            # the 2 tokens before it, not the full row.
            stop_ids = [t for t in self.LM.tok_encode(" STOP") if t != eos]
            stop_str = self.LM.tok_decode(stop_ids, skip_special_tokens=False)
            row = content[:2] + stop_ids + [content[2], content[2]] + [eos] * 3
            _patch(row)
            reqs = _reqs([stop_str])
            res = self.LM.generate_until(reqs)
            bounded = self.LM.tok_decode(content[:2], skip_special_tokens=False)
            for r in reqs:
                info = r.length_info[0]
                assert info["response_length_tokens"] == 2, info  # exact, not len(row)
                assert info["response_length_chars"] == len(bounded), info
            # (C) scoring unchanged vs the pre-fix postprocess of the full generation
            assert res[0] == postprocess_generated_text(
                self.LM.tok_decode(row), [stop_str], None
            )

            # (D) int-token close path: has_close/span from the RAW close boundary the
            # strip uses; response_length from the bounded view (eos overflow excluded).
            toks = [
                t for t in self.LM.tok_encode("Alpha Bravo Charlie Delta") if t != eos
            ]
            O, R1, R2, A1 = toks[:4]
            C = next(t for t in self.LM.tok_encode(" Zulu") if t != eos)
            self.LM.think_end_token = C
            self.LM.think_end_token_str = self.LM.tok_decode(
                [C], skip_special_tokens=False
            )
            self.LM.think_start_token = self.LM.tok_decode(
                [O], skip_special_tokens=False
            )
            self.LM.track_thinking_metrics = True
            self.LM.think_open_prefilled = False
            _patch([O, R1, R2, C, A1] + [eos] * 4)
            reqs = _reqs(["\n\n"])
            self.LM.generate_until(reqs)
            info = reqs[0].length_info[0]
            assert info["thinking_format_has_close"] == 1, info
            assert info["thinking_format_correct"] == 1, info
            assert info["thinking_length_tokens"] == 4, info  # [O,R1,R2,C]
            assert info["response_length_tokens"] == 5, info  # eos overflow excluded
            assert info["thinking_length_tokens"] <= info["response_length_tokens"]

            # (E) int path, string stop BEFORE the close, close only in overflow: the
            # bounded response opened but never closed, so has_close must be 0
            # (batch-invariant), not 1 from a sibling-driven overflow close.
            halt = [t for t in self.LM.tok_encode(" HALT") if t != eos]
            halt_str = self.LM.tok_decode(halt, skip_special_tokens=False)
            _patch([O, R1] + halt + [C, A1] + [eos] * 3)
            reqs = _reqs([halt_str])
            res_overflow = self.LM.generate_until(reqs)
            info = reqs[0].length_info[0]
            assert info["thinking_format_has_open"] == 1, info
            assert info["thinking_format_has_close"] == 0, (
                info
            )  # overflow close excluded
            assert info["thinking_format_correct"] == 0, info
            assert "thinking_length_tokens" not in info, info
            # exact bounded count: only [O, R1] precede the stop
            assert info["response_length_tokens"] == 2, info

            # (F) SCORING must agree with the metric and be batch-invariant. The same
            # doc without overflow (no close, no answer past the stop) must score
            # identically: the strip may only key off an IN-BOUNDS close, never one
            # that exists solely because a batch sibling kept the batch generating.
            _patch([O, R1] + halt + [eos] * 4)  # what bs=1 would have produced
            reqs = _reqs([halt_str])
            res_bs1 = self.LM.generate_until(reqs)
            assert res_overflow[0] == res_bs1[0], (
                f"scored text is batch-dependent: {res_overflow[0]!r} vs {res_bs1[0]!r}"
            )
            # metric said has_close=0, so nothing may have been stripped at a close
            assert reqs[0].length_info[0]["thinking_format_has_close"] == 0
        finally:
            self.LM.__dict__.pop("_model_generate", None)  # restore the bound method
            for k, v in saved.items():  # restore think attrs to their __init__ values
                setattr(self.LM, k, v)

    def test_int_close_uses_in_context_close_string(self) -> None:
        """`correct` must not depend on how the close token decodes in ISOLATION.

        On the int path the format order/re-open checks string-search the close inside
        `gen_text`, which renders the token IN CONTEXT. `think_end_token_str` is decoded
        from the id alone and can differ (leading space, merge). If the search misses,
        every well-formed response would be flagged malformed and lose its
        `thinking_length_*`. The close string must instead be read off the token offsets.
        """
        eos = self.LM.tokenizer.eos_token_id
        nz = [t for t in self.LM.tok_encode("Alpha reason answer") if t != eos]
        O, R, A = nz[0], nz[1], nz[2]
        C = next(t for t in self.LM.tok_encode(" Zulu") if t != eos)

        saved = {
            k: getattr(self.LM, k, None)
            for k in (
                "think_end_token",
                "think_start_token",
                "think_end_token_str",
                "track_thinking_metrics",
                "think_open_prefilled",
            )
        }
        try:
            self.LM.think_end_token = C
            self.LM.think_start_token = self.LM.tok_decode(
                [O], skip_special_tokens=False
            )
            # deliberately WRONG isolated decode: the old code string-searched this and
            # missed, yielding correct=0 for a perfectly well-formed response.
            self.LM.think_end_token_str = "###NEVER-APPEARS###"
            self.LM.track_thinking_metrics = True
            self.LM.think_open_prefilled = False

            def fake(context, attention_mask=None, stop=None, max_length=None, **kw):
                gen = [[O, R, C, A] + [eos] * 3]
                return torch.cat(
                    [context, torch.tensor(gen, dtype=context.dtype)], dim=1
                )

            self.LM._model_generate = fake
            req = Instance(
                request_type="generate_until",
                doc={},
                arguments=("p", {"until": ["\n\n"], "max_gen_toks": 16}),
                idx=0,
                metadata=("t", 0, 1),
            )
            self.LM.generate_until([req])
            info = req.length_info[0]
            assert info["thinking_format_has_close"] == 1, info
            assert info["thinking_format_correct"] == 1, info  # was 0 before the fix
            assert info["thinking_length_tokens"] == 3, info  # [O, R, C]
        finally:
            self.LM.__dict__.pop("_model_generate", None)
            for k, v in saved.items():
                setattr(self.LM, k, v)

    def test_special_token_close_is_stripped_at_token_level(self) -> None:
        """A close erased by `skip_special_tokens=True` must still strip.

        The scored text comes from `tok_decode(...)` (skip_special_tokens=True). A close
        registered as a *special* token (DeepSeek-R1 style) is dropped there, so a string
        strip could never find it: the reasoning would survive, glued to the answer.
        Such a close must be routed through the token-id strip instead. A plain added
        token (Qwen3 style) survives the decode and must keep the string path.
        """
        from transformers import AutoTokenizer

        def build(special: bool):
            tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
            if special:
                tok.add_special_tokens({"additional_special_tokens": [OPEN_T, CLOSE_T]})
            else:
                tok.add_tokens([OPEN_T, CLOSE_T])
            close = {"think_end_token": CLOSE_T}
            lm = HFLM(
                pretrained="EleutherAI/pythia-70m",
                tokenizer=tok,
                device="cpu",
                **close,
            )
            row = tok.encode(f"reason{CLOSE_T}answer")

            def fake(context, attention_mask=None, stop=None, max_length=None, **kw):
                return torch.cat(
                    [context, torch.tensor([row], dtype=context.dtype)], dim=1
                )

            lm._model_generate = fake
            req = Instance(
                request_type="generate_until",
                doc={},
                arguments=("p", {"until": ["\n\n"], "max_gen_toks": 16}),
                idx=0,
                metadata=("t", 0, 1),
            )
            return lm, lm.generate_until([req])[0], req.length_info[0]

        # special close -> demoted to its token id so the id-based strip can fire
        lm, scored, info = build(special=True)
        assert isinstance(lm.think_end_token, int), lm.think_end_token
        assert lm.think_end_token_str == CLOSE_T
        assert scored == "answer", scored  # NOT 'reasonanswer'
        assert info["thinking_format_correct"] == 1, info

        # plain added close -> survives the scoring decode, keeps the string path
        lm, scored, info = build(special=False)
        assert lm.think_end_token == CLOSE_T
        assert scored == "answer", scored
        assert info["thinking_format_correct"] == 1, info

    def test_track_thinking_metrics_override(self) -> None:
        """Tracking derives from the close token alone; `enable_thinking` never gates it."""

        def lm(**kw):
            return HFLM(pretrained="EleutherAI/pythia-70m", device="cpu", **kw)

        close = {"think_end_token": CLOSE_T}
        # derived: on whenever a close is known, regardless of the reasoning mode
        assert lm(**close).track_thinking_metrics is True
        assert lm(**close, enable_thinking=False).track_thinking_metrics is True
        # forced off / forced on, independently of enable_thinking
        assert lm(**close, track_thinking_metrics=False).track_thinking_metrics is False
        assert (
            lm(
                **close, enable_thinking=False, track_thinking_metrics=True
            ).track_thinking_metrics
            is True
        )
        # cannot track without a close token (has_close / the span are undefined)
        assert lm(track_thinking_metrics=True).track_thinking_metrics is False
        # an open alone is not enough either
        open_only = {"think_start_token": OPEN_T}
        assert (
            lm(**open_only, track_thinking_metrics=True).track_thinking_metrics is False
        )

    def test_autodetect_think_tokens_matrix(self) -> None:
        """End-to-end decoupling on a real model with a synthetic reasoning template.

        `enable_thinking` must not influence detection, the strip, or the metrics;
        `autodetect_think_tokens` is the only lever over template scanning.
        """
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        tok.chat_template = (
            "{% set inner_token = '<think>' %}{% set outer_token = '</think>' %}"
        )

        def lm(**kw):
            return HFLM(
                pretrained="EleutherAI/pythia-70m", tokenizer=tok, device="cpu", **kw
            )

        # DEFAULT: autodetect is opt-in, so a reasoning template is NOT scanned.
        # No close -> no strip, no thinking metrics.
        m = lm()
        assert (m.think_start_token, m.think_end_token) == (None, None)
        assert m.track_thinking_metrics is False

        # opt in -> tokens detected -> strip armed (think_end_token set) + metrics on
        m = lm(autodetect_think_tokens=True)
        assert (m.think_start_token, m.think_end_token) == (OPEN_T, CLOSE_T)
        assert m.track_thinking_metrics is True

        # enable_thinking must NOT influence detection either way (the decoupling)
        m = lm(autodetect_think_tokens=True, enable_thinking=False)
        assert m.think_end_token == CLOSE_T
        assert m.track_thinking_metrics is True
        m = lm(enable_thinking=True)  # autodetect still off -> still nothing
        assert m.think_end_token is None
        assert m.track_thinking_metrics is False

        # an explicit close works without autodetect (deliberate strip request)
        close = {"think_end_token": CLOSE_T}
        m = lm(**close)
        assert m.think_end_token == CLOSE_T
        assert m.think_start_token is None  # not detected; close-only style
        assert m.track_thinking_metrics is True

        # open only -> nothing to strip or measure
        open_only = {"think_start_token": OPEN_T}
        m = lm(**open_only)
        assert m.think_start_token == OPEN_T
        assert m.think_end_token is None
        assert m.track_thinking_metrics is False
