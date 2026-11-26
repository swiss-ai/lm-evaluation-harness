"""WMT24++ MetricX-24 (lm_eval/tasks/wmt24pp_chrf/metricx.py): the input and score follow
google-research/metricx, batching does not change a segment's score, and every pair has a
wmt24pp_metricx task that keeps chrF and BLEU."""

import glob

import pytest

from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.wmt24pp_chrf import metricx


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

TASKS = "lm_eval/tasks/wmt24pp_chrf"


class _Tokenizer:
    """Bytes as token ids, plus an EOS, like the mT5 tokenizer's output shape."""

    pad_token_id = 0
    eos = 1

    def __call__(self, text, max_length, truncation):
        ids = [2 + (b % 200) for b in text.encode()][: max_length - 1] + [self.eos]
        return {"input_ids": ids}


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = transformers.MT5Config(
        vocab_size=metricx.SCORE_TOKEN_ID + 23,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=1,
        num_heads=2,
        tie_word_embeddings=False,
        decoder_start_token_id=0,
    )
    model = transformers.MT5ForConditionalGeneration(config).eval()
    # Point the score row along the decoder's output, so the toy scores land inside
    # MetricX's [0, 25] rather than all being clamped to 0.
    ids = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        out = model(
            input_ids=ids,
            decoder_input_ids=torch.zeros((1, 1), dtype=torch.long),
            output_hidden_states=True,
        )
        hidden = out.decoder_hidden_states[-1][0, 0]
        model.lm_head.weight[metricx.SCORE_TOKEN_ID] = 10 * hidden / hidden.norm() ** 2
    return model


ITEMS = [
    {"source": "Hello.", "hypothesis": "こんにちは。", "reference": "こんにちは。"},
    {
        "source": "A much longer sentence about the gallery.",
        "hypothesis": "ギャラリーについてのずっと長い文です。",
        "reference": "ギャラリーに関する、はるかに長い文。",
    },
    {"source": "Water.", "hypothesis": "水", "reference": "水。"},
]


def test_input_format_is_metricx24s():
    assert metricx.metricx_input("S", "H", "R") == "source: S candidate: H reference: R"


def test_process_results_feeds_chrf_bleu_and_metricx():
    doc = {"source": "Water.", "target": "水。"}
    out = metricx.process_results(doc, ["水"])
    assert out["chrf"] == out["bleu"] == ("水。", "水")
    assert out["metricx24"] == {
        "source": "Water.",
        "hypothesis": "水",
        "reference": "水。",
    }


def _one(model, tokenizer, item):
    """MetricX's own way: one segment, no padding, EOS removed, one decoder step."""
    text = metricx.metricx_input(item["source"], item["hypothesis"], item["reference"])
    ids = torch.tensor([tokenizer(text, 1536, True)["input_ids"][:-1]])
    with torch.no_grad():
        logits = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            decoder_input_ids=torch.zeros((1, 1), dtype=torch.long),
        ).logits
    return logits[0, 0, metricx.SCORE_TOKEN_ID].clamp(0, 25).item()


def test_batched_scores_match_one_at_a_time_in_item_order(tiny_model):
    tok = _Tokenizer()
    expected = [_one(tiny_model, tok, it) for it in ITEMS]
    got = metricx.score(
        ITEMS, model=tiny_model, tokenizer=tok, device="cpu", batch_size=2
    )
    assert got == pytest.approx(expected, abs=1e-5)
    assert len({round(s, 4) for s in got}) == len(ITEMS)


def test_scores_are_clamped_to_0_25(tiny_model):
    tok = _Tokenizer()
    with torch.no_grad():
        tiny_model.lm_head.weight[metricx.SCORE_TOKEN_ID] *= 1000
    try:
        got = metricx.score(ITEMS, model=tiny_model, tokenizer=tok, device="cpu")
    finally:
        with torch.no_grad():
            tiny_model.lm_head.weight[metricx.SCORE_TOKEN_ID] /= 1000
    assert all(0.0 <= s <= 25.0 for s in got)


def test_aggregation_refuses_an_empty_task():
    with pytest.raises(RuntimeError, match="no segments"):
        metricx.metricx24_agg([])


def test_every_chrf_pair_has_a_metricx_task():
    chrf = {
        p.split("wmt24pp_chrf_")[1]
        for p in glob.glob(f"{TASKS}/wmt24pp_chrf_en-*.yaml")
    }
    mx = {
        p.split("wmt24pp_metricx_")[1]
        for p in glob.glob(f"{TASKS}/wmt24pp_metricx_en-*.yaml")
    }
    assert chrf == mx and len(mx) == 55
    group = load_yaml(f"{TASKS}/wmt24pp_metricx.yaml")
    assert len(group["task"]) == 55


@pytest.mark.parametrize(
    "pair,bleu_fn",
    [("en-ja_JP", "bleu_ja"), ("en-zh_TW", "bleu_zh"), ("en-de_DE", None)],
)
def test_metricx_task_configs(pair, bleu_fn):
    cfg = load_yaml(f"{TASKS}/wmt24pp_metricx_{pair}.yaml")
    assert cfg["task"] == f"wmt24pp_metricx_{pair}"
    assert cfg["process_results"].__name__ == "process_results"
    metrics = {m["metric"]: m for m in cfg["metric_list"]}
    assert list(metrics) == ["chrf", "bleu", "metricx24"]
    assert metrics["metricx24"]["higher_is_better"] is False
    assert metrics["metricx24"]["aggregation"].__name__ == "metricx24_agg"
    bleu = metrics["bleu"]["aggregation"]
    assert (bleu.__name__ == bleu_fn) if bleu_fn else bleu == "bleu"
