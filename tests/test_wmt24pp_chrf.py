"""WMT24++ chrF/BLEU (lm_eval/tasks/wmt24pp_chrf): Japanese and Chinese BLEU use
sacrebleu's own tokenizers. With the default 13a tokenizer a Japanese sentence, written
without spaces, is a single token, so BLEU is near 0 for any translation that is not
character-for-character the reference."""

import pytest
import sacrebleu

from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.wmt24pp_chrf import utils


REF = "ウエスト・ハリウッドの新ギャラリーで『ビセンテ・シソ：大地と水の記憶』が開催される。"
PRED = "ウェスト・ハリウッドの新ギャラリーにて「ヴィセンテ・シソ：大地と海の記憶」を開催する。"


def _has_mecab():
    try:
        sacrebleu.corpus_bleu(["テスト"], [["テスト"]], tokenize="ja-mecab")
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_mecab(), reason="needs lm_eval[wmt24pp] (sacrebleu[ja])")
def test_japanese_bleu_scores_words_not_whole_sentences():
    items = [(REF, PRED)]
    default = sacrebleu.corpus_bleu([PRED], [[REF]]).score
    assert default < 1  # one 13a token per sentence: no n-gram matches
    assert utils.bleu_ja(items) > 10
    assert (
        utils.bleu_ja(items)
        == sacrebleu.corpus_bleu([PRED], [[REF]], tokenize="ja-mecab").score
    )


def test_chinese_bleu_uses_the_zh_tokenizer():
    ref, pred = "今天天气很好，我们去公园散步。", "今天天气很好，我们去公园走走。"
    assert (
        utils.bleu_zh([(ref, pred)])
        == sacrebleu.corpus_bleu([pred], [[ref]], tokenize="zh").score
    )
    assert utils.bleu_zh([(ref, pred)]) > sacrebleu.corpus_bleu([pred], [[ref]]).score


def test_only_japanese_and_chinese_change_tokenizer():
    utils.require_bleu_tokenizer("de_DE")  # 13a, nothing to check
    assert utils.BLEU_TOKENIZERS == {"ja": "ja-mecab", "zh": "zh"}


def test_a_japanese_task_without_mecab_fails_before_generating(monkeypatch):
    def broken(*args, tokenize=None, **kwargs):
        raise ImportError("No module named 'MeCab'")

    monkeypatch.setattr(sacrebleu, "corpus_bleu", broken)
    with pytest.raises(RuntimeError, match=r"lm_eval\[wmt24pp\]"):
        utils.require_bleu_tokenizer("ja_JP")


@pytest.mark.parametrize(
    "pair,fn",
    [
        ("en-ja_JP", "bleu_ja"),
        ("en-zh_CN", "bleu_zh"),
        ("en-zh_TW", "bleu_zh"),
        ("en-de_DE", None),
    ],
)
def test_task_configs_pick_the_aggregation(pair, fn):
    cfg = load_yaml(f"lm_eval/tasks/wmt24pp_chrf/wmt24pp_chrf_{pair}.yaml")
    bleu = next(m for m in cfg["metric_list"] if m["metric"] == "bleu")
    if fn:
        assert callable(bleu["aggregation"]) and bleu["aggregation"].__name__ == fn
    else:
        assert bleu["aggregation"] == "bleu"
    assert [m["metric"] for m in cfg["metric_list"]] == ["chrf", "bleu"]
