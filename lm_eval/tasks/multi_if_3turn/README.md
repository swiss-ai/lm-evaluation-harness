# Multi-IF, all three turns

`multi_if_3turn` runs the full multi-turn version of Multi-IF
([paper](https://arxiv.org/abs/2410.15553),
[code](https://github.com/facebookresearch/Multi-IF)), unlike `multi-if`,
which scores turn 1 only.

- **Data:** `facebook/Multi-IF`, 4501 conversations in 8 languages. 4445 have
  three turns and 56 have two.
- **Rollout:** `output_type: multi_turn_generate`, so the run needs
  `--apply_chat_template`. Each user turn is sent with the model's own earlier
  replies as assistant messages. There is no system prompt, as upstream.
- **Scoring:** turn N's reply is checked against turn N's instruction list,
  which already includes the earlier turns' instructions. The checkers are
  `multi-if`'s own (IFEval's, plus the multilingual ones). A conversation
  without turn N is not scored for turn N, as upstream's `metrics.py` skips it.
- **Metrics:**
  - `turnN_overall` is upstream's `turn_N_all_languages_overall`. It is the
    unweighted mean of prompt-level strict, instruction-level strict,
    prompt-level loose and instruction-level loose accuracy, pooled over all
    conversations that have turn N.
  - `overall` is the mean of the three turn scores.
- **Repeats:** all four metrics declare `native` repeat support, so with
  `--repeats N` they report `<metric>_mean_repeats`.
- **Sampling:** generation defaults follow upstream's vLLM script (temperature
  0.6, top_p 0.9, 1024 new tokens). Thinking runs override them.

`multi_if_3turn_half` is the same task on a fixed half of each language:
2249 conversations, drawn with a fixed seed (`utils.process_docs_half`). Use it
instead of `--limit`, which takes the first rows of a language-sorted dataset
and so drops whole languages.

Not included: per-language scores. Upstream also reports each turn per
language.
