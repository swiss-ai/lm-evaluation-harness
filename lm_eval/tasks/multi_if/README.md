# Multi-IF Evaluation

### Tasks

| Task | `output_type` | Turns | Metrics |
|---|---|---|---|
| `multi-if` | `generate_until` | turn 1 only | `{prompt,inst}_level_{strict,loose}_acc`, `overall`, `overall_<language>` |
| `multi-if-multiturn` | `multi_turn_generate` | up to `MAX_TURNS` (default 3) | same, prefixed `turn_{k}_` |

`multi-if-multiturn` plays each conversation turn by turn: the prompt for turn k is
the full history (user turns 1..k plus the model's own replies to turns 1..k-1),
rendered with the model's chat template. It requires `--apply_chat_template`:

```bash
lm_eval --model vllm --model_args pretrained=<model> \
  --tasks multi-if-multiturn --apply_chat_template
```

The number of turns played is `MAX_TURNS` in `utils.py`. Both tasks share
`process_docs` / `process_results`; `process_results` scores a string result as
a single turn and a list result as one response per turn.

### Scoring and edge cases

- **Constraints are cumulative.** Each turn's prompt states one new instruction,
  but turn k is scored against every instruction from turns 1..k
  (`turn_k_instruction_id_list` in the dataset already contains them). This
  matches the paper and the official scorer, so later turns are harder.
- **Conversations with fewer turns.** 56 of the 4501 conversations have only 2
  turns. They stop after turn 2 and emit no `turn_3_*` metrics, so turn-3
  scores are over 4445 conversations, not counted as failures.
- **`MAX_TURNS` lower than the conversation length.** Later turns are not
  played and their metrics are not reported. With `MAX_TURNS = 1` and greedy
  decoding, `turn_1_*` should match `multi-if` run with `--apply_chat_template`.
- **Instruction kwargs can change between turns** (4 conversations, e.g. a
  different forbidden word). Each turn is scored with its own kwargs.
- **Empty-string kwargs.** Conversations `2215:14:zh` and `2616:4:zh` have
  `keyword: ""` / `first_word: ""`. The checkers treat these as missing and
  pick a random value, so those instructions are scored non-deterministically.
- **`langdetect` is non-deterministic** (`change_case:english_*`,
  `language:response_language`): short responses can be scored differently
  across runs. This also affects `multi-if`.
- **Context length.** Every turn resends the full history. If prompt plus
  `max_gen_toks` exceeds the model's maximum length, the prompt is truncated
  from the left, dropping early turns. Use a `max_model_len` of at least about
  `MAX_TURNS * max_gen_toks` plus the prompts.
- **Turn-end token.** If the model's EOS token is not its chat turn-end token,
  generation runs past the turn. See `docs/multi_turn_rollout.md` for the fix.
- **Decoding differs from the paper.** These tasks decode greedily; the
  official setup uses temperature 0.6, top_p 0.9, max 1024 tokens
  (`--gen_kwargs temperature=0.6,top_p=0.9,do_sample=True,max_gen_toks=1024`).
- **Headline metric.** `overall` (`turn_k_overall` for the multi-turn task) is
  the paper's headline: the mean of the four accuracies, each aggregated over
  all docs first (`utils.agg_overall`). `overall_<language>` is the same score
  restricted to one language (english, french, hindi, portuguese, spanish,
  russian, italian, chinese). `overall` pools all languages, as the official
  scorer's `all_languages` does.

### Paper

Title: Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following
Abstract: https://arxiv.org/abs/2410.15553

We introduce Multi-IF, a new benchmark designed to assess LLMs' proficiency in following multi-turn and multilingual instructions. Multi-IF, which utilizes a hybrid framework combining LLM and human annotators, expands upon the IFEval by incorporating multi-turn sequences and translating the English prompts into another 7 languages, resulting in a dataset of 4501 multilingual conversations, where each has three turns. Our evaluation of 14 state-of-the-art LLMs on Multi-IF reveals that it presents a significantly more challenging task than existing benchmarks. All the models tested showed a higher rate of failure in executing instructions correctly with each additional turn. For example, o1-preview drops from 0.877 at the first turn to 0.707 at the third turn in terms of average accuracy over all languages. Moreover, languages with non-Latin scripts (Hindi, Russian, and Chinese) generally exhibit higher error rates, suggesting potential limitations in the models’ multilingual capabilities.

Homepage: https://github.com/facebookresearch/Multi-IF/tree/main


### Citation

```
@article{he2024multi,
  title={Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following},
  author={He, Yun and Jin, Di and Wang, Chaoqi and Bi, Chloe and Mandyam, Karishma and Zhang, Hejia and Zhu, Chen and Li, Ning and Xu, Tengyu and Lv, Hongjiang and others},
  journal={arXiv preprint arXiv:2410.15553},
  year={2024}
}
```
