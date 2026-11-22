# IFEval Multilingual

Instruction-following prompts written natively in 11 languages (ar, de, el, es, fr, hi, ja, ko, pl, sr, zh;
2,183 prompts), in the format of IFEval: IFEval's own instructions plus language-specific ones (local date and number
formats, quotation marks, formal / informal address, scripts, letters, radicals, honorifics, grammatical forms).
Dataset: [ilannissim/ifeval-multilingual](https://huggingface.co/datasets/ilannissim/ifeval-multilingual) (v15),
one config per language.

## Tasks

| task | |
|---|---|
| `ifeval_multilingual` | group: macro average of the 11 languages |
| `ifeval_multilingual_<code>` | one language |

Metrics as IFEval: `prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`,
`inst_level_loose_acc`. Generation as `ifeval` (greedy, `max_gen_toks` 1280).

## Scoring

`grader/` is the dataset's own scorer, copied from ritranslation-hooks (`configs/benchmark/ifeval`, commit 34f1cdb):

- IFEval instructions: IFEval's checkers (`lm_eval/tasks/ifeval`), some replaced per language by the manifest
  (`grader/manifest.yaml`: word and sentence counts without spaces in ja / zh, capitals, commas, forbidden words in
  any script).
- `native:*` instructions: their parameters are the JSON string `kwargs[i].native`; the checker is the type's
  (`grader/data_types.py`) or the language-profile entry's (`grader/entries.json`).

Changes from the source: imports, the profile lookup (`entries.json` instead of the profile files), Stanza on the CPU,
and the analyzers loaded through `grader/deps.py`.

Checked against the source scorer on the 6,744 Qwen answers recorded when the dataset was made: identical strict and
loose results for every instruction, all languages.

## Dependencies

Nothing to install by hand. A few instructions need language analyzers; `grader/deps.py` installs each on first use
(`pip install --no-deps --target`), only if the environment lacks it, and downloads its models:

| analyzer | instructions | size |
|---|---|---|
| stanza 1.10.1 + models de, el (gud), pl | de diminutives, el formal plural, pl non-virile past | ~0.6 GB |
| camel-tools 1.6.0 + MSA morphology, MLE and BERT disambiguators | ar broken plurals, idafa chains, verbal sentences | ~0.55 GB |
| opencc-python-reimplemented 0.1.7 | zh traditional characters | small |
| pymorphy3 2.0.6 + ru / uk dictionaries | ru / uk (not in v15) | ~0.05 GB |

- Cache: `$IFEVAL_MULTILINGUAL_CACHE`, else `~/.cache/ifeval_multilingual` (`STANZA_RESOURCES_DIR`, `CAMELTOOLS_DATA`
  respected if set). A file lock serializes jobs sharing it.
- Needs from the environment: `pip`, `torch`, `transformers` (CAMeL BERT; tested with 5.12.1), internet on first use.
- A failed install raises; no instruction is skipped.
- Analyzers run on the CPU: de, ar and pl take about 0.2–0.3 s per response.
