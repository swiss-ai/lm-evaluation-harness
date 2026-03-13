# WMT Translation Benchmark

Translation evaluation using WMT24, WMT24++, and WMT25 datasets with **CometKiwi22**
as the evaluation metric (reference-free).


## Setup

```bash
pip install subset2evaluate unbabel-comet
```

No further setup needed — data is downloaded automatically via
[subset2evaluate](https://github.com/zouharvi/subset2evaluate) when the task
first runs (and cached afterwards).

## Running

```bash
# Single language pair
lm_eval --tasks wmt25_en-zh_CN --backend vllm ...

# All WMT25 tasks
lm_eval --tasks wmt25_translation --backend vllm ...

# Everything (wmt24 + wmt24pp + wmt25)
lm_eval --tasks wmt_translation --backend vllm ...
```

## Task structure

```
wmt_translation/
├── _wmt_translation_yaml          # Common template (custom_dataset + CometKiwi22)
├── wmt_translation.yaml           # Top-level group
├── wmt24_translation.yaml         # WMT24 group (11 language pairs)
├── wmt24pp_translation.yaml       # WMT24++ group (55 language pairs)
├── wmt25_translation.yaml         # WMT25 group (16 language pairs)
├── metric.py                      # CometKiwi22 aggregation-step metric
├── utils.py                       # Runtime data loading via subset2evaluate
├── create_tasks.py                # Regenerate YAML configs (for maintainers)
└── <dataset>_<lang_pair>.yaml     # Per-language-pair task configs (82 total)
```

## How it works

1. **Data loading** (`utils.py`): Uses `custom_dataset` to download WMT data via
   `subset2evaluate` at runtime. Data is cached in memory across tasks within a
   single evaluation run.

2. **Prompting**: Each source text is wrapped in a translation instruction prompt.

3. **Metric** (`metric.py`): [CometKiwi22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)
   is a reference-free translation quality model (scores 0–1, higher is better).
   It runs in the aggregation step (batched on GPU) for efficiency, following the
   same pattern as `aya_redteaming` and `multijail`.

## Regenerating task configs

To regenerate the YAML files (e.g. if new WMT language pairs are added):

```bash
python lm_eval/tasks/wmt_translation/create_tasks.py --hardcoded
# or: --from-subset2evaluate  (discovers pairs from subset2evaluate)
# or: --from-json /path/to/wmt_data.json
```

## Data source

Data comes from [subset2evaluate](https://github.com/zouharvi/subset2evaluate) which
provides WMT shared task datasets. See also
[apertus-translate](https://github.com/swiss-ai/apertus-translate).
