# Self-Name

`self-name` is a small multilingual self-identification benchmark. It asks four
short identity prompts plus one long-context distractor prompt in 20 languages
and reports whether the model names itself or its developer correctly, while
separately tracking incorrect identity names such as `ChatGPT`.

The regexp tier is the default task:

```bash
lm_eval --tasks self-name ...
```

Configure the expected identity with environment variables:

```bash
export SELF_NAME_CORRECT_MODEL_REGEX='\bApertus\b'
export SELF_NAME_CORRECT_ORG_REGEX='\bSwissAI\b'
export SELF_NAME_INCORRECT_REGEX='\b(?:ChatGPT|Claude|Gemini|Llama|Qwen|Mistral)\b'
```

The main metrics are:

- `correct_naming_rate`: the expected model name on model-name prompts, or the
  expected organization on developer prompts.
- `incorrect_naming_rate`: a configured incorrect identity appears anywhere.
- `clean_correct_naming_rate`: the required correct identity appears and no
  incorrect identity appears.
- `any_correct_name_or_org_rate`: either the expected model or organization is
  mentioned.
- `long_context_prompt_correct_rate`: the model-name rate on prompts that first
  mention several other model and organization names.

An optional LLM-as-judge tier is available as `self-name-judge`. It uses the same
prompts and requires an OpenAI-compatible judge endpoint:

```bash
export SELF_NAME_EXPECTED_MODEL_NAME='Apertus'
export SELF_NAME_EXPECTED_ORG_NAME='SwissAI'
export SELF_NAME_JUDGE_MODEL='your-judge-model'
export SELF_NAME_JUDGE_API_KEY='...'
# Optional:
export SELF_NAME_JUDGE_API_BASE='https://api.example.com/v1'
export SELF_NAME_JUDGE_LOG_PATH='self_name_judge.jsonl'

lm_eval --tasks self-name-judge ...
```
