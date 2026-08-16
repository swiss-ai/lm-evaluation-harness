# Common Pitfalls and Troubleshooting Guide

This document highlights common pitfalls and troubleshooting tips when using this library. We'll continue to add more tips as we discover them.

## YAML Configuration Issues

### Newline Characters in YAML (`\n`)

**Problem:** When specifying newline characters in YAML, they may be interpreted incorrectly depending on how you format them.

```yaml
# ❌ WRONG: Single quotes don't process escape sequences
generation_kwargs:
  until: ['\n']  # Gets parsed as the literal characters '\' and 'n' i.e "\\n"

```
```yaml
# ✅ RIGHT: Use double quotes for escape sequences
generation_kwargs:
  until: ["\n"]  # Gets parsed as an actual newline character

```

**Solutions:**
- Use double quotes for strings containing escape sequences
- For multiline content, use YAML's block scalars (`|` or `>`)
- When generating YAML programmatically, be careful with how template engines handle escape sequences

### Quoting in YAML

**When to use different types of quotes:**

- **No quotes**: Simple values (numbers, booleans, alphanumeric strings without special characters)
  ```yaml
  simple_value: plain text
  number: 42

  ```

- **Single quotes (')**:
  - Preserves literal values
  - Use when you need special characters to be treated literally
  - Escape single quotes by doubling them: `'It''s working'`
  ```yaml
  literal_string: 'The newline character \n is not processed here'
  path: 'C:\Users\name'  # Backslashes preserved

  ```

- **Double quotes (")**:
  - Processes escape sequences like `\n`, `\t`, etc.
  - Use for strings that need special characters interpreted
  - Escape double quotes with backslash: `"He said \"Hello\""`
  ```yaml
  processed_string: "First line\nSecond line"  # Creates actual newline
  unicode: "Copyright symbol: \u00A9"  # Unicode character

  ```

## Evaluation Result Validity

A run can exit 0 and still write a number that is not a measurement. These are three ways we
hit that in practice, all on this fork; none of them raise an error, so they are easy to carry into a results table.

### An unreachable judge model uses the full walltime before writing NaN

**Problem:** LLM-judge-backed tasks (e.g. `harmbench`) call
out to a separately-served judge model. If that judge is not actually being served, every call
fails, each sample is retried several times, and the task still "completes" - with
`{"score": np.nan}` for every sample and a NaN aggregate. Neither the exit code nor the log volume distinguishes this from a real run; the only tell is the aggregate itself, at walltime end (the retry loop, not generation, is what exhausted it).

**What to check:** before relying on any judge-backed task, confirm the judge is actually being
served under the name the task expects - a stale or renamed serving endpoint produces exactly
this symptom, not a connection error. If a task's aggregate is NaN, treat it as "did not run",
not as a score.

### An all-NaN aggregate is written like any other result

**Problem:** related to the above but distinct - even where a judge is reachable, it can return
values the metric can't use (observed on a toxicity-judge task: every sample scored NaN with the
judge responding normally, not erroring). In both cases the aggregate lands in the results JSON
indistinguishable in shape from a valid score; nothing marks the task as failed.

**What to check:** treat an all-NaN aggregate as a failed task rather than a low score when
post-processing results, particularly before feeding a results table into any further comparison
or averaging - an unnoticed NaN can silently drop out of an average in a way a real low score
would not.

### `--apply_chat_template` on a `loglikelihood` task changes what is being measured

**Problem:** `loglikelihood` tasks like `lambada` or `squadv2` work by measuring how much
probability the model assigns to a natural continuation of a text fragment. Under
`--apply_chat_template`, that fragment is wrapped as a user turn, and the target text is then
scored as if it were the start of an assistant reply - a different, and generally much less
likely, continuation. Measured on one base model: `lambada` accuracy went from 0.72 (perplexity
3.78) to exactly 0.0 (perplexity 8569); `squadv2` F1 from 32.1 to 10.4. The only signal is a log
line at eval start - *"Chat template formatting change affects loglikelihood and multiple-choice
tasks."* - easy to miss in a large task list, and nothing prevents the run from completing and
reporting the degenerate score as final.

**What to check:** if you are applying a chat template, check whether every task in your list is
actually meant to be read that way. This is not specific to any one task set - it applies to any
`loglikelihood`-type task run with `--apply_chat_template` on.
