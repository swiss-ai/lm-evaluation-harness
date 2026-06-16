# BFCL v3

This directory integrates the Berkeley Function Calling Leaderboard v3 tasks into
the evaluation harness in prompt mode.

Implemented task groups:

- `bfcl_v3`: the supported single-turn and multi-turn suites.
- `bfcl_v3_single_turn`: Python AST, Java, JavaScript, and live single-turn categories.
- `bfcl_v3_multi_turn`: base, missing-function, missing-parameter, and long-context
  multi-turn categories.
- `bfcl_v3_python`: Python-only single-turn and multi-turn categories.
- `bfcl_v3_apertus`: BFCL categories using Apertus tool-call blocks.

The task expects models to emit BFCL prompt-mode calls, for example:

```text
[calculate_triangle_area(base=10, height=5)]
```

The Apertus variants expect tool calls in this format:

```text
<|tools_prefix|>[{"calculate_triangle_area": {"base": 10, "height": 5}}]<|tools_suffix|>
```

They pass structured messages and the per-document BFCL function list through
the model tokenizer's `apply_chat_template(..., tools=[...])` path, so
`bfcl_v3_apertus` tasks require `--apply_chat_template`. The emitted Apertus
tool blocks are parsed back into BFCL's canonical call structure before scoring.
Java and JavaScript variants additionally translate JSON argument values into
the language-literal strings expected by the existing BFCL Java/JavaScript
checkers.

Scoring uses the BFCL v3 AST and multi-turn checkers vendored from the reference
implementation.
Java, JavaScript, and multi-turn categories require the `bfcl` optional extra:

```bash
pip install -e ".[bfcl]"
```

Multi-turn categories use the harness `generate_until_multiturn` output type, which
batches active conversations by wave. Each BFCL episode alternates model generation,
constrained parsing of prompt-mode or Apertus tool calls, execution against
simulated BFCL tool APIs, and appending execution results to the conversation. The
executor uses an AST dispatcher over the vendored simulated tools rather than
evaluating arbitrary model text directly.

Distributed multi-turn evaluation is not supported yet; run `generate_until_multiturn`
tasks with `world_size=1`.
