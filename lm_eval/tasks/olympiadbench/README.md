# OlympiadBench

This integration covers the text-only, open-ended OlympiadBench subsets:

- `OE_TO_maths_en_COMP`
- `OE_TO_maths_zh_CEE`
- `OE_TO_maths_zh_COMP`
- `OE_TO_physics_en_COMP`
- `OE_TO_physics_zh_CEE`

The theorem-proving (`TP`) subsets are intentionally excluded for now because
the OlympiadBench reference implementation does not automatically score them.
The multimodal (`MM`) subsets are also excluded in this first pass.

Task groups:

- `olympiadbench`: alias for all supported text-only open-ended tasks.
- `olympiadbench_text_only`: all supported text-only open-ended tasks.
- `olympiadbench_text_only_math`: text-only math tasks.
- `olympiadbench_text_only_physics`: text-only physics tasks.

Individual task names follow the dataset config names, lowercased:

```text
olympiadbench_oe_to_maths_en_comp
olympiadbench_oe_to_maths_zh_cee
olympiadbench_oe_to_maths_zh_comp
olympiadbench_oe_to_physics_en_comp
olympiadbench_oe_to_physics_zh_cee
```

Scoring uses an adapted version of the OlympiadBench automatic math judge,
including boxed-answer extraction, numerical tolerance support, and symbolic
comparison through SymPy's LaTeX parser.

Run with the math extra installed:

```bash
pip install -e ".[olympiadbench]"
lm_eval --tasks olympiadbench_text_only ...
```
