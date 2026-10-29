# INCLUDE-2.0

Region-aware multilingual MCQ benchmark (91 languages, 127 language-country pairs, 220K+ questions), from
"INCLUDE-2.0: Disentangling Knowledge Gaps in Multilingual LLM Evaluation" (EMNLP 2026).
Dataset: https://huggingface.co/datasets/epfl-nlp/include-2.0 (public, MIT).

### Tasks
- `include_2_gen_0shot`: group over all 91 languages, 0-shot, open generation ("The answer is (X)"), exact match on the letter.
- `include_2_<language>_gen_0shot`: a single language, e.g. `include_2_yoruba_gen_0shot`.

Each row also carries `country`, `script` and `regional_feature` for slicing results.

### Citation
Romanou et al., INCLUDE-2.0: Disentangling Knowledge Gaps in Multilingual LLM Evaluation, EMNLP 2026. https://aclanthology.org/2026.emnlp-main.1344/
