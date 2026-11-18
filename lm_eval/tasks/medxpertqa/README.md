# MedXpertQA

[MedXpertQA](https://arxiv.org/abs/2501.18362) (Zuo et al., 2025): expert-level medical questions from
specialty board exams, 10 options (A-J). Dataset: `TsinghuaC3I/MedXpertQA`, licence MIT.

## Tasks

- `medxpertqa_cot`: the text subset (`Text`, test split, 2,450 questions), zero-shot, generative. The prompt asks for
  step-by-step reasoning and a last line `Final Answer: X`, as `medqa_4options_cot` does; the last such line is
  scored by exact match. Declares `repeat_support`, so `--repeats N` reports mean@N.

The multimodal subset (`MM`) is not included.
