# IFBench

### Paper

Title: Generalizing Verifiable Instruction Following
Abstract: https://arxiv.org/abs/2507.02833

IFBench is a new, challenging benchmark for precise instruction following. It
introduces 58 new and out-of-distribution verifiable constraints (with
corresponding verification functions), combined with prompts from a held-out set
of WildChat. It is built upon and extends IFEval (Zhou et al. 2023) — the data
format (`key`, `prompt`, `instruction_id_list`, `kwargs`) and strict/loose
scoring are identical to IFEval; only the set of verifiable constraints differs.
Accepted to NeurIPS 2025 (Datasets & Benchmarks).

Homepage: https://github.com/allenai/IFBench

Dataset: https://huggingface.co/datasets/allenai/IFBench_test

### Citation

```
@misc{pyatkin2025generalizing,
   title={Generalizing Verifiable Instruction Following},
   author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
   year={2025},
   journal={Advances in Neural Information Processing Systems},
   volume={38},
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `ifbench`

The task reports four metrics, matching IFEval:
`prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`,
`inst_level_loose_acc`. The paper generally reports the **prompt-level loose
accuracy**.

### Implementation notes

`instructions.py`, `instructions_registry.py`, and `instructions_util.py` are
vendored from the IFBench repository (https://github.com/allenai/IFBench), with
only the intra-package imports rewritten to `from lm_eval.tasks.ifbench import ...`.
`utils.py` mirrors the IFEval harness scoring (the strict/loose logic is the same
as upstream IFBench's `evaluation_lib.py`), filtering `None`-valued kwargs before
building each instruction description. The harness `process_results` was verified
to reproduce IFBench's reference `run_eval.py` scores exactly on the upstream
`sample_output.jsonl` (prompt-level strict 0.2621, prompt-level loose 0.2966).

Extra Python dependencies beyond IFEval: `emoji`, `syllapy` (which needs
`setuptools<81`, as it imports `pkg_resources` without declaring it)
(plus `nltk` resources `punkt`, `punkt_tab`, `stopwords`,
`averaged_perceptron_tagger_eng`, which `instructions_util.py` downloads on
import).

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
