# MATH

## Paper
Measuring Mathematical Problem Solving With the MATH Dataset
https://arxiv.org/abs/2103.03874

Many intellectual endeavors require mathematical problem solving, but this skill remains beyond the capabilities of computers. To measure this ability in machine learning models, we introduce MATH, a new dataset of 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations.

NOTE: On `yxu/dev`, this task uses the Swiss six-shot instruction-model protocol
with a 2048-token generation budget. It shares upstream Minerva's maintained
answer extraction, equivalence checks and Math Verify scorer. Install
`lm-eval[math]`. The four-shot variant is in `lm_eval/tasks/minerva_math`.
See `docs/swiss_task_ports.md` for provenance and retained upstream fixes.

Homepage: https://github.com/hendrycks/math


## Citation
```
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
```

### Groups and Tasks

#### Groups

- `hendrycks_math`: the MATH benchmark with six fixed demonstrations.

#### Tasks

- `hendrycks_math_algebra`
- `hendrycks_math_counting_and_prob`
- `hendrycks_math_geometry`
- `hendrycks_math_intermediate_algebra`
- `hendrycks_math_num_theory`
- `hendrycks_math_prealgebra`
- `hendrycks_math_precalc`
- `hendrycks_math500`

### Checklist

The checklist is the following:

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * Answer extraction and scoring use the maintained Minerva helper, with the Swiss boxed-answer fallback.


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
