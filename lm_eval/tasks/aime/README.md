# AIME

## Data

Set of 90 mathematical problems from the American Invitational Mathematics Examination
(AIME), editions 2022-2024. The problems have been extracted from the Art of Problem Solving (AOPS) wiki page.

Homepage: https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions

HF Dataset: https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions

### Implementation

The metric and regexes used are taken from the implementation of GSM8K, given the similarity of the tasks and the lack of paper corresponding to the AIME benchmark. An extra filter `no-filter` is added to provide a baseline evaluating the raw output of the model.
### Citation

```text
@dataset{aime_1983_2024,
  author = {Hemish Veeraboina},
  title = {AIME Problem Set 1983-2024},
  year = {2024},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/hemishveeraboina/aime-problem-set-1983-2024}
}

@dataset{aime_2024,
  author = {Maxwell Jia},
  title = {AIME Problem Set 2024},
  year = {2024},
  publisher = {Huggingface},
  url = {https://huggingface.co/datasets/Maxwell-Jia/AIME_2024}
}

@dataset{aime_2025,
  author = {math-ai},
  title = {AIME Problem Set 2025},
  year = {2025},
  publisher = {Huggingface},
  url = {https://huggingface.co/datasets/math-ai/aime25}
}
```

### Groups, Tags, and Tasks

#### Groups

* `math_word_problems`

#### Tasks

* `aime`: `AIME 1983-2024 problems`
* `aime24`: `AIME 2024 problems`
* `aime25`: `AIME 2025 problems`

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
