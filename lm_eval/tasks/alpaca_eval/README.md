# Alpaca-Eval 2

## Paper

Title: `Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators`

Abstract: `https://arxiv.org/abs/2404.04475`

`LLM-based auto-annotators have become a key component of the LLM development process due to their cost-effectiveness and scalability compared to human-based evaluation. However, these auto-annotators can introduce biases that are hard to remove. Even simple, known confounders such as preference for longer outputs remain in existing automated evaluation metrics. We propose a simple regression analysis approach for controlling biases in auto-evaluations. As a real case study, we focus on reducing the length bias of AlpacaEval, a fast and affordable benchmark for instruction-tuned LLMs that uses LLMs to estimate response quality. Despite being highly correlated with human preferences, AlpacaEval is known to favor models that generate longer outputs. We introduce a length-controlled AlpacaEval that aims to answer the counterfactual question: "What would the preference be if the model's and baseline's output had the same length?" To achieve this, we first fit a generalized linear model to predict the biased auto-annotator's preferences based on the mediators we want to control for (length difference) and other relevant features. We then obtain length-controlled preferences by predicting preferences while conditioning the GLM with a zero difference in lengths. Length-controlling not only improves the robustness of the metric to manipulations in model verbosity, but we also find that it increases the Spearman correlation with LMSYS Chatbot Arena from 0.94 to 0.98.`

Homepage: `https://github.com/tatsu-lab/alpaca_eval`

### Citation

```
@misc{dubois2023alpacafarm,
  title={AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback}, 
  author={Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani and Jimmy Ba and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
  year={2023},
  eprint={2305.14387},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  month = {5},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/alpaca_eval}}
}

@misc{dubois2025lengthcontrolledalpacaevalsimpleway,
      title={Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators}, 
      author={Yann Dubois and Balázs Galambosi and Percy Liang and Tatsunori B. Hashimoto},
      year={2025},
      eprint={2404.04475},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.04475}, 
}
```

### Implementation Details
1. The model generates responses to 805 AlpacaEval instructions.
2. Responses are compared pairwise against GPT-4 baseline outputs.
3. Llama-3.3-70B-Instruct (via CSCS SwissAI API) judges each pair using
   logprob-based classification, producing continuous preference scores.
4. A GLM debiases for output length to produce the length-controlled winrate.

### Groups, Tags, and Tasks

#### Groups

* `group_name`: `Short description`

#### Tags

* `tag_name`: `Short description`

#### Tasks

* `alpaca_eval`: Length-controlled AlpacaEval 2 using Llama-3.3-70B-Instruct as judge.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog