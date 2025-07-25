# RewardBench

This directory contains the evaluation setup for benchmarking reward models in 24 languages. The benchmark data contains prompts and chosen and rejected response pairs for each prompt. Win rate is used as an evaluation metric where reward model is supposed to assign a higher score to (or prefer) the chosen response. The data comes from the original [RewardBench paper](https://arxiv.org/abs/2403.13787) for English and from the [multilingual RewardBench paper](https://arxiv.org/abs/2410.15522) for other languages. The reward models are typically evaluated in three settings: 
1) A language model with a classification head explicitly trained as a reward model. In this case, the score assigned by the reward model is used as an explicit reward (explicit RM).
2) A language model trained with DPO for alignment. In this case, the implicit reward score can be used as a reward (implicit RM)
3) Any generative language model is prompted with both responses and asked to output its preference (generative RM).

Currently, only generative RM evaluation is supported (`generative_rm` subdirectory).

## Papers

### RewardBench: Evaluating Reward Models for Language Modeling
[Paper](https://arxiv.org/abs/2403.13787) | [Data](https://huggingface.co/datasets/allenai/reward-bench)

The RewardBench dataset is a collection of prompt-chosen-rejected trios spanning chat, reasoning, and safety, to benchmark how reward models perform on challenging, structured and out-of-distribution queries and presents many findings on propensity for refusals, reasoning limitations, and instruction following shortcomings of various reward models towards a better understanding of the RLHF process



### M-RewardBench: Evaluating Reward Models in Multilingual Settings
[Paper](https://arxiv.org/abs/2410.15522) | [Data](https://huggingface.co/datasets/CohereLabsCommunity/multilingual-reward-bench)

A systematic evaluation of several reward models in multilingual settings shows that the performance of RMs is improved with improved translation quality and it is demonstrated that the models exhibit better performance for high-resource languages.

### Tasks

* `reward_bench`: Evaluation on all languages and subtasks (Chat, Chat Hard, Safety, Reasoning). Averaging is done over tasks.
* `reward_bench_chat`: Evaluation on all languages for Chat tasks.
* `reward_bench_chat_hard`: Evaluation on all languages for Chat Hard tasks.
* `reward_bench_safety`: Evaluation on all languages for Safety tasks.
* `reward_bench_reasoning`: Evaluation on all languages for Reasoning tasks.
* `reward_bench_by_lang`: Evaluation on all languages and subtasks, but averaging is done over languages, not tasks.
* `reward_bench_{lang}`: Evaluation on all subtasks for a given language {lang}. List of languages can be found below.
* `reward_bench_{lang}_{subtask}`: Evaluation on a given {subtask} for a given language {lang}. Subtasks are `chat`, `chat_hard`, `safety` and `reasoning`.

### Languages
Arabic (arb_Arab), Czech (ces_Latn), German (deu_Latn), Greek (ell_Grek), English (en_Latn), French (fra_Latn), Hebrew (heb_Hebr), Hindi (hin_Deva), Indonesian (ind_Latn), Italian (ita_Latn), Japanese (jpn_Jpan), Korean (kor_Hang), Dutch (nld_Latn), Persian (pes_Arab), Polish (pol_Latn), Romanian (ron_Latn), Russian (rus_Cyrl), Spanish (spa_Latn), Turkish (tur_Latn), Ukranian (ukr_Cyrl), Vietnamese (vie_Latn), Chinese Simplified (zho_Hans), Chinese Traditional (zho_Hant)

### Citation
```bibtex
@article{Lambert2024RewardBenchER,
  title={RewardBench: Evaluating Reward Models for Language Modeling},
  author={Nathan Lambert and Valentina Pyatkin and Jacob Daniel Morrison and Lester James Validad Miranda and Bill Yuchen Lin and Khyathi Raghavi Chandu and Nouha Dziri and Sachin Kumar and Tom Zick and Yejin Choi and Noah A. Smith and Hanna Hajishirzi},
  journal={ArXiv},
  year={2024},
  volume={abs/2403.13787},
  url={https://api.semanticscholar.org/CorpusID:268537409}
}
```

```bibtex
@article{Gureja2024MRewardBenchER,
  title={M-RewardBench: Evaluating Reward Models in Multilingual Settings},
  author={Srishti Gureja and Lester James Validad Miranda and Shayekh Bin Islam and Rishabh Maheshwary and Drishti Sharma and Gusti Winata and Nathan Lambert and Sebastian Ruder and Sara Hooker and Marzieh Fadaee},
  journal={ArXiv},
  year={2024},
  volume={abs/2410.15522},
  url={https://api.semanticscholar.org/CorpusID:273502644}
}
```