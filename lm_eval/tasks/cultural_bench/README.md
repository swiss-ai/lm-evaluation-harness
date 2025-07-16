# CulturalBench: A Benchmark for Evaluating Cultural Awareness in Large Language Models

## Overview

CulturalBench is a set of 1,227 human-written and human-verified questions for effectively assessing LLMs’ cultural knowledge, covering 45 global regions including the underrepresented ones like Bangladesh, Zimbabwe, and Peru.

We evaluate models on two setups: CulturalBench-Easy and CulturalBench-Hard which share the same questions but asked differently.

- CulturalBench-Easy: multiple-choice questions (Output: one out of four options i.e. A,B,C,D). Evaluate model accuracy at question level (i.e. per question_idx). There are 1,227 questions in total.
- CulturalBench-Hard: binary (Output: one out of two possibilties i.e. True/False). Evaluate model accuracy at question level (i.e. per question_idx). There are 1,227x4=4908 binary judgements in total with 1,227 questions provided.

## Dataset Information

- **Dataset**: [kellycyy/CulturalBench](https://huggingface.co/datasets/kellycyy/CulturalBench)
- **Paper**: [CulturalBench: a Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of Large Language Models](https://arxiv.org/pdf/2410.02677)

## Countries/Regions Included

Argentina, Australia, Bangladesh, Brazil, Canada, Chile, China, Czech Republic, Egypt, France, Germany, Hong Kong, India, Indonesia, Iran, Israel, Italy, Japan, Lebanon, Malaysia, Mexico, Morocco, Nepal, Netherlands, New Zealand, Nigeria, Pakistan, Peru, Philippines, Poland, Romania, Russia, Saudi Arabia, Singapore, South Africa, South Korea, Spain, Taiwan, Thailand, Turkey, Ukraine, United Kingdom, United States, Vietnam, Zimbabwe

## Task Structure

### Main Group Tasks
- `cultural_bench`: Overall group combining both easy and hard variants
- `cultural_bench_easy`: Group of all easy (multiple choice) country tasks  
- `cultural_bench_hard`: Group of all hard (true/false) country tasks

### Individual Country Tasks

#### Easy Tasks (Multiple Choice with 4 options)
Each country has an easy task with format `cultural_bench_easy_{country}`: `cultural_bench_easy_argentina`, `cultural_bench_easy_australia`, etc.

#### Hard Tasks (True/False)  
Each country has a hard task with format `cultural_bench_hard_{country}`: `cultural_bench_hard_argentina`, `cultural_bench_hard_australia`, etc.

## Usage

### Evaluate all countries (both easy and hard)
```bash
lm_eval --model hf --model_args pretrained=your_model --tasks cultural_bench
```

### Evaluate only easy tasks
```bash
lm_eval --model hf --model_args pretrained=your_model --tasks cultural_bench_easy
```

### Evaluate only hard tasks
```bash
lm_eval --model hf --model_args pretrained=your_model --tasks cultural_bench_hard
```

### Evaluate specific countries
```bash
lm_eval --model hf --model_args pretrained=your_model --tasks cultural_bench_easy_united_states,cultural_bench_hard_united_states
```


## Citation

```
@misc{chiu2024culturalbenchrobustdiversechallenging,
      title={CulturalBench: a Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs}, 
      author={Yu Ying Chiu and Liwei Jiang and Bill Yuchen Lin and Chan Young Park and Shuyue Stella Li and Sahithya Ravi and Mehar Bhatia and Maria Antoniak and Yulia Tsvetkov and Vered Shwartz and Yejin Choi},
      year={2024},
      eprint={2410.02677},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02677}, 
}
``` 