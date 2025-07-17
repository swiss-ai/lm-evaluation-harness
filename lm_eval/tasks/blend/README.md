# BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages

## Overview

This is the official repository of BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages (Submitted to NeurIPS 2024 Datasets and Benchmarks Track).

Large language models (LLMs) often lack culture-specific everyday knowledge, especially across diverse regions and non-English languages. Existing benchmarks for evaluating LLMs' cultural sensitivities are usually limited to a single language or online sources like Wikipedia, which may not reflect the daily habits, customs, and lifestyles of different regions. That is, information about the food people eat for their birthday celebrations, spices they typically use, musical instruments youngsters play, or the sports they practice in school is not always explicitly written online. To address this issue, we introduce BLEnD, a hand-crafted benchmark designed to evaluate LLMs' everyday knowledge across diverse cultures and languages. The benchmark comprises 52.6k question-answer pairs from 16 countries/regions, in 13 different languages, including low-resource ones such as Amharic, Assamese, Azerbaijani, Hausa, and Sundanese. We evaluate LLMs in two formats: short-answer questions, and multiple-choice questions. We show that LLMs perform better in cultures that are more present online, with a maximum 57.34% difference in GPT-4, the best-performing model, in the short-answer format. Furthermore, we find that LLMs perform better in their local languages for mid-to-high-resource languages. Interestingly, for languages deemed to be low-resource, LLMs provide better answers in English.


## Dataset Information

- **Dataset**: [nayeon212/BLEnD](https://huggingface.co/datasets/nayeon212/BLEnD)
- **Paper**: [BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages](https://arxiv.org/abs/2406.09948)

## Countries/Regions Included
- Algeria
- Assam
- Azerbaijan
- China
- Ethiopia
- Greece
- Indonesia
- Iran
- Mexico
- North Korea
- Northern Nigeria
- South Korea
- Spain
- UK
- US
- West Java

## Task Structure

### Main Group Task
- `blend.yaml`: Main group that aggregates all country-specific tasks

### Individual Country Tasks
Each country/region has its own task file that filters the dataset:
- `blend_algeria.yaml`
- `blend_assam.yaml`
- `blend_azerbaijan.yaml`
- `blend_china.yaml`
- `blend_ethiopia.yaml`
- `blend_greece.yaml`
- `blend_indonesia.yaml`
- `blend_iran.yaml`
- `blend_mexico.yaml`
- `blend_north_korea.yaml`
- `blend_northern_nigeria.yaml`
- `blend_south_korea.yaml`
- `blend_spain.yaml`
- `blend_uk.yaml`
- `blend_us.yaml`
- `blend_west_java.yaml`

## Usage

### Evaluate all countries
```bash
lm_eval --model hf --model_args pretrained=your_model --tasks blend
```

### Evaluate specific countries
```bash
lm_eval --model hf --model_args pretrained=your_model --tasks blend_uk,blend_us,blend_china
```

### Evaluate a single country
```bash
lm_eval --model hf --model_args pretrained=your_model --tasks blend_ethiopia
```

## Citation

If you use this implementation, please cite:

```
@misc{myung2025blendbenchmarkllmseveryday,
      title={BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages}, 
      author={Junho Myung and Nayeon Lee and Yi Zhou and Jiho Jin and Rifki Afina Putri and Dimosthenis Antypas and Hsuvas Borkakoty and Eunsu Kim and Carla Perez-Almendros and Abinew Ali Ayele and Víctor Gutiérrez-Basulto and Yazmín Ibáñez-García and Hwaran Lee and Shamsuddeen Hassan Muhammad and Kiwoong Park and Anar Sabuhi Rzayev and Nina White and Seid Muhie Yimam and Mohammad Taher Pilehvar and Nedjma Ousidhoum and Jose Camacho-Collados and Alice Oh},
      year={2025},
      eprint={2406.09948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.09948}, 
}
``` 