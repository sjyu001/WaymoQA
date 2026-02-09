# WaymoQA — Dataset & Benchmark for Multi-View Safety-Critical Driving QA

**WaymoQA** is a training-enabled, **multi-view (8-camera)** driving VQA dataset designed to evaluate and improve **safety-critical reasoning** for autonomous driving. It introduces a structured **two-stage safety-critical reasoning** formulation (Stage-1 immediate risk resolution + Stage-2 induced secondary risk mitigation) and provides **objective MCQ benchmarking** alongside training splits.

> This repository is the **official** code + (annotations and/or media release per policy) for the paper:
> **WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving** 
---

## Table of Contents
- [News](#news)
- [ToDo](#ToDo)
- [Key Features](#key-features)
- [Release & License Notice](#release--license-notice)
- [Dataset Overview](#dataset-overview)
- [Getting Started](#Getting-Started)
- [Contact](#Contact)
- [Citation](#citation)
- [Acknowledgements](#Acknowlegement)

---

## News
- `2026-02-09`: Release WaymoQA v1.0.

---

## ToDo

- [x] Release question & anwswer data
- [ ] Release traind and testing code

---

## Key Features
- **Training-enabled dataset**: provides train split and objective test evaluation.
- **Multi-view**: 8 synchronized camera views per timestep/clip:
  - `FRONT_LEFT`, `FRONT`, `FRONT_RIGHT`,
  - `SIDE_LEFT`, `SIDE_RIGHT`,
  - `REAR_LEFT`, `REAR`, `REAR_RIGHT`
- **Safety-critical reasoning**: covers both normal driving reasoning and safety-critical scenarios; supports **two-stage safety reasoning**.
- **Objective evaluation**: **Multiple-Choice (MCQ)** test set to avoid LLM-as-a-judge bias.

---

## Dataset Overview
WaymoQA is constructed from long-tail scenarios in Waymo E2E and filtered using NHTSA pre-crash scenario taxonomy. The dataset includes:

- **ImageQA**: multi-view keyframes (8 views at a timestamp)
- **VideoQA**: short multi-view clips
- **Task types**: normal reasoning + safety-critical reasoning, including two-stage reasoning

**Splits**
- `train`: training-enabled (open-ended and/or MCQ depending on release)
- `test`: objective MCQ benchmark (no open-ended free-form scoring required)

> Exact counts and split composition are described in the paper and `docs/DATASET_CARD.md`.

---
# Getting Started
We have released our question-answer annotations, please download it from [HERE](https://drive.google.com/drive/folders/1RcB-RJU0Vh_Z70bkyJdU7M6tjnMu9559?usp=drive_link).

For the visual data, you can download **Waymo E2E** data from [HERE](https://waymo.com/open/download/?_gl=1*12kiif9*_ga*MTE2MTc2NjUzMy4xNzcwNjIzMDc2*_up*MQ..*_ga_KDWQB0N19R*czE3NzA2MjMwNzYkbzEkZzAkdDE3NzA2MjMwNzYkajYwJGwwJGgw).

---

## Contact
If you have any questions about the dataset, feel free to cantact me with `seungjunyu@kaist.ac.kr`.

---

## Citation
If you find our paper and project useful, please consider citing:
```bibtex

```
---

## Acknowlegement
