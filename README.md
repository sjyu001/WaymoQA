# WaymoQA — Dataset & Benchmark for Multi-View Safety-Critical Driving QA

**WaymoQA** is a training-enabled, **multi-view (8-camera)** driving VQA dataset designed to evaluate and improve **safety-critical reasoning** for autonomous driving. It introduces a structured **two-stage safety-critical reasoning** formulation (Stage-1 immediate risk resolution + Stage-2 induced secondary risk mitigation) and provides **objective MCQ benchmarking** alongside training splits.

> This repository is the **official** code for the paper:
> **WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving**
>  
>[![arXiv](https://img.shields.io/badge/arXiv-2511.20022-b31b1b.svg)](https://arxiv.org/pdf/2511.20022)
 

### Contribution
- **Task definition.** We define a new task of safety-critical reasoning under multi-view inputs within vision and language understanding for autonomous driving.
- **Dataset.** We present WaymoQA, the first safety-critical and multi-view driving QA dataset with a training split, enabling research beyond evaluation only.
- **Analysis and findings.** Through systematic analyses and experiments, we characterize MLLMs’ safety-critical understanding and show that training data markedly improve this capability.

![MainFigure](./assets/teaser.jpg)

### Key Features
- **Training-enabled dataset**: provides train split and objective test evaluation.
- **Multi-view**: 8 synchronized camera views per timestep/clip:
  - `FRONT_LEFT`, `FRONT`, `FRONT_RIGHT`,
  - `SIDE_LEFT`, `SIDE_RIGHT`,
  - `REAR_LEFT`, `REAR`, `REAR_RIGHT`
- **Safety-critical reasoning**: covers both normal driving reasoning and safety-critical scenarios; supports **two-stage safety reasoning**.
- **Objective evaluation**: **Multiple-Choice (MCQ)** test set to avoid LLM-as-a-judge bias.
- **Splits**
  - `train`: training-enabled (open-ended)
  - `validation`: training-enabled and evaluation-enabled (open-ended and/or MCQ depending on release)
  - `test`: objective MCQ benchmark (no open-ended free-form scoring required)
  > Exact counts and split composition are described in the paper.

---


## 🔥 News
- `2026-02-09`: Release WaymoQA v1.0.
- `2026-03-05`: Release Official Code.

---

## ⌛ ToDo

- [x] Release question & anwswer data
- [x] Release traind and testing code

---
# 🛠️ Getting Started
### 0. Installation
**Environments**
- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.8

**Using Third Party ([Fine-tuning Qwen-VL Series](https://github.com/2U1/Qwen-VL-Series-Finetune))**
```bash
```

### 1. Download Dataset
We have released our question-answer annotations, please download it from [HERE](https://drive.google.com/drive/folders/1RcB-RJU0Vh_Z70bkyJdU7M6tjnMu9559?usp=drive_link).

For the visual data, you can download **Waymo E2E** data from [HERE](https://waymo.com/open/download/?_gl=1*12kiif9*_ga*MTE2MTc2NjUzMy4xNzcwNjIzMDc2*_up*MQ..*_ga_KDWQB0N19R*czE3NzA2MjMwNzYkbzEkZzAkdDE3NzA2MjMwNzYkajYwJGwwJGgw).

---

## 📧 Contact
If you have any questions about the dataset, feel free to cantact me with `seungjunyu@kaist.ac.kr`.

---

## 📄 Citation
If you find our paper and project useful, please consider citing:
```bibtex
@article{yu2025waymoqa,
  title={WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving},
  author={Yu, Seungjun and Lee, Seonho and Kim, Namho and Shin, Jaeyo and Park, Junsung and Ryu, Wonjeong and Jung, Raehyuk and Shim, Hyunjung},
  journal={arXiv preprint arXiv:2511.20022},
  year={2025}
}
```
