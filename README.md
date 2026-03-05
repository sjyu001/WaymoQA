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

![teaser](./assets/teaser.jpg)

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

**Data Preprocessing Environment**
```
conda env create -f environment.yml
conda activate data
python -m pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
```

**Training Environment (Third Party: [Fine-tuning Qwen-VL Series](https://github.com/2U1/Qwen-VL-Series-Finetune))**
```bash
cd Qwen-VL-Series-Finetune
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
pip install vllm pandas
```

### 1. Download Dataset
We have released our question-answer annotations, please download it from [HERE](https://drive.google.com/drive/folders/1RcB-RJU0Vh_Z70bkyJdU7M6tjnMu9559?usp=drive_link).

For the visual data, you can download **Waymo E2E** data from [HERE](https://waymo.com/open/download/?_gl=1*12kiif9*_ga*MTE2MTc2NjUzMy4xNzcwNjIzMDc2*_up*MQ..*_ga_KDWQB0N19R*czE3NzA2MjMwNzYkbzEkZzAkdDE3NzA2MjMwNzYkajYwJGwwJGgw).

### 2. Data Preprocessing
#### 2.1 Extract Images from TFRecord
Directly reading TFRecords on-the-fly during benchmarking/training can be very time-consuming due to repeated TFRecord scanning and image decoding.  
To improve efficiency, we preprocess the dataset by exporting 8-camera images in advance and use these preprocessed files for faster benchmarking and training.

```bash
# Example for Image QA
python dataset/extract_for_imageqa.py \
  --split train \
  --target-jsonl ./dataset/question/train.jsonl \
  --output-dir ./dataset/train_imgs

# Example for Video QA
python dataset/export_videoqa_mosaics.py \
  --split train \
  --target-jsonl ./dataset/question/train.jsonl \
  --output-dir ./dataset/train_imgs \
  --gap 12 \
  --tile-align center
```
#### 2.2 Build LLaVA-style conversation JSON for Training
For Qwen fine-tuning, we convert WaymoQA into a Qwen-compatible training format (e.g., LLaVA-style `conversations` with aligned multi-view image/video inputs).
> If you plan to use the validation split for training, we recommend using ```validation_open_ended.jsonl``` rather than ```validation_mcq.jsonl```.
> The MCQ version is intended for benchmarking/evaluation (fixed answer choices), while the open-ended version better matches training objectives and avoids overfitting to the provided options.

```bash
python dataset/build_llava_conversations.py \
  --inputs ./dataset/question/train.jsonl \
  --out ./dataset/llava_train.json \
  --video-mosaic-dir ./dataset/train_imgs \
  --video-stride 5 \ # Adjust based on your GPU memory
  --video-max-frames -1
```

#### 2.3 Folder Structure
The folder structure should be organized as follows before training.
```bash
WaymoQA
+-- dataset/
|   +-- questions/				# downloaded
|   |   +-- train.jsonl
|   |   +-- validation.jsonl
|   |   +-- test.jsonl
|   +-- waymoe2e/ 				# downloaded or extracted
|   |   +-- train.tfrecord
|   |   +-- validation.tfrecord
|   |   +-- test.tfrecord
|   |   +-- train_imgs/
|   |   |   +-- xxx.jpg
|   |   |   +-- ...
|   |   +-- validation_imgs/
|   |   |   +-- xxx.jpg
|   |   |   +-- ...
|   |   +-- test_imgs/
|   |   |   +-- xxx.jpg
|   |   |   +-- ...
+-- scripts/
+-- Qwen-VL-Series-Finetune/
```

### 3. Evaluation and Training
#### 3.1 Evaluation
```bash
bash scripts/service.sh


```

#### 3.2 Training
```bash
bash 
```
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

