# CONCIL: Continual Learning for Multimodal Concept Bottleneck Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)

**Official PyTorch implementation** of the ACM MM 2025 paper:

> **Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models**

| [**Paper (PDF)**](https://arxiv.org/pdf/2411.17471) | [**Project Page**](https://xll0328.github.io/concil/) | [**Code**](https://github.com/xll0328/CONCIL---ACM-MM-2025-BNI-Track-) |
|:---:|:---:|:---:|
| [arXiv](https://arxiv.org/pdf/2411.17471) · [Local PDF](MM25___CONCIL-6.pdf) | [xll0328.github.io/concil](https://xll0328.github.io/concil/) | [GitHub](https://github.com/xll0328/CONCIL---ACM-MM-2025-BNI-Track-) |

---

## Table of Contents

- [Overview](#overview)
- [Task: CICIL](#task-cicil)
- [Method: CONCIL](#method-concil)
- [Results](#results)
- [Environment](#environment)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training & Evaluation](#training--evaluation)
- [Metrics](#metrics)
- [Citation](#citation)
- [License](#license)

---

## Overview

**CONCIL** (**C**onceptual **C**ontinual **I**ncremental **L**earning) is the first continual learning framework for **Concept Bottleneck Models (CBMs)**. It addresses **concept-incremental and class-incremental continual learning (CICIL)**: at each phase, new classes and new concepts are introduced, and the model must retain all previously learned concepts and classes without catastrophic forgetting.

- **Gradient-free analytic updates:** Concept and decision layer updates are reformulated as linear regression with closed-form solutions, avoiding gradient-based updates that cause forgetting.
- **Recursive matrix updates:** Uses only current-phase data and previous summary matrices (e.g. $R_c$, $R_y$), so no raw historical data need be stored.
- **Absolute knowledge memory:** The recursive form is equivalent to training on all past and current data jointly.
- **Efficiency:** Lightweight recursive matrix operations, suitable for real-time and large-scale multimodal continual learning.

**Datasets:** CUB-200-2011 (200 classes, 116 concepts), Animals with Attributes (AwA, 50 classes, 85 concepts).

---

## Task: CICIL

**Concept-Incremental and Class-Incremental Continual Learning (CICIL)** for CBMs (paper Section 3, Figure 1):

<p align="center">
  <img src="figures/intro-figure.png" width="85%" alt="CICIL task: sequential tasks with growing concepts and classes" />
</p>
<p align="center"><em>Figure 1: The CICIL task. Each task introduces new classes and expands the concept set; training and test data include input $\mathbf{x}$, concept vectors $\mathbf{c}$, and labels $y$.</em></p>

- **T sequential tasks.** Each task $t$ has training set $\mathcal{D}_t^{\text{train}}$ and test set $\mathcal{D}_t^{\text{test}}$ with input $\mathbf{x}$, concept vectors $\mathbf{c}$, and labels $y$.
- **Classes** are disjoint across tasks: $Y_t \cap Y_{t'} = \emptyset$ for $t' \neq t$.
- **Concepts** are cumulative: $\mathcal{C}^{\text{concepts}}_{\leq t} = \mathcal{C}^{\text{concepts}}_{\leq t-1} \cup \mathcal{C}^{\text{concepts}}_t$.
- **Access:** Only current task data and previous-phase parameters are available.
- **Objectives:** (i) **Stability** — retain accuracy on previous concepts/classes; (ii) **Plasticity** — learn new concepts and classes.

The CBM has a concept extractor $g: \mathbf{x} \mapsto \mathbf{z} \mapsto \mathbf{c}$ and a classifier $f: \mathbf{c} \mapsto y$. Both concept dimension $L_{\leq t}$ and class space $|Y_{\leq t}|$ grow over phases.

---

## Method: CONCIL

CONCIL has two stages (paper Figure 2):

<p align="center">
  <img src="figures/framework.png" width="95%" alt="CONCIL framework: base training and incremental analytic updates" />
</p>
<p align="center"><em>Figure 2: CONCIL framework. Base training (Task 0) jointly trains backbone, concept layer, and classifier; backbone is then frozen. Incremental tasks use recursive analytic updates for concept layer ($W_c$) and classifier ($W_y$) with expanding concept/class dimensions.</em></p>

1. **Base training (phase 0):** Jointly train backbone $g_1$, concept layer $g_2$, and classifier $f$; then **freeze the backbone**.
2. **Feature expansion:** Map $\mathbf{z}$ to $\mathbf{z}^* = \sigma(\mathbf{z} W_{fe})$ (and similarly for concept→class). This increases capacity for the analytic layers.
3. **Incremental phases ($t \geq 1$):**
   - **Concept layer:** Regularized linear regression from $\mathbf{z}^*$ to $\mathbf{c}$; recursive update via Sherman–Morrison–Woodbury using current-phase data and $R_c^{(t-1)}$. Update $W_c^{(t)}$ with columns for old and new concepts.
   - **Classifier:** Linear regression from expanded concepts to labels; recursive update for $R_y^{(t)}$ and $W_y^{(t)}$ for old and new classes.

**Hyperparameters (paper Section 5.4):** $\lambda_1 = 500$, $\lambda_2 = 1$, $d_{z^*} = 25000$, $d_{\hat{C}^*} = 25000$. Phase split: initial phase uses first 50% of classes and 50% of concepts; each subsequent phase adds a fraction of the remainder (e.g. $(1-n\%)/(p-1)$ of classes and $(1-m\%)/(p-1)$ of concepts for $p$ phases).

---

## Results

**Table 1** in the paper reports CONCIL vs. baseline on CUB and AwA (concept/class accuracy and forgetting rates across phases). Main result figure (accuracy and forgetting curves):

<p align="center">
  <img src="figures/incremental_learning_results.png" width="90%" alt="CONCIL vs baseline: accuracy and forgetting rates" />
</p>
<p align="center"><em>CONCIL vs. baseline: average concept/class accuracy (top) and average concept/class forgetting rate (bottom) across phases on CUB and AwA.</em></p>

Additional per-phase visualizations: see `VISUAL/` (e.g. `CONCIL-CUB-phase_*.png`, `Baseline-AwA-phase_*.png`). Notebooks in `VISUAL/` can be used to reproduce or extend plots.

---

## Environment

**Requirements:** Python 3.8+, PyTorch, torchvision, transformers, CUDA (recommended).

```bash
conda create -n concil python=3.8
conda activate concil
pip install -r requirements.txt
```

Run all commands from the **repository root**.

---

## Project Structure

```
├── README.md
├── MM25___CONCIL-6.pdf          # Paper (local)
├── figures/                     # Paper figures (for README)
│   ├── intro-figure.png        # Task definition (Fig. 1)
│   ├── framework.png          # Method (Fig. 2)
│   └── incremental_learning_results.png
├── requirements.txt
├── run_concil_example.sh
├── src/
│   ├── utils/                  # data_path.example.yml, config
│   ├── experiments/            # CONCIL_1114.py (main), cl_baseline.py
│   ├── analytic/               # RecursiveLinear, Buffer, Learner
│   ├── data/, models/, processing/
├── command/                     # CONCIL_*.sh, processing.sh
└── VISUAL/                      # Result PNGs, notebooks
```

---

## Dataset Preparation

1. **Download:** [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) → `source_data/CUB_200_2011`; [AwA2](https://www.mohamedaly.info/datasets/awa2) → `source_data/Animals_with_Attributes2`.

2. **Preprocess** (from repo root):

```bash
# CUB
python src/processing/cub_data_processing.py \
  -save_dir processed_data/cub_processed_data \
  -data_dir source_data/CUB_200_2011

# AwA
python src/processing/awa_data_processing.py \
  -save_dir processed_data/awa_processed_data \
  -data_dir source_data/Animals_with_Attributes2
```

This produces `train.pkl`, `test.pkl`, and can update `src/utils/data_path.yml`.

---

## Configuration

Paths are read from **`src/utils/data_path.yml`** (gitignored). Copy the example and set your paths:

```bash
cp src/utils/data_path.example.yml src/utils/data_path.yml
```

Edit `processed_dir` and `source_dir` for `cub` and `awa`.

---

## Training & Evaluation

**Single run (CUB, 8 phases):**

```bash
./run_concil_example.sh
# or:
python src/experiments/CONCIL_1114.py \
  -dataset cub -num_stages 8 -buffer_size 25000 -saved_dir results/concil_cub
```

**Main arguments:**

| Argument        | Default | Description                    |
|----------------|---------|--------------------------------|
| `-dataset`     | cub     | `cub` or `awa`                 |
| `-num_stages`  | 8       | Number of incremental phases   |
| `-buffer_size` | 25000   | Replay buffer size ($d_{z^*}$) |
| `-gg1`         | 500     | Concept regularization ($\lambda_1$) |
| `-gg2`         | 1       | Class regularization ($\lambda_2$)   |
| `-saved_dir`   | —       | Logs and results               |

**Batch experiments:**

```bash
bash command/CONCIL_cub_exp.sh   # CUB, stages 2–10
bash command/CONCIL_awa_exp.sh   # AwA
bash command/CONCIL_tc_11_14.sh  # Hyperparameter sweep
```

**Baseline:**

```bash
python src/experiments/cl_baseline.py -dataset cub -num_stages 8 -saved_dir results/baseline_cub
```

---

## Metrics

- **Average concept accuracy** $A_{\text{concept}}(t)$: mean concept accuracy over tasks 1…t.
- **Average class accuracy** $A_{\text{class}}(t)$: mean class accuracy over tasks 1…t.
- **Average concept forgetting rate** $F_{\text{concept}}(t)$: mean drop in concept accuracy on previous tasks vs. their best.
- **Average class forgetting rate** $F_{\text{class}}(t)$: mean drop in class accuracy on previous tasks vs. their best.

Higher accuracy and lower forgetting rates are better.

---

## Citation

```bibtex
@inproceedings{lai2025learning,
  title={Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models},
  author={Lai, Songning and Liao, Mingqian and Hu, Zhangyi and Yang, Jiayu and Chen, Wenshuo and Xiao, Hongru and Tang, Jianheng and Liao, Haicheng and Yue, Yutao},
  booktitle={Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year={2025}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
