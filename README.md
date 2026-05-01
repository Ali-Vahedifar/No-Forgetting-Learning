<div align="center">

# NFL: No Forgetting Learning

### A buffer-free approach to continual learning that actually works.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Paper (NeurIPS 2026)](#) · [Results](#key-results) · [Quick Start](#getting-started) · [Reproduce Everything](#reproduce-the-paper)

</div>

---

## What's this about?

Most continual learning methods cheat a little — they keep a buffer of old examples around to remind the model what it learned before. That works, but it doesn't scale (memory grows with every task), and it's a non-starter when you can't store old data (think GDPR, medical records, anything privacy-sensitive).

We asked: **what if the network already has everything it needs?**

Overparameterized networks are full of redundancy. Instead of pruning it away, we repurpose it. NFL freezes and unfreezes different parts of the network in a specific sequence — isolating new learning, adapting shared features under distillation, then consolidating everything. No replay buffer. No growing memory. Just the network itself.

**Three variants, one idea:**

| | Backbone | What it adds |
|---|---|---|
| **NFL** | ResNet-18 | Stepwise freezing + dual knowledge distillation |
| **NFL+** | ResNet-18 | + Auto-Encoder for feature preservation + bias correction |
| **NFL+LoRA** | ViT-B/16 | + Low-rank adaptation + Fisher regularization |

---

## Key Results

NFL+ uses **2.53%** of the buffer storage that replay methods need, and still closes most of the performance gap.

**ImageNet-1000, 10 tasks (CIL):**

| Method | ACC (%) | Buffer? |
|---|---|---|
| DyTox | 40.15 | 20K exemplars |
| MEMO | 38.90 | 20K exemplars |
| **NFL+ (ours)** | **38.42** | **None** |
| DCNet | 37.80 | None |
| LwF | 11.24 | None |

**ViT-B/16 on ImageNet-A, 20 tasks (CIL):**

| Method | ACC (%) | BWT |
|---|---|---|
| **NFL+LoRA (ours)** | **59.10** | **−0.08** |
| EWC-LoRA | 55.20 | −0.46 |
| CL-LoRA | 52.80 | −0.55 |

Full tables with all datasets, task counts, and metrics are in the paper.

---

## Getting Started

```bash
git clone https://github.com/anonymous/nfl-continual-learning.git
cd nfl-continual-learning
pip install -r requirements.txt
```

That's it. CIFAR-100 downloads automatically. For ImageNet variants, see [Data Setup](#data-setup).

### Train NFL+ on CIFAR-100

```bash
python train.py --method nfl+ --dataset cifar100 --num_tasks 10 --scenario cil
```

### Train NFL+LoRA on ImageNet-R

```bash
python train.py --method nfl+lora --dataset imagenet-r --num_tasks 10 --scenario cil
```

### Compare against baselines

```bash
python train.py --method ewc --dataset cifar100 --num_tasks 10 --scenario cil
python train.py --method der++ --dataset cifar100 --num_tasks 10 --scenario cil --buffer_size 2000
python train.py --method lwf --dataset tinyimagenet --num_tasks 20 --scenario til
```

Available methods: `nfl`, `nfl+`, `nfl+lora`, `ewc`, `si`, `lwf`, `der++`, `sgd`

---

## How the Pipeline Works

The core idea is **stepwise freezing** — training different network components in a deliberate sequence. We systematically analyzed all 8 possible freeze/train configurations and found only 3 are useful:

```
Step 1:  Train on current task normally
Step 2:  [FFT] Freeze backbone + old head → train only the new head
Step 3:  [TTF] Freeze new head → adapt backbone under distillation
Step 4:  [TTT] Unfreeze everything → consolidate with dual soft targets
```

The ordering matters. Starting with joint training (Step 4 first) lets the new task dominate. Isolating the new head first (Step 2) prevents gradient interference from corrupting learned features.

**NFL+** adds an Auto-Encoder after Step 1 to compress the old-task feature manifold, then uses it for bias correction in the final step.

**NFL+LoRA** keeps the same 4-step logic but confines all ViT updates to low-rank matrices (A, B). After each task: estimate Fisher → accumulate → merge LoRA into base weights → reinitialize. Memory stays constant no matter how many tasks you train.

---

## Reproduce the Paper

Every number in every table:

```bash
# Run everything (our methods + baselines, all datasets, 10 seeds each)
python run_experiments.py --all --n_runs 10

# Just our methods
python run_experiments.py --ours-only --n_runs 10

# Just ViT experiments
python run_experiments.py --vit-only --n_runs 10

# One specific config
python run_experiments.py --method nfl+ --dataset imagenet1000 \
    --num_tasks 50 --scenario cil --n_runs 10
```

This covers:
- **ResNet-18:** CIFAR-100, Tiny-ImageNet (10/20 tasks), ImageNet-1000 (10/20/50 tasks) × CIL & TIL
- **ViT-B/16:** CIFAR-100, ImageNet-R, ImageNet-A (10/20 tasks) × CIL
- **Baselines:** EWC, SI, LwF, DER++, SGD lower bound

Results are saved as JSON with accuracy matrices, all metrics, and hyperparameters. Aggregated mean ± std across seeds printed at the end.

> External baselines (DCNet, NISPA, SpaceNet, PEC, DyTox, MEMO, iCaRL, CL-LoRA, EWC-LoRA) need their own repos. Links are in `models/baselines.py`.

---

## Data Setup

| Dataset | Size | Setup |
|---|---|---|
| CIFAR-100 | 32×32, 100 classes | Auto-downloads |
| Tiny-ImageNet | 64×64, 200 classes | `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip` → `./data/` |
| ImageNet-1000 | 224×224, 1K classes | Standard ILSVRC2012 → `./data/imagenet/` |
| ImageNet-R | 224×224, 200 classes | [Download](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar) → `./data/imagenet-r/` |
| ImageNet-A | 224×224, 200 classes | [Download](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar) → `./data/imagenet-a/` |

---

## Project Layout

```
├── models/
│   ├── nfl.py            # NFL, NFL+, NFL+LoRA — models and trainers
│   ├── backbone.py       # ResNet-18/34/50
│   ├── vit_lora.py       # ViT-B/16 + LoRA + Fisher
│   └── baselines.py      # EWC, SI, LwF, DER++, SGD
├── data/
│   └── datasets.py       # All 5 datasets with task splitting
├── utils/
│   └── metrics.py        # ACC, BWT, PS (our proposed metric)
├── train.py              # Single entry point for all methods
├── run_experiments.py    # Reproduce every table
└── test_nfl.py           # 27 unit tests (all passing)
```

---

## Hyperparameters

All hyperparameters are selected using only the first task's validation split — never re-tuned on later data. This follows the first-task HPO protocol from [Cha & Cho (TMLR 2025)](https://arxiv.org/abs/2501.xxxxx).

| Method | Dataset | lr | temp | α/η | Ω | λ | rank |
|---|---|---|---|---|---|---|---|
| NFL | CIFAR-100 | 0.1 | 2.0 | 0.3 | — | — | — |
| NFL | Tiny/IN-1K | 0.03 | 2.0 | 0.3–0.5 | — | — | — |
| NFL+ | CIFAR-100 | 0.1 | 2.0 | 0.5 | 0.5 | — | — |
| NFL+ | Tiny/IN-1K | 0.03 | 2.0 | 0.5 | 0.5 | — | — |
| NFL+LoRA | All | 0.001 | 2.0 | 0.5 | — | 1.0 | 8 |

Training: Adam optimizer, batch size 64, early stopping (patience 10), 100 epochs (200 for ImageNet-1000, 50 for ViT), single NVIDIA A6000.

---

## The PS Metric

We also propose a **Plasticity-Stability score** — the harmonic mean of how well the model learns new tasks (plasticity) and how well it retains old ones (stability):

```
PS = 2·P·S / (P+S)

P = (1/(T-1)) Σ (A_{k,k} - A_{k-1,k}) / (1 - A_{k-1,k})    ← learning efficiency
S = 1 - (1/(T-1)) Σ (A_{k,k} - A_{T,k})                      ← retention rate
```

ACC tells you the average performance. BWT tells you forgetting. PS tells you the *balance* — which is what actually matters for deployment.

---

## Tests

```bash
python test_nfl.py
```

27 tests covering: KD loss, Auto-Encoder, multi-head classifier, all three model variants (NFL/NFL+/NFL+LoRA), LoRA merge-and-reset correctness, Fisher penalty, backbone variants, metrics computation, and the metrics tracker.

---

## Limitations

Honest about what doesn't work perfectly:

- **KD degrades under distribution shift.** When tasks are very different, soft targets become unreliable. This is inherent to all distillation-based CL, but it hits harder without a buffer to recalibrate.
- **The multi-step pipeline is slower.** 3-4× training time per task versus single-pass methods. See the computational cost analysis in the paper.
- **Auto-Encoder bottleneck dimension.** We found 128 (feature_dim/4) robust across all benchmarks, but very different feature complexities might need tuning.
- **Bias correction needs a held-out set.** A mild assumption beyond strict buffer-free operation.

---

## Citation

```bibtex
@inproceedings{anonymous2026nfl,
  title={No Forgetting Learning: Buffer-Free Continual Learning Classification},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

---

<div align="center">

**Questions? Issues? Open a GitHub issue.**

Built with PyTorch. Tested on NVIDIA A6000.

</div>
