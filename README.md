<div align="center">

# NFL: No Forgetting Learning

### A buffer-free approach to continual learning that actually works.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

![NFL Continual Learning Diagram](NFL.png)


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


## The PS Metric

We also propose a **Plasticity-Stability score** — the harmonic mean of how well the model learns new tasks (plasticity) and how well it retains old ones (stability):

```
PS = 2·P·S / (P+S)

P = (1/(T-1)) Σ (A_{k,k} - A_{k-1,k}) / (1 - A_{k-1,k})    ← learning efficiency
S = 1 - (1/(T-1)) Σ (A_{k,k} - A_{T,k})                      ← retention rate
```

ACC tells you the average performance. BWT tells you forgetting. PS tells you the *balance* — which is what actually matters for deployment.

---

## Citation

```bibtex
@misc{vahedifar2025forgettinglearningmemoryfreecontinual,
      title={No Forgetting Learning: Memory-free Continual Learning}, 
      author={Mohammad Ali Vahedifar and Qi Zhang},
      year={2025},
      eprint={2503.04638},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.04638}, 
}
```

---

<div align="center">

**Questions? Issues? Open a GitHub issue.**

Built with PyTorch. Tested on NVIDIA A6000.

</div>
