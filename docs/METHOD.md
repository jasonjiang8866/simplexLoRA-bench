# Method and compatibility decisions

Primary references:

- [Paper, section 3.1 and appendix B](https://arxiv.org/html/2610.37206v1)
- [Official repository, inspected commit 8233d7ba77b4d34da52dfb973d88357beb96cbda](https://github.com/brain-lab-research/SimplexLoRA/tree/8233d7ba77b4d34da52dfb973d88357beb96cbda)
- Relevant upstream paths: `src/optimizers.py`,
  `peft/src/peft/tuners/lora/layer.py`, `peft/src/peft/tuners/lora/bnb.py`.

The paper describes short joint training of A/B and scalar importance weights,
Euclidean projection onto the n-scaled simplex, floor(r0 * weight) rank
assignment, and standard LoRA reinitialization at rank updates. After K
updates, importance is reset to one and frozen; final training uses the
resulting rank profile. This repository implements that sequence independently.

The inspected code and paper differ:

| Decision | Paper / inspected implementation | This benchmark |
|---|---|---|
| Projection | Paper: projection directly onto mass n; code: project to mass 1, multiply by n | Direct Euclidean projection to mass n |
| Integer ranks | Paper: floor; code: ceil | Floor, and assert parameter budget is not exceeded |
| Resizing | Current active code reinitializes A/B; QR/SVD path is commented out | Reinitialize A/B; do not claim weight-preserving QR/SVD |
| Zero rank | Code retires adapters from chosen set | Permanently remove zero-rank adapter, persist target metadata |
| Optimizer | Custom AdamW and SGD importance update | PyTorch AdamW for A/B+head; explicit regularized SGD for importance |
| Optimizer state | New A/B get fresh state | Reset replaced A/B state; preserve classifier moments |
| Scheduler | Two phases with reduced search LR and fresh warmup | Search LR=0.1×base, then fresh linear warmup/decay |

This is therefore a **paper-based stock-PEFT port**, not bitwise execution of
the official fork. Gradient clipping at 1.0 applies to the joint trainable
parameters in both methods (including importance during search).

Budget fairness: a rank budget only guarantees a parameter budget if adapter
matrix dimensions have equal cost per rank. Qwen's GQA q/v projections differ,
so default targets are q_proj only. Unequal target costs are rejected rather
than presented as budget matched. Floor rounding may leave unused capacity;
every allocation logs actual parameter count and the r=8 budget. Temporary
importance scalars and transient allocation memory are overhead and are not
hidden. Rank growth may be overcomplete if concentrated on a small layer.

The search uses only training labels, never validation gradients. Validation
sets the time-to-target milestone, and the held-out test split is evaluated
only once at the fixed final budget. Both methods train for the same total
number of steps, so Simplex's search is charged to its budget. The classification
head remains trained through rank search (matching the upstream non-LoRA
parameter behavior); it is not reinitialized at the final rank allocation.

Only one active default adapter is supported. Distributed training, adapter
merging into quantized weights, checkpoint continuation within a run, dropout
ablation, alternative rank-budget apportionment, and mixed target dimensions
are outside this initial experiment.
