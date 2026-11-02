# Method and compatibility decisions

The [2026-11-02 three-method GPU diagnosis](results/dora-diagnosis-41396288500.md)
verified the pinned PEFT 0.17.1 DoRA path on Qwen2.5 sequence classification,
bitsandbytes int8 frozen base, q_proj and trainable score head. Magnitude updated
and exact saved trainable tensors reloaded with zero output difference in all
nine diagnostic runs. This evidence applies to the pinned image/hardware/config;
it is not a claim of universal PEFT or quantization compatibility.

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


## DoRA extension (separate three-method experiment)

Primary sources reviewed:
- [Liu et al., ICML 2024, sections 4.1–4.3](https://arxiv.org/html/2402.09353v2).
- [Authors' implementation and PEFT guidance](https://github.com/NVlabs/DoRA), inspected revision
  `7e2f10abbe8efe212c8fca1d983ae1d04ef13a18`, particularly
  [`commonsense_reasoning/peft/src/peft/tuners/dora.py`](https://github.com/NVlabs/DoRA/blob/7e2f10abbe8efe212c8fca1d983ae1d04ef13a18/commonsense_reasoning/peft/src/peft/tuners/dora.py).
  Its `dora_simple` path detaches the norm; its legacy int8 DoRA class raises
  `NotImplementedError`. This benchmark uses the maintained PEFT 0.17.1 int8
  implementation instead of that historical fork.
- Pinned PEFT 0.17.1: [int8 dispatch](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/lora/bnb.py),
  [DoRA variant](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/lora/variants.py),
  [magnitude and detached norm](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/lora/dora.py),
  [checkpoint conversion](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/utils/save_and_load.py).

Use stock `LoraConfig(use_dora=True)`, with the same SEQ_CLS task, q_proj r=8,
alpha=16, dropout=0 and `modules_to_save=["score"]`. No dependencies were upgraded.
The pinned int8 class dispatches to `DoraLinearVariant`; its magnitude starts
from the dequantized base-plus-adapter row norms. PEFT detaches the denominator
norm during backpropagation, as described in paper section 4.3. The frozen base
remains bitsandbytes int8; temporary dequantization for the DoRA norm is normal
DoRA overhead, not a switch to an FP16 base. No adapter merge, FSDP or offload is used.
The checkpoint serializer removes the magnitude `.weight` suffix and the loader
restores it. Validation checks the actual tensor keys/shapes and exact loaded
parameter values, including the classifier, as well as output agreement.

Source support is not CUDA evidence. The new gate uses a tiny local Qwen2
sequence classifier, fresh int8 bases, all three methods, frozen-weight equality,
optimizer coverage, finite gradients, magnitude changes and independent reload.
Each real Qwen2.5 training run also verifies all optimizer parameters/states,
frozen-base fingerprints and reloads its saved adapter into a fresh pinned int8
base. Output tolerance is atol=rtol=0.002; all trainable checkpoint tensors must
match exactly. Reload time is separately recorded, outside training/validation
and step timing; memory peaks are captured before the extra reload model.

The new configs add only method/evidence metadata. There are no changes to LR,
initialization, mean CE, joint max_norm=1 clipping, >100 alerts, rank or schedule.
The authors suggest tuning DoRA hyperparameters for best results; this experiment
instead answers the fixed-configuration comparison, not each method's tuned optimum.
A/B/head/magnitude gradients and pre/post-clip norms are separate. A zero A
gradient at B=0 is valid. Simplex importance is updated by explicit SGD outside
AdamW, included in clipping and reported separately. Magnitude uses the same
AdamW LR/decay as A/B/head; it is never silently frozen or excluded.

Methods run in fresh sequential processes on one GPU:

| Seed | First | Second | Third |
|---|---|---|---|
| 17 | LoRA | Simplex | DoRA |
| 42 | Simplex | DoRA | LoRA |
| 123 | DoRA | LoRA | Simplex |

This balances position across seeds, not every possible carry-over effect.
Manifest execution order, paired batch hashes and initial head hashes are checked.
DoRA's additional magnitude parameters, parameter bytes and AdamW state bytes
are reported, without reducing rank. Actual GPU memory includes temporary norms
and monitoring overhead. Per-step magnitude summaries compare to initialization;
initial/final values and checkpoints support inspection of its updates.

Original two-method configs and archived results remain independent. Missing
`methods` denotes the legacy LoRA/Simplex schema, never a completed DoRA matrix.
Legacy summaries regenerate byte-for-byte. The new strict schema requires nine
runs; formal acceptance additionally requires exactly 500 steps each. New reports
include all three paired contrasts and retain null time-to-target values.
The historical near-unit Simplex importance/floor-rounding limitation remains.
