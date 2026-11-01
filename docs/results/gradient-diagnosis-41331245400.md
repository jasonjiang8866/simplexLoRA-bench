# Gradient diagnosis — 2026-11-02 SGT

**Decision: proceed with the specified 500-step benchmark, without changing
initialization, learning rates, loss, clipping or the >100 flag.** Large norms
are dominated by the classification head's feature scale, not evidence of an
exploding adapter gradient or an identified implementation error. This is not
a claim that the joint clipping/search configuration is optimal.

[Raw Actions artifact and successful run](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41331245400)
· [machine-readable analysis](gradient-diagnosis-41331245400.json)
· [reproducible analyzer](../../runpod/report_experiment.py).

## Verified observations

All six runs completed 50 steps, with the full 8192/1024/2048 subsets, batch 8,
length 128, the first 50 learning rates of the 500-step schedule, and rank
updates at 10/20/30. Model/data revisions match the original smoke. Independent
analysis of all **300** steps found:

| Check | Result |
|---|---|
| Missing or NaN/Inf gradients | 0/300 |
| Pre-clipping norm range | 38.9447–328.3361 |
| Norm >100, LoRA / Simplex | 86/150 / 125/150; flags retained |
| Mean head share of squared gradient norm, LoRA / Simplex | 99.7733% / 99.9153% |
| Largest post-clipping norm | 1.000000188 (FP32 rounding) |
| Largest analytic head-gradient relative error | 1.641e-7 |
| Mean pooled-hidden norm per batch, range over steps | 281.62–308.13 |
| Head initialization standard deviation, across seeds | 0.01986–0.02018 |
| Paired batch hashes, all 50 positions for all seeds | identical |
| Paired initial head hashes, loss and total norm | identical |

The head-gradient observer reconstructs
`(softmax(logits) - onehot(labels)).T @ pooled_hidden / batch_size`.
Agreement with autograd, stable hidden-feature norms and declining mean losses
support the feature/head-scale explanation. It also verifies last-nonpadding
pooling against the attention mask, reconstructed logits and batch-mean CE.
The initial head standard deviation near 0.02 multiplied by hidden-vector
norm near 290 implies an approximately 5.8 logit standard deviation under
independent random head entries. Observed first-batch logit standard deviations
are 4.534, 6.460 and 4.142 for seeds 17/42/123, with losses 4.472, 9.048 and
4.927. This explains large initial CE/head derivatives without a sum-reduced
loss or padding error. It is a configuration-scale diagnosis, not evidence that
this initialization and joint clipping are optimal for rank-search learning.
All live optimizer parameters are checked after each step. Rank updates retained
the classifier optimizer state and removed replaced adapter state. A-gradient
zeros when B is initialized to zero remain expected diagnostics.

Mean loss in the first versus final ten steps decreased in all six runs:
LoRA seeds 17/42/123: 3.345→1.293, 8.364→1.277, 5.722→1.427;
Simplex: 3.362→2.656, 8.535→4.669, 5.786→3.467.
These 50-step values do not establish final method superiority.

Post-hoc inspection of every rank-update window (steps 9–11, 19–21, 29–31)
also found the expected reset signature: adapter A-gradient norm becomes zero
at steps 11/21/31 for every Simplex seed after B is reinitialized to zero,
while B still receives a finite gradient. For example, seed 17 at steps 10→11
has A norm 0.03977→0, B norm 3.27535→3.62210 and head norm 80.1047→119.8193.
The other seeds show the same reset signature. Those head/loss changes are
across different batches and are not an isolated causal estimate of resizing.
The analyzer now preserves all nine surrounding records per Simplex run,
including optimizer inventories and before/after ranks, in
`rank_transition_windows` when re-run on the retained raw archive.

## Rank-search limitation to carry into interpretation

Simplex importance weights stay extremely close to one: observed projected
weights span approximately **0.9999936–1.0000108**. The prescribed floor rule
therefore produces ranks 7 or 8 around an integer boundary. Final total ranks
are 181, 178 and 179 versus the uniform budget of 192; final adapter parameters
are 324352, 318976 and 320768 versus 344064.

Joint clipping applies to importance gradients as documented, and the head
dominates that norm. This combination strongly scales down the importance
updates. The observed rank allocation is sensitive to small deviations around
one; it should not be described as evidence of strongly differentiated layer
importance. We preserve this preregistered configuration for the requested full
comparison. Any future separate-group clipping, head initialization or rank
apportionment ablation must be reported separately for both methods; it must
not overwrite these results or be presented as a reproduction of paper scores.

## Execution and persistence

- Image source: `b5d40812c9ba7eb1a8ef8c28d4b9e0e262367753`.
- Image digest: `sha256:2d630d23e069ed5b486520dbd303c3c35a7cc8c56ec79d269c710819adf7fa08`.
- Training source hash: `2f1b91c4b02e5d0b937edfd962ef941cb6f1f1497cf3470eba93d4be971734b9`.
- Actual CUDA/int8 gate passed before workers; Qwen had 168 quantized linears.
- GPU `b3etq0rz9zjox0`, US-PA-1, $0.69/h: 624.88 seconds, estimated $0.119768.
- CPU `aqm6nnirql03ex`, AP-JP-1, $0.06/h: 406.28 seconds, estimated $0.006771.
- Both deletion confirmations are in the artifact. Independent
  [post-run inspection](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41332728072)
  reported zero managed Pods. Global Volume was retained.
- Global path: `/global/simplexlora-bench/simplex-41331245400-1/`.
- Archive: 104232960 bytes, 110 files; SHA-256
  `f65fcc416e70805d6c0f68351f25dac2ff360feacde987a8cca42d85fb1b51ac`.
- GitHub artifact ID 13093479651, ZIP SHA-256
  `a1a63458e261648545479fd96097e0f6b5c7975ca0a9ebd63104df0e61f40b87`;
  downloaded bytes were independently verified. Retention: 90 days.
- Fresh CPU restoration verified all files and regenerated the identical summary.

Total estimated diagnostic compute: **$0.12654**, not an invoice; existing
volume and request charges are separate.

## Full-run plan within the approved $10 session budget

Six 50-step training-plus-validation windows total 174.35 seconds; the complete
GPU worker, including setup, test evaluation, gate and export, took 481.91s.
Extrapolation with unchanged per-step/evaluation cost is
`481.91 + 9 × 174.35 = 2051.04s`. Adding the observed startup/retrieval overhead
gives roughly **36.6 GPU minutes**, about **$0.43 compute including CPU restore**.
This is an estimate, not a measured full benchmark.

Use a **75-minute GPU deadline and $1 compute cap**; the ceiling including
5-minute cleanup reserves and CPU rate cap is $0.98667. This leaves substantial
headroom within the user's approved total $10. Recheck live inventory, rate,
account budget and absence of managed Pods before creation. Formal results
remain pending until all six 500-step workers and restoration pass.

**Subsequent outcome:** the [formal benchmark](benchmark-41333468131.md)
completed all six 500-step runs and cross-region restoration. All six measured
50-step prefixes match this diagnosis exactly for batches, initial heads,
learning rates, losses, gradient norms and rank histories. The original
diagnostic and smoke archives are retained unchanged.
