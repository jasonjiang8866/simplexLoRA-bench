# Paired gradient-clipping ablation (pre-registered design)

Status: local implementation and CPU validation; **no GPU results or paid execution**.
Base: `4faeaef4b64885a80be4c40ec8e0042792dac1c7` (merged PR #3).
This is an optimization-policy ablation of this repository's implementation,
**not a direct reproduction of the SimplexLoRA paper**. It cannot establish that
head-dominated gradients caused the earlier SimplexLoRA scores.

## Question and primary endpoint

For each existing method and seed, does separating the classifier from adapter
clipping change final test macro-F1 at the same 500-step budget? Report each
paired grouped-minus-global difference and its mean/sample SD over seeds
17, 42 and 123. Do not pool methods or treat 500 steps as 500 independent samples.
Three seeds support descriptive comparisons, not general superiority claims.
Mechanistic and runtime metrics below are secondary; do not select a policy,
LR, threshold or run based on final test results.

## Exact clipping policies

After backward and before AdamW/manual importance SGD, use PyTorch L2 clipping,
max_norm=1.0, epsilon=1e-6 and error_if_nonfinite=True. No AMP/GradScaler or gradient
accumulation is introduced. Each parameter appears exactly once.

| Policy | Classification head | Adapter parameters | Overall L2 bound |
|---|---|---|---|
| global (fresh control) | One joint group with adapters | A, B, DoRA magnitude, active Simplex importance | 1 |
| grouped | One group, cap 1 | One joint group, cap 1: A, B, magnitude, active importance | sqrt(2) |

The coefficient of each group is min(1, 1/(group_norm+1e-6)). Importance is NOT
clipped separately from A/B, nor is each layer clipped independently. Magnitude
remains trainable. Missing, duplicate or unknown trainable parameters fail closed;
zero gradients (including initial A gradients when B=0) are valid. Groups are
rebuilt every step from live parameters after Simplex replaces/retires adapters.
Frozen base tensors and frozen importance are excluded.

Grouped clipping changes both relative group scaling and the permitted total
norm. These are properties of the tested policy, not two separately identified
causes. A matched-total-budget policy would require a separately registered third
arm; it is not silently added here. Larger adapter gradients need not produce
proportional AdamW parameter updates, and improved F1 is not assumed.

Simplex's existing order is preserved: clip data-loss gradients; update A/B/head
with AdamW; update importance by manual SGD using the clipped importance gradient
plus the existing, *unclipped*, regularizer; periodically project/allocate ranks
and reinitialize A/B; reset importance to one when search ends. Weight decay,
importance regularization, projection, rank floor rounding and optimizer-state
preservation are unchanged.

## Fixed configuration and pairing

`benchmark-clipping.json` adds only `clipping_strategies=[global,grouped]` to
`benchmark-dora.json`; `diagnostic-clipping.json` does the same to
`diagnostic-dora.json`. Tests enforce equality of every other field.

- Same pinned Qwen2.5-0.5B revision, int8 base and q_proj targets; r=8, alpha=16.
- Same pinned AG News revision, data_seed, 8192/1024/2048 subsets, tokenization,
  batch_size=8, max_length=128, seed-specific batch order and model initialization.
- Same AdamW/importance LR, decay, warmup, search schedule and 500-step horizon.
- Same evaluation every 25 steps, target macro-F1 >=0.85 and final-only test use.
- Same warnings (>100 is diagnostic, not an acceptance failure by itself).
- Same pinned direct dependencies; no upgrade or baseline/history edits.

Both arms are new, separately seeded processes using one source/image and one
GPU UUID. Historical runs remain historical, never substitute for control runs.
The order retains the existing method rotation and makes each policy pair
adjacent, alternating which policy runs first:

| Seed | Pair 1 | Pair 2 | Pair 3 |
|---|---|---|---|
| 17 | LoRA: global, grouped | Simplex: grouped, global | DoRA: global, grouped |
| 42 | Simplex: grouped, global | DoRA: global, grouped | LoRA: grouped, global |
| 123 | DoRA: global, grouped | LoRA: grouped, global | Simplex: global, grouped |

Nine pairs cannot split first position exactly 50/50 (5 versus 4). This reduces
ordering confounding; it does not eliminate device temperature/load drift.
Within each method/seed pair the verifier requires equal initial full trainable
and frozen-base hashes, head evidence, data hashes, every batch hash and LR.
GPU UUID, package versions, CUDA and Python must match across the entire matrix.
Once policies produce different ranks, adapter shapes/RNG consumption can diverge;
that is a downstream effect, not a reason to force rank allocations to match.

The root manifest binds requested/resolved configuration, source hash and order;
each arm has an independent sealed manifest. A source/config change requires a
new directory. Resume only skips fully verified runs. Corrupt/incomplete runs
fail closed and require a deliberate rerun in a new directory; no selective
best-run replacement. Fresh worker processes are requested for every new run.

## Measurements and acceptance

- Every step: pre/post L2 norms for head, A, B, magnitude and importance; per-policy
  group norms/coefficient/tensor counts; exact clipping bound validation; loss,
  LR, batch hash, instrumented step time and synchronized clipping-call time.
- Simplex search: per-layer importance before/after SGD; gradient-only update L2,
  regularizer update L2, realized floating-point delta L2/max; projected weights,
  projection delta and final reset flag. The final reset is not counted as SGD.
- Rank events: full per-layer allocation, projected weights, parameter budget,
  retired layers, all three search events and final parameter count.
- Final validation/test macro-F1; time-to-target and first eligible evaluation;
  mean/steady median/P95 step time, training wall time and peak allocated/reserved
  memory. Setup and checkpoint reload are separately recorded.
- TTT includes search, validation and monitoring, excludes setup and final test.
  Search-stage Simplex crossings remain ineligible. Unreached runs stay null and
  right-censored; paired time deltas use both-reached pairs, with n and censor
  records disclosed. Do not replace unreached values by zero or the cap.
- Summaries split search/post-search phases and exclude inactive importance
  steps from importance averages. Raw vectors and rank histories remain available.
- All 18 runs must complete with finite, present gradients, unchanged frozen base,
  complete optimizer coverage, valid rank budgets and exact checkpoint reload.
  There is **no requirement for grouped F1, importance or rank diversity to improve**.

Both arms have identical instrumentation. Clipping time includes group construction
and synchronization; instrumented overhead is not isolated GPU kernel performance.
Global baseline math remains the old single clip call. CPU regression compares
legacy versus instrumented global loss, rank history and all final saved tensors
bit-for-bit for all three methods. This is not a promise of bitwise CUDA determinism.

## Commands (do not allocate cloud resources)

```bash
python -m simplexbench.ablation plan --config configs/diagnostic-clipping.json
python -m simplexbench.ablation plan --config configs/benchmark-clipping.json
pytest -q -m 'not cuda'
python -m unittest discover -s runpod -p 'test_*.py' -v
python -m simplexbench.compat --output compatibility-cpu.json
```

After a separately approved GPU launch, the prepared training entrypoints are:

```bash
python -m simplexbench.ablation matrix --config configs/diagnostic-clipping.json --output outputs/clip-diagnosis
python -m simplexbench.ablation matrix --config configs/benchmark-clipping.json --output outputs/clip-formal
python -m simplexbench.ablation summarize --output outputs/clip-formal
```

Matrix runs the real CUDA/int8 compatibility gate before workers; CPU fallback
remains forbidden in production. CPU tests use explicit local Qwen2 fixtures and
CUDA test doubles, not downloaded pretrained weights or bitsandbytes kernels.
Output contains ablation.json, compatibility.json, per-arm manifests and sealed
runs, summary.json and summary.md. The new summary must be used; legacy RunPod
report scripts are not an ablation parser.

## Scale and provisional runtime estimate

| Phase | Fresh runs | Steps/run | Total optimizer steps |
|---|---:|---:|---:|
| Diagnostic prefix | 18 | 50 | 900 |
| Formal (new runs from step zero) | 18 | 500 | 9000 |
| Both | 36 | — | 9900 |

Diagnostic uses schedule_steps=500, so its LR trajectory matches the formal
prefix. It does not shorten the 500-step LR horizon. Review numerical stability,
pairing, clipping bounds, importance decomposition and rank budgets before formal
execution; a negative accuracy signal is not a reason to retune the registered arm.

Historical nine-run diagnosis GPU lifetime was 16.933 minutes; formal lifetime
was 54.026 minutes. Doubling gives approximately **34 + 108 = 142 GPU minutes**
before extra clipping/telemetry overhead and startup variation. Allow roughly
**2.5–3 GPU-hours** for planning, not a measured result or guaranteed ceiling.
The previous formal raw archive was 163.5 MB; two arms are roughly 327 MB before
new telemetry, compression and file overhead. Check actual artifact/storage limits.

At the *historical* $0.69/GPU-hour, 2.5–3 hours means $1.73–$2.07 GPU compute alone.
This is not a current quote or an approved spend. CPU restoration, temporary
storage, transfer/requests, retries and cleanup reserve are additional. The old
$4.50 authorization is not reused.

No RunPod workflow, paid trigger, resource, secret or persistent volume is changed
in this local preparation. Existing RunPod modes intentionally reject these new
profiles until a separately reviewed integration handles the nested 18-run
archive, new summary acceptance, diagnostic review identity and CPU restoration.
Before GPU approval, present a current-price launch specification with exact new
commit/image digest, supported GPU/region, inventory, total cap covering all costs,
deadline/retry policy and independently verifiable cleanup. None are assumed ready
from an old image or historical inspection.
