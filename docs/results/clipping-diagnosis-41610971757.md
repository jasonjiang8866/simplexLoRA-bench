# Paired clipping diagnosis — 2026-11-03

Status: **accepted diagnosis**; formal 500-step benchmark not yet dispatched at this checkpoint.

- [GPU and cross-region CPU recovery run](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41610971757): all 18 x 50-step workers and nine pairs passed.
- Source: `2773aac993dd5e7aa639da3979421202f56fd7cc`; immutable image and hashes are in `clipping-diagnosis-review.json`.
- GPU: RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb, US-PA-1; CPU reader: EU-CZ-1. Real CUDA/int8 gate passed. Original model/data revisions and all registered hyperparameters preserved.
- All 340 archive files recovered; outer ZIP and inner TAR SHA-256 checked. Local revalidation reproduced exactly the GPU/CPU summary bytes.
- Both temporary Pods were deleted. [Independent inventory](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41621509349) found zero managed Pods at 2026-11-03 18:42:44 SGT. The existing Global Volume is retained.
- Estimated compute: $0.392302 (GPU 2026.33 s at $0.69/h; CPU 235.29 s at $0.06/h), excluding storage/request charges; this is not a billing invoice.
- User increased the total experiment budget to **$10**, including already incurred costs, temporary storage, recovery and cleanup. Browser dispatch authorization persists; merge remains outside this execution scope.

## Diagnostic observations (50 steps, not final benchmark)

Grouped minus global; macro-F1 differences are in percentage points. Three paired seeds; mean ± sample SD.

| Method | Test macro-F1 delta (pp) | Mean step time change |
|---|---:|---:|
| LoRA | -0.3092 ± 0.6891 | +0.034% |
| SimplexLoRA | +0.1123 ± 0.2530 | -1.263% |
| DoRA | +0.2897 ± 0.1521 | -1.268% |

During Simplex search, the head contributes about 99.89–99.97% of pre-clip squared gradient norm. Grouping increases maximum importance deviation from one from roughly 0.000006–0.000012 to 0.000224–0.000726 (35.7–58.5x across paired seeds). This supports the expected scaling change, not an accuracy superiority or causal claim. Grouped total norm may reach sqrt(2), compared with 1 for joint clipping.

## Next execution

Keep the immutable image/source and all registered settings. Run 18 independent 500-step processes from zero. The diagnostic-based linear GPU projection is 113.3 minutes; use the reviewed 150-minute deadline, 15-minute CPU recovery limit and $1.85 compute cap. Both phase compute ceilings sum to $2.50, leaving $7.50 of the updated total authorization for incremental storage, requests, cleanup and any reconciled recovery. Actual aggregate spend must still be tracked. Formal dispatch requires this factual review artifact and another live read-only preflight; it must not reuse historical control results or tune on diagnostic F1.
