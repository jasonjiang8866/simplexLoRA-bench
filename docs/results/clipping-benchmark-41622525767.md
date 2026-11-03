# Paired gradient-clipping benchmark — 2026-11-03

**Status: accepted.** All 18 fresh 500-step processes (9,000 optimizer steps, nine pairs) passed validation. Grouped clipping did not show a consistent test macro-F1 benefit in these three seeds. The global policy remains the existing default.

Each method (LoRA, SimplexLoRA and DoRA) was independently trained with global and grouped clipping. Every pair shares initial full trainable/frozen hashes, dataset/batches and LR. One GPU UUID, source and immutable image cover the entire formal matrix. Historical benchmark results were not substituted for the fresh control runs.

## Primary endpoint

Test macro-F1, mean ± sample SD over seeds 17, 42 and 123. F1 is shown as percent; differences are percentage points. Paired SD is calculated on the three differences.

| Method | Global F1 (%) | Grouped F1 (%) | Grouped − global (pp) |
|---|---:|---:|---:|
| lora | 88.4845 ± 0.5803 | 88.4872 ± 0.6200 | +0.0027 ± 0.1465 |
| simplex | 88.3818 ± 0.5387 | 88.1950 ± 0.2811 | -0.1868 ± 0.2586 |
| dora | 88.7435 ± 0.6463 | 88.5976 ± 0.4436 | -0.1459 ± 0.3659 |

LoRA is essentially unchanged in mean; Simplex and DoRA means are slightly lower with grouped clipping. These descriptive results neither establish equivalence nor prove one policy generally superior. No hyperparameters or runs were selected using final test results.

| Method | Seed 17 Δ (pp) | Seed 42 Δ (pp) | Seed 123 Δ (pp) |
|---|---:|---:|---:|
| lora | -0.0575 | -0.1042 | +0.1697 |
| simplex | -0.4025 | +0.0999 | -0.2579 |
| dora | -0.0919 | +0.1900 | -0.5357 |

## Runtime and target

All 18 runs reached eligible validation macro-F1 ≥ 0.85. TTT includes search, validation and instrumentation, excludes setup/final test; Simplex search-phase crossings are ineligible. No censored values were converted to zero.

| Method | Global / grouped TTT mean (s) | Paired TTT Δ mean ± SD (s) | Mean paired step-time change | Global / grouped peak allocated (MiB, mean) |
|---|---:|---:|---:|---:|
| lora | 118.57 / 114.36 | -4.20 ± 15.62 | -1.828% | 4493.01 / 4493.54 |
| simplex | 112.99 / 120.89 | +7.90 ± 7.49 | -0.719% | 4577.28 / 4576.24 |
| dora | 111.44 / 123.26 | +11.83 ± 9.94 | +1.415% | 4662.89 / 4663.56 |

These small timing differences include monitoring and run-order/device-load variation; they are not isolated kernel speedups.

## Mechanistic observations

During the first 30 Simplex steps, grouped clipping increases maximum importance deviation from one by about 35.7–58.5×: global 5.60e-6–1.24e-5 versus grouped 2.24e-4–7.26e-4. The realized change remains below 0.001. Final per-layer ranks remain 7 or 8 in every seed; larger importance gradients did not produce broad rank diversification or an F1 improvement in this experiment. Gradient SGD, regularization, actual floating-point delta, projection and final reset remain separately recorded.

Grouped clipping caps head and adapters separately at 1, allowing a total L2 bound of sqrt(2); global clipping caps the total at 1. Therefore the experiment changes both geometry and total allowed norm. It does not isolate classifier suppression as a causal explanation and is not a direct paper reproduction.

## Provenance and acceptance

- Image source: `2773aac993dd5e7aa639da3979421202f56fd7cc`; training source hash `c61c6c9721df078032022d68eccedab9ad90beda7f2e3beebdfb13c208e1d91d`.
- Image: `ghcr.io/jasonjiang8866/simplexlora-bench@sha256:6f9bd108b6e39c473660311c618fabb6e254903bd62cb75746afdfc3d66ee60f`.
- Model: Qwen/Qwen2.5-0.5B at `060db6499f32faf8b98477b0a26969ef7d8b9987`; AG News at `eb185aade064a813bc0b7f42de02595523103ca4`. Same pre-registered 8192/1024/2048 subsets, int8 base, r=8, q_proj, 500-step schedule.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb, US-PA-1. Reader: fresh CPU in EU-CZ-1, after GPU deletion.
- Real CUDA/int8 gate, all 18 sealed runs, pairing/order, finite gradient bounds, frozen base, optimizer coverage, rank budgets and exact checkpoint reload passed. Local independent revalidation regenerated identical summary bytes.
- All 340 completed files were recovered on the fresh CPU, with exact archive/file hashes and identical summary.

## Lifecycle and cost

- [Formal run](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41622525767): 2026-11-03 18:50–20:50 SGT. GPU deleted at 20:43:53; CPU deleted at 20:50:07.
- [Independent final inventory](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41645158559): zero managed Pods at 2026-11-03 21:32:06 SGT. Existing Global Volume and historical results retained.
- GPU lifetime 6820.96 s at $0.69/h; CPU lifetime 373.95 s at $0.06/h. Formal compute estimate **$1.313583**; diagnosis plus formal **$1.705885**, against the **$10 total** authorization.
- These are elapsed-time compute estimates, not final invoices. Temporary disk, incremental Global Volume and request fees are additional; retained storage can continue to accrue after compute deletion. No retry or extra training run was launched.

## Evidence

- [Full Actions artifact](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41622525767/artifacts/13232300604) (90-day retention): all logs, metadata and nested checkpoints/telemetry.
- [Compact machine-readable report](clipping-benchmark-41622525767.json) includes all paired results, aggregate metrics, raw importance/rank histories, lifecycle evidence and checksums.
- ZIP SHA-256: `fae1507cbd79e3efba7b4fea6908e3f0fcd16e3c9c94a60c996bb146b525b1f1`.
- TAR SHA-256: `de475ae76024727c58eadfd01c6d7995080052d2a2d37f4a185309b5b8339c0c` (330035200 bytes).
- Summary SHA-256: `c677d97dc35f22ca5fed5f3284b157fcdbc2993b64443a41a26efe72731a3d86`.
- Global snapshot SHA-256: `aab3ffa19b16192118e4a5aa16a818760cb65594da2b7b12478487f47f46d90c`.

No new paid follow-up is required to accept this result. PR merge remains a separate decision.
