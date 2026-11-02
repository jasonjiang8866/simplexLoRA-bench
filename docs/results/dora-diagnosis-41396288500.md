# Three-method CUDA/int8 diagnosis — 2026-11-02 SGT

**Passed: nine independent runs × 50 optimizer steps, 450 total. This is diagnostic evidence, not the 500-step formal benchmark.**

- [Actions 41396288500](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41396288500)
- [Raw artifact 13119824949](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41396288500/artifacts/13119824949): 52,039,451 ZIP bytes, SHA-256 `d709cdaa46985ad92c07a401c0b653e58a2063d34a230cb344447a02dbecb355`; 90-day retention.
- Inner archive: 157,081,600 bytes, SHA-256 `ca65392b9f144a85ea8d0a1d4ee225b8575a7eeabccf03cfbc63dd4fdfac5c8e`; 170 files, 156,764,909 payload bytes. Independently revalidated locally with byte-identical summary.
- Global Volume: `cmvwmviz9000007jteci4ajn8`; path `/global/simplexlora-bench/simplex-41396288500-1`; snapshot SHA-256 `b53cd152d038678e216a07eb3729f61022e1b0eb323ee30a3eacd2216efeaacd`.
- Image: `ghcr.io/jasonjiang8866/simplexlora-bench@sha256:3af1106bf5798af9643cce4792f50cea4af832c68a9e803b87dedec9e8a77b71`; source commit `fba99733fd746af43fe7574e9cbcddb775737162`; training source hash `c9a8254ea13319910afb90a8d7b2a8f85739680936cb3c59150e6547371862a1`.
- GPU `7wkwevfzvy0v82` in US-PA-1 at $0.69/h; deleted before fresh CPU `xden7i4o24fl20` in AP-JP-1 at $0.06/h. Cross-region recovery verified all files and same summary; both deletions confirmed.
- Compute estimate **$0.202260**, from create-to-confirmed-delete durations; storage/request fees extra, not an invoiced amount.

## Diagnosis decision

**Proceed without changing hyperparameters.** Real Qwen2 CUDA/int8 forward/backward and checkpoint reload passed for all three methods. The actual Qwen2.5 runs also verify the frozen base, optimizer coverage/state, every trainable saved tensor, classification head, and DoRA magnitude. All nine reloads have maximum logit difference **0**. No missing gradients or NaN/Inf in 450 steps. Expected zero A gradients with initially zero B remain valid.

DoRA has **369,152** trainable model parameters versus **347,648** for LoRA at r=8, including **21,504** additional magnitude values. DoRA magnitude delta L2 spans **0.19868–0.23117**, with max component changes **0.00472–0.00579**. AdamW tensor state is **2,953,508 bytes** versus **2,781,380 bytes** for LoRA.

Head squared-norm share averages **99.773% LoRA / 99.915% Simplex / 99.759% DoRA**; pooled hidden norm is about **296**. Largest analytic head-gradient relative error is **1.887e-7**. The original spike threshold and clipping remain in force. Simplex importance deviations remain at most **1.085e-5**, with rank 7–8 after floor rounding; this limitation remains part of the interpretation.

## Formal forecast and bounds

Observed GPU lifetime plus nine additional copies of the 50-step training/validation cost gives **56.35 GPU minutes**, approximately **$0.6556 including observed CPU restoration**. This is a forecast, not measured formal performance. Use **90-minute GPU / 15-minute CPU deadlines**, **$1.20 phase cap**, and $1.159167 conservative compute ceiling including five-minute cleanup reserves per phase. Session authorization is $4.50; diagnosis used an estimated $0.202260.

Projected formal payload is about **163.9 MB**, below the **512 MiB payload / 1 GiB archive** limits; most diagnostic payload consists of fixed tokenizer files. Actual formal size/count must still pass verification. Execute all nine formal runs in the balanced cyclic manifest order on one GPU; preserve old six-run results as historical evidence.

![Validation diagnostic curves](dora-diagnosis-41396288500.svg)

![Magnitude and importance diagnostics](dora-diagnosis-41396288500-adaptation.svg)

The following metrics describe the **50-step diagnostic only**.

## Final fixed-budget accuracy

Mean ± sample standard deviation across three paired seeds. Test data is
evaluated once at the final budget; no best-validation checkpoint selection.

| Method | Validation macro-F1 | Test macro-F1 |
|---|---:|---:|
| lora | 0.6388 ± 0.0350 | 0.6197 ± 0.0452 |
| simplex | 0.3181 ± 0.1111 | 0.2985 ± 0.1092 |
| dora | 0.6356 ± 0.0334 | 0.6194 ± 0.0479 |

| Seed | LoRA test F1 | Simplex test F1 | Simplex − LoRA |
|---|---:|---:|---:|
| 17 | 0.670674 | 0.411320 | -0.259354 |
| 42 | 0.603861 | 0.193271 | -0.410590 |
| 123 | 0.584457 | 0.291031 | -0.293427 |

Paired difference mean ± sample SD: **-0.321124 ± 0.079331** (F1 units).
Three seeds do not establish a general significance or method superiority claim.

## Time-to-target and censoring

Target: validation macro-F1 ≥ 0.85 at evaluations every 25 steps.
Time begins at training, includes validation, monitoring and rank search,
and excludes model/data setup and final test evaluation. Simplex search-stage
evaluations are ineligible because the adapters are subsequently reinitialized.
The milestone is the first observed eligible crossing, not a sustained threshold.

| Method | Seed | Reached | First eligible step | Time to target (s) | End of observation (s) |
|---|---:|---|---:|---:|---:|
| lora | 17 | False | — | null (right-censored) | 27.481 |
| simplex | 17 | False | — | null (right-censored) | 28.769 |
| dora | 17 | False | — | null (right-censored) | 29.529 |
| lora | 42 | False | — | null (right-censored) | 28.850 |
| simplex | 42 | False | — | null (right-censored) | 27.838 |
| dora | 42 | False | — | null (right-censored) | 31.372 |
| lora | 123 | False | — | null (right-censored) | 29.174 |
| simplex | 123 | False | — | null (right-censored) | 29.071 |
| dora | 123 | False | — | null (right-censored) | 30.725 |

- lora: reached 0/3; right-censored 3; reached-only mean null.
- simplex: reached 0/3; right-censored 3; reached-only mean null.
- dora: reached 0/3; right-censored 3; reached-only mean null.

Unreached times remain null; no zero substitution or mean over censored values.

## Runtime and memory

Step times synchronize CUDA and include probes/optimizer/rank changes, excluding
validation. Steady median excludes startup/search; P95 below is per-run P95.
Peak memory includes the loaded model, training and evaluation; reserved memory
is allocator reservation, not additional tensor memory.

| Method | Step mean (ms) | Steady median (ms) | Per-run P95 (ms) | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| lora | 181.188 ± 6.827 | 170.901 ± 4.072 | 185.181 ± 12.623 | 4.3877 ± 0.0011 | 4.5534 ± 0.0011 |
| simplex | 180.769 ± 2.202 | 170.191 ± 2.362 | 184.293 ± 1.854 | 4.4700 ± 0.0011 | 4.6523 ± 0.0020 |
| dora | 190.388 ± 5.966 | 182.262 ± 3.505 | 195.768 ± 11.009 | 4.5532 ± 0.0006 | 4.7695 ± 0.0000 |

| Method | Seed | Training + validation (s) | Setup (s) | Final trainable parameters |
|---|---:|---:|---:|---:|
| lora | 17 | 27.481 | 25.083 | 347648 |
| simplex | 17 | 28.769 | 24.758 | 327936 |
| dora | 17 | 29.529 | 21.586 | 369152 |
| lora | 42 | 28.850 | 23.846 | 347648 |
| simplex | 42 | 27.838 | 23.357 | 322560 |
| dora | 42 | 31.372 | 23.469 | 369152 |
| lora | 123 | 29.174 | 23.989 | 347648 |
| simplex | 123 | 29.071 | 24.671 | 324352 |
| dora | 123 | 30.725 | 23.935 | 369152 |

## Gradient evidence

Original norm >100 threshold and clip max_norm=1.0 remain unchanged.
Expected zero A gradients when B is zero are not treated as missing gradients.

| Method | Seed | Norm >100 steps | Missing / nonfinite | Pre-clip range | Mean head squared-norm share | Maximum post-clip norm |
|---|---:|---:|---:|---:|---:|---:|
| lora | 17 | 24/50 | 0 / 0 | 38.945–274.225 | 99.8324% | 1.000000125 |
| simplex | 17 | 37/50 | 0 / 0 | 66.032–272.789 | 99.9086% | 1.000000188 |
| dora | 17 | 23/50 | 0 / 0 | 41.598–273.448 | 99.8208% | 1.000000131 |
| lora | 42 | 37/50 | 0 / 0 | 40.004–284.443 | 99.7857% | 1.000000131 |
| simplex | 42 | 49/50 | 0 / 0 | 93.552–328.336 | 99.9584% | 1.000000132 |
| dora | 42 | 37/50 | 0 / 0 | 39.279–285.234 | 99.7755% | 1.000000220 |
| lora | 123 | 25/50 | 0 / 0 | 42.950–275.769 | 99.7018% | 1.000000172 |
| simplex | 123 | 39/50 | 0 / 0 | 76.487–281.133 | 99.8789% | 1.000000133 |
| dora | 123 | 26/50 | 0 / 0 | 45.489–270.622 | 99.6816% | 1.000000126 |

Largest analytic head-gradient relative error: 1.887e-07.
Group norms (head, A, B, magnitude, importance, other), clipping coefficients, losses,
hidden/logit scales, all pairing checks and full rank weights are retained in JSON.

## Rank allocation

| Seed | Step | Total rank / 192 | Layer rank counts | Adapter parameters / 344064 |
|---|---:|---:|---|---:|
| 17 | 10 | 180 / 192 | r7: 12, r8: 12 | 322560 / 344064 |
| 17 | 20 | 182 / 192 | r7: 10, r8: 14 | 326144 / 344064 |
| 17 | 30 | 181 / 192 | r7: 11, r8: 13 | 324352 / 344064 |
| 42 | 10 | 178 / 192 | r7: 14, r8: 10 | 318976 / 344064 |
| 42 | 20 | 178 / 192 | r7: 14, r8: 10 | 318976 / 344064 |
| 42 | 30 | 178 / 192 | r7: 14, r8: 10 | 318976 / 344064 |
| 123 | 10 | 176 / 192 | r7: 16, r8: 8 | 315392 / 344064 |
| 123 | 20 | 179 / 192 | r7: 13, r8: 11 | 320768 / 344064 |
| 123 | 30 | 179 / 192 | r7: 13, r8: 11 | 320768 / 344064 |

LoRA remains r=8 at all 24 q_proj layers. The classifier has 3584 parameters.
Floor rounding leaves unused adapter capacity; rank search and resets are charged
to the same 50-step budget. The JSON preserves all layer-specific ranks and weights.

## DoRA paired comparisons

| Seed | DoRA − LoRA | Simplex − LoRA | DoRA − Simplex |
|---|---:|---:|---:|
| 17 | +0.002738 | -0.259354 | +0.262092 |
| 42 | -0.001152 | -0.410590 | +0.409438 |
| 123 | -0.002342 | -0.293427 | +0.291084 |
test_f1_delta_simplex_minus_lora: -0.321124 ± 0.079331 (sample SD).
test_f1_delta_dora_minus_lora: -0.000252 ± 0.002657 (sample SD).
test_f1_delta_dora_minus_simplex: +0.320871 ± 0.078059 (sample SD).

## Parameter and optimizer overhead

Same initial rank does not mean equal parameter count. Optimizer bytes are tensor storage, not allocator reservation.

| Method | Seed | Initial trainable | Final trainable | Magnitude parameters | Optimizer state bytes | Magnitude Δ L2 / max |
|---|---:|---:|---:|---:|---:|---:|
| lora | 17 | 347648 | 347648 | 0 | 2781380 | 0 / 0 |
| simplex | 17 | 347648 | 327936 | 0 | 2623684 | 0 / 0 |
| dora | 17 | 369152 | 369152 | 21504 | 2953508 | 0.19868 / 0.00471997 |
| lora | 42 | 347648 | 347648 | 0 | 2781380 | 0 / 0 |
| simplex | 42 | 347648 | 322560 | 0 | 2580676 | 0 / 0 |
| dora | 42 | 369152 | 369152 | 21504 | 2953508 | 0.23117 / 0.00579113 |
| lora | 123 | 347648 | 347648 | 0 | 2781380 | 0 / 0 |
| simplex | 123 | 347648 | 324352 | 0 | 2595012 | 0 / 0 |
| dora | 123 | 369152 | 369152 | 21504 | 2953508 | 0.206308 / 0.00541446 |

Simplex also trains one external SGD importance scalar per adapted layer during search (count is in each rank-history weights list); these are outside AdamW/model parameter totals. Magnitude initial/final values, group gradients and per-step changes are retained in raw evidence.

