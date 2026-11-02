# Validation record

## Formal six-run benchmark — 2026-11-01 UTC / 2026-11-02 SGT

[Full report and raw evidence](results/benchmark-41333468131.md): all six
independent runs completed 500 steps, after the actual CUDA/int8 gate. All
3000 steps had finite, present gradients and verified clipping; norm >100 flags
remain 141/1500 for LoRA and 206/1500 for Simplex. All six reached validation
F1 0.85. The three-seed test F1 means are 0.884845 and 0.883818 respectively;
this small comparison does not establish general method superiority.

Global Volume restoration on a fresh CPU in AP-JP-1 verified 110 files and
reproduced the GPU summary from US-PA-1. A local independent checkpoint/hash
validation also reproduced it byte-for-byte. Both Pods were deleted; final
inspection run 41336870451 reported zero managed Pods. Estimated diagnostic
plus formal compute was $0.558499 within the approved $10 session budget.

This completion is separate evidence from 29 CPU tests, 16 lifecycle tests,
Docker publication, the original eight-step GPU smoke, and the diagnostic runs.

## Gradient diagnosis — 2026-11-02 SGT

[Six 50-step GPU runs and CPU cross-region restoration passed](results/gradient-diagnosis-41331245400.md). All 300 steps passed analytic head-gradient, padding/loss, clipping and optimizer checks; both Pods were deleted. Large norms and near-unit importance weights remain reported limitations, not suppressed flags. The subsequent formal results are recorded above.

## Gradient diagnosis preparation — 2026-11-02 SGT

See [preflight evidence](results/gradient-diagnosis-preflight.md). New probes,
29 CPU tests, 16 lifecycle tests, CPU PEFT gates, image publication and read-only
RunPod dry-run passed. At this preparation checkpoint paid GPU diagnosis still
awaited budget approval; it subsequently completed as recorded above. This
entry records preparation evidence only, not full-benchmark completion.


## RunPod CUDA smoke — 2026-11-01

See the [GPU smoke and Global Volume record](results/runpod-smoke-41324992190.md)
for the immutable image, Actions runs, CUDA/int8 gate, six eight-step runs,
cross-region persistence evidence and compute cleanup. At that checkpoint the
full performance benchmark was pending. All 48 steps were inspected: zero nonfinite/missing
gradient events, but 46 pre-clipping norm >100 flags; this is not a clean
training-stability result.

## Original isolated CPU validation — 2026-10-30

Local isolated Python 3.12 environment: PyTorch 2.9.0+cpu,
Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.11.0.
Result: **27 CPU tests passed, 1 CUDA test deselected**; `pip check` passed.

The real-library CPU compatibility gate passed for LoRA and SimplexLoRA:
forward/backward, unchanged frozen base, two rank updates followed by freezing,
and save/reload with maximum logit difference 0.0 for both methods.

Regression tests additionally exercise Euclidean projection KKT conditions,
floor rank budget, explicit rank contraction/expansion and zero-rank retirement,
saved PEFT metadata, stale optimizer parameter removal, gradient anomalies,
artifact corruption/missing classifier, incomplete matrices and censored targets,
and RunPod dry-run/no-retry behavior.
The complete training/artifact loop is also exercised for both methods with
a real tiny Qwen2 classifier and explicit CPU doubles for CUDA/data loading.
Those fixture runs are temporary test data, not int8 benchmark evidence.

CPU tests do not establish 8-bit GPU compatibility or benchmark performance.
No GPU was available locally during that validation. At that time actual
CUDA/int8 checks and GPU metrics were unmeasured; the later smoke record above
supersedes that status for the short smoke configuration only.
The Dockerfile and GitHub workflow provide separate image-build validation;
consult the PR checks for their current result. No RunPod API create request
was made during development.


## DoRA extension — 2026-11-02 SGT

See [DoRA preparation and status](results/dora-preflight.md). CPU integration,
Docker publication, CUDA/int8 gate, GPU diagnosis and nine-run performance are
separate milestones. Historical six-run results above are not DoRA results.

The [three-method diagnostic 41396288500](results/dora-diagnosis-41396288500.md)
passed the real Qwen2 CUDA/int8 gate, nine Qwen2.5 runs × 50 steps, exact
trainable tensor reload with zero logit difference, frozen-base checks and
optimizer coverage/state checks. No missing/nonfinite gradients in 450 steps.
DoRA magnitude updated in every seed. A fresh AP-JP-1 CPU restored all 170 files
from the US-PA-1 GPU export and reproduced the summary; both Pods were deleted.
The [formal three-method benchmark 41399541318](results/dora-benchmark-41399541318.md)
then completed all nine independent 500-step runs on one US-PA-1 GPU with the
same image. All 4,500 steps had zero missing/nonfinite gradients, with unchanged
base weights, complete optimizer coverage/state, and zero checkpoint logit error.
DoRA magnitude changed in every seed. Original norm>100 warnings remain:
LoRA 141/1500, Simplex 206/1500, DoRA 134/1500; clip max_norm stays 1.0.
All three methods reached validation F1 0.85 in 3/3 runs, with no censoring.

After confirmed GPU deletion, a fresh EU-CZ-1 CPU restored 170 files and the
identical summary. Independent local verification checked every file hash,
checkpoint shape and byte-identical summary regeneration. Payload 163,209,130
bytes and archive 163,522,560 bytes remain within limits; Actions ZIP is
53,273,092 bytes with 90-day retention. Both Pods were confirmed deleted;
[final independent inspection 41406335557](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41406335557)
reported `managed_pod_count: 0` at 2026-11-02T07:26:45Z.
Formal compute estimate is $0.626519; diagnosis plus formal $0.828779 against
the approved $4.50, with storage/request charges additional.

New LoRA/Simplex runs reproduce all six historical test scores, while their
timings remain separate measurements. Historical archives are retained, not
reclassified as three-method results. CPU tests, Docker build/publication,
CUDA/int8 gate, diagnosis, formal benchmark and recovery are distinct evidence;
none by itself establishes statistical method superiority.
