# Validation record

## Gradient diagnosis — 2026-11-02 SGT

[Six 50-step GPU runs and CPU cross-region restoration passed](results/gradient-diagnosis-41331245400.md). All 300 steps passed analytic head-gradient, padding/loss, clipping and optimizer checks; both Pods were deleted. Large norms and near-unit importance weights remain reported limitations, not suppressed flags. Full 500-step performance results are pending.

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
