# Validation record — 2026-10-30

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
No GPU was available locally. Actual CUDA/int8 gate, six training runs,
time-to-target, F1, peak VRAM, and GPU step time remain unmeasured.
The Dockerfile and GitHub workflow provide separate image-build validation;
consult the PR checks for their current result. No RunPod API create request
was made during development.
