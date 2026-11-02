# DoRA extension preparation — 2026-11-02 SGT

Status: implementation, CPU/Docker CI, image publication and both read-only
dry-run passed. Paid CUDA validation remains pending. No DoRA GPU
results or method-superiority conclusions exist at this preparation checkpoint.

- Starting main: `2dd9fec21320f5cc6017634002068ed26e786d4c`; no open PRs.
- Latest historical CI: [41337362698](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41337362698), successful.
- New read-only inspection: [41383612789](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41383612789), successful, `managed_pod_count: 0` at 2026-11-02T03:32:11Z.
- Fixed dependencies unchanged: PEFT 0.17.1, torch 2.9.0, Transformers 4.57.1,
  bitsandbytes 0.48.1. Source review supports attempting int8 DoRA, not claiming it works on CUDA.
- Local Python 3.12 / torch 2.9.0+cpu: 33 CPU tests passed, one CUDA test deselected;
  17 lifecycle tests passed. Tiny real Qwen2 gate passed all three methods.
- Nine-process production orchestration is implemented; the nine-run integration
  fixture executes real tiny CPU Qwen2 training in-process with explicit CUDA/data doubles.
  Fixture scores are discarded and are not performance evidence.
- Tests reject missing magnitude tensors, accidentally frozen magnitude, optimizer
  omission, a six-run result presented as nine, a 50-step result presented as 500,
  and a Llama-only gate presented as Qwen2 validation.
- The retained formal artifact `41333468131` was independently re-read using the
  new code: all hashes/checkpoint shapes passed; regenerated summary was byte-identical.
  Original paired Simplex−LoRA mean remains −0.00102678114261819.

## Published image and CI

- [PR CI 41384544981](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41384544981): Python 3.11, Python 3.12 and Docker successful.
- [Source CI 41384529402](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41384529402): all jobs successful.
- [Image publication 41384626384](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41384626384): successful; 33 CPU tests and three-method CPU gate executed inside image build.
- Source: `fba99733fd746af43fe7574e9cbcddb775737162`.
- Tag: `ghcr.io/jasonjiang8866/simplexlora-bench:sha-fba99733fd746af43fe7574e9cbcddb775737162`.
- Image: `ghcr.io/jasonjiang8866/simplexlora-bench@sha256:3af1106bf5798af9643cce4792f50cea4af832c68a9e803b87dedec9e8a77b71`.
- Training source hash: `c9a8254ea13319910afb90a8d7b2a8f85739680936cb3c59150e6547371862a1`.
- Publication receipt artifact: `13115181095`, 531 bytes, ZIP SHA-256
  `bd70ac486b375904f95e09c203ebae55aba69fdfd5aa3f6f76cb6424bffec3bd`.
  Downloaded and verified `image.txt` against `image-provenance.json`.
  Anonymous GHCR tag retrieval independently reproduced the exact OCI index digest.
  The linux/amd64 image config's OCI revision label independently matched the source SHA.
- Later report/test/documentation changes do not alter training source; dispatch
  pins the controller and image to the source SHA above, never a moving branch.

## Read-only diagnostic preflight

[41385187995](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41385187995)
passed at 2026-11-02 11:50 SGT with `confirm_paid_experiment=false`:

- `diagnostic-dora`, nine × 50 steps; full data subsets and 500-step LR prefix.
- MIG $0.69/hour, available EUR-IS-2 / US-PA-1 / US-WA-1.
- cpu3c, 2 vCPU, AP-JP-1, $0.06/hour; existing Global Volume identity matched.
- Zero account Pods; account balance/monthly budget reserve checks passed.
- Proposed GPU deadline 45m, CPU 15m, compute cap $0.65;
  controller ceiling $0.641667 includes five-minute cleanup reserves and the
  conservative CPU maximum $0.20/hour. Storage/request fees remain separate.
- Artifact `13114689295`, 1642 bytes, ZIP SHA-256
  `85a5ac8ab207816449adc2374a7bfe11404521b6c9b07640c1228597222d5aeb`.
- No compute was created. No new Global Volume result directory was written.

## Read-only formal preflight and proposed authorization

[41385592363](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41385592363)
passed at 2026-11-02T03:55:05Z with `confirm_paid_experiment=false`:

- `benchmark-dora`, exactly nine × 500 steps, same immutable image/source.
- GPU deadline ceiling 120m; CPU restoration deadline 15m; proposed compute cap $1.60.
  Controller worst-case $1.504167 includes five-minute cleanup reserves per phase
  and the conservative CPU maximum $0.20/hour.
- Live inventory/prices and Global Volume matched the diagnostic preflight;
  `other_pod_count: 0`. No compute was created or results written to the volume.
- Artifact `13114613570`, 1629 bytes, ZIP SHA-256
  `78b25f2be5e6689b20e96f24dc5c3277a113a7474626a8d1667990e7963b2ed5`.
  Both preflight artifacts have 90-day retention.

Proposed current-session compute authorization: $2.25 total ($0.65 diagnosis,
$1.60 formal). These are deadline-based caps, **not measured duration estimates**.
After diagnosis passes, forecast nine full runs and CPU restoration from measured
timings and choose a tighter formal deadline within this cap. Stop if the diagnostic
fails or the forecast exceeds it. Storage/request fees are additional; existing
volume retention remains unchanged. No paid authorization has yet been received.

Pending: explicit current-session budget allocation, actual CUDA/int8 gate,
GPU diagnosis and measured formal duration forecast, reviewed decision,
nine × 500 formal runs, CPU restoration, raw results, costs and cleanup.
No historical $10 allocation was treated as open-ended current authorization.
