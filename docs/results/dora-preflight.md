# DoRA extension preparation — 2026-11-02 SGT

Status: implementation/CPU validation complete locally; CI, new image publication,
read-only inventory/price plan and paid GPU validation tracked below. No DoRA GPU
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

Pending: new immutable image receipt and CI links, live inventory/price dry-run,
explicit budget allocation, actual CUDA/int8 gate, GPU diagnosis, reviewed decision,
nine × 500 formal runs, CPU restoration, raw artifacts, costs and cleanup.
