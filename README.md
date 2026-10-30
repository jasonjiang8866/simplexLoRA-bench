# simplexLoRA-bench

An isolated **8-bit frozen base + sequence-classification head** benchmark:
standard LoRA **r=8** versus a paper-based SimplexLoRA implementation, with
three paired seeds and a RunPod launch path.

This is an engineering experiment, not a reproduction of the paper's reported
scores. No GPU performance result is claimed until the CUDA gate and six runs
have passed. The implementation uses stock PEFT; it does not install the
authors' modified PEFT fork.

| Default | Value |
|---|---|
| Model / data | Qwen2.5-0.5B / AG News (4 classes) |
| Base / adapters | bitsandbytes int8 / FP32 LoRA and classification head |
| Targets | q_proj, equal parameter cost across layers |
| Seeds | 17, 42, 123; same train/dev/test subset across all runs |
| Budget | 500 total optimizer steps per run, including rank search |
| Simplex search | 3 updates every 10 steps; then fixed-rank training |
| Metrics | validation/test macro-F1, time-to-target, step time, allocated/reserved peak VRAM, gradient anomalies |
| Target | validation macro-F1 ≥ 0.85, evaluated every 25 steps |

### Isolated installation and compatibility

Python 3.11/3.12. Choose the correct PyTorch wheel **before** installing the project.
For CUDA 12.8 (including Blackwell), in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install ".[gpu,test]"
pip check
pytest -q -m "not cuda"
python -m simplexbench.compat --cuda --output compatibility-gpu.json
```

CPU-only validation uses the PyTorch CPU index and `.[test]`; omit `--cuda`.
It tests real PEFT, Transformers and randomly initialized local Llama weights,
without downloading a model. It does **not** validate bitsandbytes GPU kernels.
The CUDA gate constructs, saves and loads a tiny local model in actual int8,
then checks backward, rank changes, frozen-base integrity and adapter reload.

### Run experiments

```bash
# Six short runs: pipeline verification, not accuracy evidence.
simplex-bench matrix --config configs/smoke.json --output outputs/smoke
# Six full runs, each in a fresh subprocess.
simplex-bench matrix --config configs/benchmark.json --output outputs/benchmark
# Continue after interruption: verify complete runs; restart incomplete runs.
simplex-bench matrix --config configs/benchmark.json --output outputs/benchmark --resume
simplex-bench summarize --output outputs/benchmark
```

The matrix resolves model and dataset revisions to immutable Hub commits once,
saves the resolved config and source hash, and uses that manifest for every run.
Changing source/config requires a new output directory. Original incomplete
attempts are retained in `incomplete/`; **mid-step optimizer resume is not implemented**.
For custom data, supply a Hub dataset with separate train/test splits, text and
integer label columns. Update the config's model, dataset, columns and class
count. The model must support a `score` classification head and the target
modules must have equal per-rank costs. Private/gated repositories use a local
`HF_TOKEN`; the default model and dataset do not need one.

### Results and interpretation

Each run writes `steps.jsonl`, `evaluations.json`, `anomalies.json`,
`ranks.json`, `provenance.json`, `result.json` and a reloadable PEFT
adapter **including the classifier**. The status file seals artifact hashes.
The summary requires all six runs, contiguous steps, final evaluations, valid
adapter tensors, matching data/hardware/software, and completed rank search.

- Report mean ± sample standard deviation and paired F1 deltas. Three seeds
  provide a small engineering comparison, not a robust significance claim.
- Time-to-target is measured from the first training batch, includes rank
  search, validation and monitoring, and excludes model/data setup. Only
  post-search Simplex evaluations qualify, since search weights are discarded.
  Unreached targets are `null` with a censoring time, never zero or dropped
  silently. Reached-only averages are explicitly labeled.
- Step time is CUDA-synchronized and includes batch transfer, backward,
  gradient monitoring, optimizer and rank changes; excludes validation.
  Steady-state median excludes startup and rank-search steps.
- Peak allocated/reserved VRAM includes the loaded model and training/evaluation.
  Gradient logs record pre-clipping norm, NaN/Inf, missing and zero gradients.
  NaN/Inf or missing trainable gradients abort; norm >100 is flagged.
  Zero A gradients at standard LoRA initialization are expected and only logged.

See [RunPod instructions](runpod/README.md), [method decisions](docs/METHOD.md)
and [validation record](docs/VALIDATION.md).
