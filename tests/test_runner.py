"""Exercise orchestration with a real tiny model and explicit CPU test doubles.

These are not CUDA/quantization or performance evidence.
"""
import copy
import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForSequenceClassification

from simplexbench import runner
from simplexbench.cli import verify_run
from simplexbench.common import digest


@pytest.mark.parametrize("method", ["lora", "simplex", "dora"])
def test_training_loop_with_cpu_fixture(tmp_path, monkeypatch, method):
    run_cpu_fixture(tmp_path, monkeypatch, method)


def run_cpu_fixture(tmp_path, monkeypatch, method, seed=17):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    config = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())
    base = Qwen2ForSequenceClassification(Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, num_labels=4))
    model = get_peft_model(copy.deepcopy(base),
        LoraConfig(task_type="SEQ_CLS", r=8, target_modules=["q_proj"], modules_to_save=["score"], use_dora=method == "dora"))
    class Tokenizer:
        pad_token_id = 0
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}")
    records = [{"input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.ones(4, dtype=torch.long),
                "labels": torch.tensor(i % 4)} for i in range(16)]
    monkeypatch.setattr(runner, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "load_model", lambda c, method: (model, 1))  # explicit test double
    monkeypatch.setattr(runner, "load_base", lambda c: (copy.deepcopy(base), 1))
    monkeypatch.setattr(runner, "load_data", lambda *a: (
        {k: records for k in ("train", "validation", "test")}, {"fixture": "cpu-only"}))
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test double")
    version = runner.importlib.metadata.version
    monkeypatch.setattr(runner.importlib.metadata, "version",
                        lambda p: "test-double" if p == "bitsandbytes" else version(p))
    config.update(methods=["lora", "simplex", "dora"], evidence_version=2)
    from simplexbench.common import execution_order
    manifest = {"config": config, "execution_order": execution_order(config)}
    manifest["id"] = digest(manifest)
    output = tmp_path / f"{method}-{seed}"
    result = runner.train(config, method, seed, output, manifest["id"])
    assert result["steps"] == 8
    assert verify_run(output, manifest, method, seed)["status"] == "passed"
    return manifest


def test_nine_run_strict_matrix_and_missing_dora(tmp_path, monkeypatch):
    from simplexbench.cli import summarize
    from simplexbench.common import atomic_json, file_hash
    from safetensors.torch import load_file, save_file
    for seed in (17, 42, 123):
        for method in ("lora", "simplex", "dora"):
            manifest = run_cpu_fixture(tmp_path, monkeypatch, method, seed)
    atomic_json(tmp_path / "manifest.json", manifest)
    summary = summarize(tmp_path)
    assert len(summary["runs"]) == 9
    assert len(summary["paired"]) == 3
    assert all("test_f1_delta_dora_minus_simplex" in p for p in summary["paired"])
    # Exercise the three-method raw-archive analyzer and report, without GPU claims.
    import importlib.util, tarfile
    spec = importlib.util.spec_from_file_location("reports", Path(__file__).parents[1] / "runpod/report_experiment.py")
    reports = importlib.util.module_from_spec(spec); spec.loader.exec_module(reports)
    spec = importlib.util.spec_from_file_location("render", Path(__file__).parents[1] / "runpod/render_report.py")
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    atomic_json(tmp_path / "compatibility.json", {"status": "passed", "device": "cpu", "bits": 32})
    archive = tmp_path.parent / "fixture.tar"
    with tarfile.open(archive, 'w') as tar:
        for path in tmp_path.rglob('*'):
            if path.is_file(): tar.add(path, arcname=str(path.relative_to(tmp_path)))
    analysis = reports.report(archive)
    assert analysis['checks']['complete_matrix']
    assert not analysis['checks']['cuda_int8_gate_passed']
    assert len(analysis['paired_f1_delta_summaries']) == 3
    assert 'DoRA − Simplex' in renderer.render(analysis)
    dora = tmp_path / "dora-17"
    checkpoint = dora / "adapter/adapter_model.safetensors"
    weights = load_file(checkpoint)
    weights.pop(next(k for k in weights if "lora_magnitude_vector" in k))
    save_file(weights, checkpoint)
    status = json.loads((dora / "status.json").read_text())
    status["artifacts"]["adapter/adapter_model.safetensors"] = file_hash(checkpoint)
    atomic_json(dora / "status.json", status)
    with pytest.raises(ValueError, match="magnitude checkpoint"):
        summarize(tmp_path)


def test_gradient_group_decomposition_and_clipping():
    params = []
    for name, grad in [('score.modules_to_save.default.weight', 300.),
                       ('layer.lora_B.default.weight', 400.),
                       ('layer.lora_A.default.weight', 0.), ('simplex_importance', 0.)]:
        p = torch.nn.Parameter(torch.ones(1)); p.grad = torch.tensor([grad]); params.append((name, p))
    before = runner.gradient_stats(params, 100.)
    assert before['norm'] == 500.
    assert before['group_norms']['head'] == 300.
    assert before['group_norms']['adapter_B'] == 400.
    assert before['missing'] == before['nonfinite'] == []
    torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.)
    assert runner.gradient_stats(params, 100.)['norm'] == pytest.approx(1.)


def test_diagnostic_lr_is_exact_full_run_prefix():
    full = json.loads((Path(__file__).parents[1] / 'configs/benchmark.json').read_text())
    diag = json.loads((Path(__file__).parents[1] / 'configs/diagnostic.json').read_text())
    for method in ['lora', 'simplex', 'dora']:
        assert [runner.learning_rate(full, s, method) for s in range(1, 51)] == [
            runner.learning_rate(diag, s, method) for s in range(1, 51)]


def test_dora_accidental_freeze_and_optimizer_omission_rejected():
    base = Qwen2ForSequenceClassification(Qwen2Config(vocab_size=32, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, num_labels=4, pad_token_id=0))
    model = get_peft_model(base, LoraConfig(task_type='SEQ_CLS', r=8,
        target_modules=['q_proj'], modules_to_save=['score'], use_dora=True))
    magnitude = next(p for n, p in model.named_parameters() if 'lora_magnitude_vector' in n)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad and p is not magnitude])
    with pytest.raises(RuntimeError, match='optimizer parameters'):
        runner.optimizer_inventory(model, optimizer)
    magnitude.requires_grad_(False)
    with pytest.raises(RuntimeError, match='frozen adapter'):
        runner.assert_trainable(model, 'dora')


def test_balanced_three_method_order_and_unchanged_hyperparameters():
    from simplexbench.common import execution_order
    root = Path(__file__).parents[1] / 'configs'
    old = json.loads((root/'benchmark.json').read_text())
    new = json.loads((root/'benchmark-dora.json').read_text())
    assert {k: v for k, v in new.items() if k not in ('methods', 'evidence_version')} == old
    order = execution_order(new)
    assert [x['method'] for x in order] == ['lora','simplex','dora', 'simplex','dora','lora', 'dora','lora','simplex']
    diag = json.loads((root/'diagnostic-dora.json').read_text())
    for method in new['methods']:
        assert [runner.learning_rate(new, s, method) for s in range(1, 51)] == [runner.learning_rate(diag, s, method) for s in range(1, 51)]
