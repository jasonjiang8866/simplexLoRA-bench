"""Exercise orchestration with a real tiny model and explicit CPU test doubles.

These are not CUDA/quantization or performance evidence.
"""
import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForSequenceClassification

from simplexbench import runner
from simplexbench.cli import verify_run
from simplexbench.common import digest


@pytest.mark.parametrize("method", ["lora", "simplex"])
def test_training_loop_with_cpu_fixture(tmp_path, monkeypatch, method):
    torch.set_num_threads(1)
    config = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())
    model = get_peft_model(Qwen2ForSequenceClassification(Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, num_labels=4)),
        LoraConfig(task_type="SEQ_CLS", r=8, target_modules=["q_proj"], modules_to_save=["score"]))
    class Tokenizer:
        pad_token_id = 0
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}")
    records = [{"input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.ones(4, dtype=torch.long),
                "labels": torch.tensor(i % 4)} for i in range(16)]
    monkeypatch.setattr(runner, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "load_model", lambda c: (model, 1))  # explicit test double
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
    manifest = {"config": config}
    manifest["id"] = digest(manifest)
    output = tmp_path / method
    result = runner.train(config, method, 17, output, manifest["id"])
    assert result["steps"] == 8
    assert verify_run(output, manifest, method, 17)["status"] == "passed"
