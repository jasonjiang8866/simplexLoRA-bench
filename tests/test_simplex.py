import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForSequenceClassification

from simplexbench.compat import check
from simplexbench.runner import gradient_stats
from simplexbench.simplex import SimplexController, paper_ranks, project_simplex, rebuild_optimizer


@pytest.mark.parametrize("values,mass", [([1., 1., 1.], 3), ([-5., 1., 9.], 3),
                                        ([0., 0.], 2), ([1000., -1000.], 2)])
def test_projection_kkt(values, mass):
    v = torch.tensor(values)
    p = project_simplex(v, mass)
    assert torch.all(p >= 0)
    assert p.sum().item() == pytest.approx(mass)
    active = p > 0
    threshold = (v - p)[active]
    torch.testing.assert_close(threshold, threshold[0].expand_as(threshold))
    assert torch.all(v[~active] - threshold[0] <= 1e-5)


def test_invalid_projection():
    with pytest.raises(ValueError):
        project_simplex(torch.tensor([float("nan")]), 1)


def test_floor_rank_budget():
    ranks = paper_ranks(torch.tensor([0.01, 0.26, 2.73]), 8)
    assert ranks.tolist() == [0, 2, 21]
    assert ranks.sum() <= 24


def tiny():
    model = LlamaForSequenceClassification(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, num_labels=2, pad_token_id=0))
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"],
                                          task_type="SEQ_CLS", modules_to_save=["score"]))


def test_rank_resize_retire_optimizer_and_reload(tmp_path):
    torch.manual_seed(3)
    model = tiny()
    base_state = copy.deepcopy(model.get_base_model().state_dict())
    controller = SimplexController(model, interval=1, updates=1, weight_lr=0, regularization=0)
    optimizer = rebuild_optimizer(model, None, 0.01, 0)
    old_parameters = {p for layer in controller.layers.values()
                      for p in (layer.lora_A["default"].weight, layer.lora_B["default"].weight)}
    with torch.no_grad():
        controller.weights.copy_(torch.tensor([0., 0.5, 2.5]))
    controller.weights.grad = torch.zeros_like(controller.weights)
    assert controller.step(1)
    assert [m.r["default"] for m in controller.layers.values()] == [4, 20]
    assert controller.frozen
    optimizer = rebuild_optimizer(model, optimizer, 0.01, 0)
    current = {p for group in optimizer.param_groups for p in group["params"]}
    assert not (current & old_parameters)
    # A zero-rank adapter must be absent from saved configuration.
    model.save_pretrained(tmp_path)
    import json
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert len(config["target_modules"]) == 2
    assert sorted(config["rank_pattern"].values()) == [4, 20]
    # Reload onto the exact original base, checking PEFT metadata after retirement.
    from peft import PeftModel
    base = LlamaForSequenceClassification(model.config)
    converted = {}
    for name, tensor in base_state.items():
        if "lora_" in name or "modules_to_save" in name:
            continue
        converted[name.replace(".base_layer", "").replace(".original_module", "")] = tensor
    base.load_state_dict(converted, strict=True)
    reloaded = PeftModel.from_pretrained(base, tmp_path).eval()
    model.eval()
    batch = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(batch).logits, reloaded(batch).logits)


def test_unequal_cost_rejected():
    model = LlamaForSequenceClassification(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2))
    model = get_peft_model(model, LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))
    with pytest.raises(ValueError, match="Unequal"):
        SimplexController(model)


def test_gradient_anomalies():
    a, b, c = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    a.grad = torch.tensor([float("nan"), 0.])
    b.grad = torch.tensor([10., 0.])
    stats = gradient_stats([("nan", a), ("spike", b), ("missing", c)], 5)
    assert stats["nonfinite"] == ["nan"]
    assert stats["missing"] == ["missing"]
    assert stats["spike"]


def test_real_peft_cpu():
    result = check()
    assert result["status"] == "passed"
    assert result["methods"]["simplex"]["rank_events"] == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires real CUDA")
def test_real_peft_int8():
    assert check(cuda=True)["bits"] == 8
