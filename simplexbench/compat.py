"""Offline, real-library PEFT integration gate; --cuda exercises actual int8."""
import argparse
import json
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForSequenceClassification

from .common import atomic_json
from .simplex import SimplexController, rebuild_optimizer


def check(cuda=False):
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA gate requested but CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(91)
    device = "cuda" if cuda else "cpu"
    report = {"status": "running", "device": device, "bits": 8 if cuda else 32, "methods": {}}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=64, num_labels=3, pad_token_id=0)
        LlamaForSequenceClassification(config).save_pretrained(root / "base")

        def base():
            kwargs = {"attn_implementation": "eager"}
            if cuda:
                kwargs.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                              device_map={"": 0}, torch_dtype=torch.float32)
            result = LlamaForSequenceClassification.from_pretrained(root / "base", **kwargs)
            if cuda:
                import bitsandbytes as bnb
                if not any(isinstance(m, bnb.nn.Linear8bitLt) for m in result.modules()):
                    raise RuntimeError("No actual int8 modules in compatibility model")
            return prepare_model_for_kbit_training(result, use_gradient_checkpointing=False)

        batch = {"input_ids": torch.tensor([[1, 3, 5, 7], [1, 8, 9, 2]], device=device),
                 "attention_mask": torch.ones(2, 4, dtype=torch.long, device=device),
                 "labels": torch.tensor([0, 2], device=device)}
        for method in ("lora", "simplex"):
            model = get_peft_model(base(), LoraConfig(
                task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                target_modules=["q_proj"], modules_to_save=["score"], lora_dropout=0.0))
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            controller = SimplexController(model, interval=2, updates=2) if method == "simplex" else None
            optimizer = rebuild_optimizer(model, None, 0.005, 0)
            for step in range(1, 7):
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite compatibility loss")
                if any(p.grad is None or not torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad):
                    raise RuntimeError("Invalid compatibility gradients")
                optimizer.step()
                if controller and controller.step(step):
                    optimizer = rebuild_optimizer(model, optimizer, 0.005, 0)
            if controller and not controller.frozen:
                raise RuntimeError("Rank search did not freeze")
            for n, p in model.named_parameters():
                if n in frozen and not torch.equal(frozen[n], p):
                    raise RuntimeError(f"Frozen base changed: {n}")
            model.eval()
            expected = model(**batch).logits.detach()
            destination = root / method
            model.save_pretrained(destination)
            loaded = PeftModel.from_pretrained(base(), destination).eval()
            actual = loaded(**batch).logits.detach()
            torch.testing.assert_close(actual, expected, atol=1e-5 if not cuda else 2e-3,
                                       rtol=1e-4 if not cuda else 2e-3)
            report["methods"][method] = {
                "forward_backward": True, "frozen_base": True, "save_reload": True,
                "max_logit_difference": float((expected - actual).abs().max()),
                "rank_events": len(controller.history) if controller else 0}
            del model, loaded, optimizer, frozen, controller
            if cuda:
                torch.cuda.empty_cache()
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", default="compatibility.json")
    args = parser.parse_args()
    try:
        report = check(args.cuda)
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        atomic_json(args.output, {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
