import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from .common import atomic_json, digest, file_hash
from .simplex import SimplexController, rebuild_optimizer


def synchronize():
    torch.cuda.synchronize()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def gradient_stats(named_parameters, spike_norm):
    missing, nonfinite, zero = [], [], []
    squared = 0.0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        g = parameter.grad
        if g is None:
            missing.append(name)
        elif not torch.isfinite(g).all():
            nonfinite.append(name)
        else:
            n = float(torch.linalg.vector_norm(g.float()))
            squared += n * n
            if n == 0:
                zero.append(name)
    norm = squared ** 0.5
    return {"norm": norm, "missing": missing, "nonfinite": nonfinite,
            "zero": zero, "spike": norm > spike_norm}


def learning_rate(c, step, method):
    search = c["rank_interval"] * c["rank_updates"]
    if method == "simplex" and step <= search:
        return c["lr"] * 0.1
    offset = search if method == "simplex" else 0
    local, duration = step - offset, c["steps"] - offset
    warmup = min(c["warmup_steps"], max(0, duration - 1))
    if warmup and local <= warmup:
        return c["lr"] * local / warmup
    return c["lr"] * max(0, (duration - local + 1) / max(1, duration - warmup))


def load_data(c, tokenizer):
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    kwargs = {"revision": c["dataset_revision"]}
    if c.get("dataset_config"):
        kwargs["name"] = c["dataset_config"]
    data = load_dataset(c["dataset"], **kwargs)
    source = data[c["train_split"]]
    # Fixed data subset and train/dev split across every seed and method.
    indices = np.arange(len(source))
    train_idx, val_idx = train_test_split(
        indices, train_size=c["train_samples"], test_size=c["validation_samples"],
        random_state=c["data_seed"], stratify=source[c["label_column"]])
    test_source = data[c["test_split"]]
    if c["test_split"] == c["train_split"]:
        raise ValueError("Test split must be separate from train")
    if c["test_samples"] > len(test_source):
        raise ValueError("test_samples exceeds test split")
    test_idx = np.random.default_rng(c["data_seed"]).permutation(len(test_source))[:c["test_samples"]]
    splits = {"train": source.select(train_idx), "validation": source.select(val_idx),
              "test": test_source.select(test_idx)}
    identity = {}
    for name, split in splits.items():
        texts, labels = list(split[c["text_column"]]), list(split[c["label_column"]])
        if set(labels) != set(range(c["num_labels"])):
            raise ValueError(f"{name}: expected all configured labels")
        identity[name] = digest({"texts": texts, "labels": labels})
        tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=c["max_length"])
        splits[name] = [{**{k: torch.tensor(v[i]) for k, v in tokens.items()},
                         "labels": torch.tensor(label)} for i, label in enumerate(labels)]
    return splits, identity


def load_model(c):
    if not torch.cuda.is_available():
        raise RuntimeError("Production benchmarks require CUDA; CPU fallback is forbidden")
    model = AutoModelForSequenceClassification.from_pretrained(
        c["model"], revision=c["model_revision"], num_labels=c["num_labels"],
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0}, torch_dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False)
    import bitsandbytes as bnb
    quantized = sum(isinstance(m, bnb.nn.Linear8bitLt) for m in model.modules())
    if not getattr(model, "is_loaded_in_8bit", False) or not quantized:
        raise RuntimeError("Model is not actually an 8-bit bitsandbytes base")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=c["rank"], lora_alpha=c["alpha"],
        lora_dropout=0.0, target_modules=c["target_modules"], bias="none",
        modules_to_save=["score"]))
    if any(p.requires_grad for n, p in model.named_parameters()
           if "lora_" not in n and "modules_to_save.default" not in n):
        raise RuntimeError("Unexpected trainable base parameters")
    return model, quantized


@torch.no_grad()
def evaluate(model, loader, num_labels):
    model.eval()
    predictions, labels = [], []
    for batch in loader:
        labels.extend(batch["labels"].tolist())
        logits = model(**{k: v.cuda() for k, v in batch.items()}).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite evaluation logits")
        predictions.extend(logits.argmax(-1).cpu().tolist())
    model.train()
    return float(f1_score(labels, predictions, average="macro",
                          labels=list(range(num_labels)), zero_division=0))


def train(c, method, seed, output, manifest_id):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "status.json", {"status": "running"})
    try:
        result = _train(c, method, seed, output, manifest_id)
        files = [p for p in output.rglob("*") if p.is_file() and p.name != "status.json"]
        atomic_json(output / "status.json", {"status": "passed", "manifest_id": manifest_id,
                    "method": method, "seed": seed,
                    "artifacts": {str(p.relative_to(output)): file_hash(p) for p in files}})
        return result
    except BaseException as exc:
        atomic_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise


def _train(c, method, seed, output, manifest_id):
    seed_all(seed)
    setup_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"],
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    splits, identity = load_data(c, tokenizer)
    model, quantized = load_model(c)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()
    controller = SimplexController(model, c["rank"], c["rank_interval"], c["rank_updates"],
                                   c["weight_lr"], c["regularization"]) if method == "simplex" else None
    optimizer = rebuild_optimizer(model, None, c["lr"], c["weight_decay"])
    initial_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loader = DataLoader(splits["train"], batch_size=c["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    evaluations = {n: DataLoader(splits[n], batch_size=c["batch_size"], shuffle=False)
                   for n in ("validation", "test")}
    setup_seconds = time.perf_counter() - setup_start
    atomic_json(output / "provenance.json", {
        "manifest_id": manifest_id, "config": c, "method": method, "seed": seed,
        "data_hashes": identity, "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate")},
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "quantized_linear_count": quantized, "setup_seconds": setup_seconds})
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    batch_iter = iter(loader)
    step_times, stable_times, evals, anomalies = [], [], [], []
    target_seconds = None
    with (output / "steps.jsonl").open("w", buffering=1) as log:
        for step in range(1, c["steps"] + 1):
            synchronize()
            before = time.perf_counter()
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)
            optimizer.zero_grad(set_to_none=True)
            if controller:
                controller.weights.grad = None
            lr = learning_rate(c, step, method)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = model(**{k: v.cuda() for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            params = list(model.named_parameters())
            if controller and not controller.frozen:
                params.append(("simplex_importance", controller.weights))
            stats = gradient_stats(params, c["gradient_spike_norm"])
            if stats["nonfinite"] or stats["missing"] or stats["spike"]:
                anomalies.append({"step": step, **stats})
                atomic_json(output / "anomalies.json", anomalies)
            if stats["nonfinite"] or stats["missing"]:
                raise RuntimeError(f"Invalid gradients at step {step}: {stats}")
            torch.nn.utils.clip_grad_norm_([p for _, p in params if p.requires_grad],
                                          max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            changed = controller.step(step) if controller else False
            if changed:
                optimizer = rebuild_optimizer(model, optimizer, lr, c["weight_decay"])
            synchronize()
            seconds = time.perf_counter() - before
            step_times.append(seconds)
            if step > c["timing_warmup_steps"] and (not controller or
                    step > c["rank_interval"] * c["rank_updates"] + c["timing_warmup_steps"]):
                stable_times.append(seconds)
            log.write(json.dumps({"step": step, "loss": float(loss.detach()), "lr": lr,
                                  "seconds": seconds, "rank_update": changed, "gradients": stats},
                                 allow_nan=False) + "\n")
            if step % c["eval_every"] == 0 or step == c["steps"]:
                f1 = evaluate(model, evaluations["validation"], c["num_labels"])
                synchronize()
                elapsed = time.perf_counter() - start
                eligible = not controller or controller.frozen
                evals.append({"step": step, "macro_f1": f1, "elapsed_seconds": elapsed,
                              "target_eligible": eligible})
                # Search-stage performance is discarded with reinitialized adapters.
                if eligible and target_seconds is None and f1 >= c["target_f1"]:
                    target_seconds = elapsed
                atomic_json(output / "evaluations.json", evals)
                print(json.dumps({"method": method, "seed": seed, **evals[-1]}), flush=True)
    synchronize()
    train_seconds = time.perf_counter() - start
    test_f1 = evaluate(model, evaluations["test"], c["num_labels"])
    if controller and not controller.frozen:
        raise RuntimeError("Incomplete rank search")
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "adapter")
    atomic_json(output / "ranks.json", controller.history if controller else [])
    atomic_json(output / "anomalies.json", anomalies)
    result = {
        "status": "passed", "method": method, "seed": seed, "manifest_id": manifest_id,
        "steps": c["steps"], "data_hashes": identity,
        "validation_macro_f1": evals[-1]["macro_f1"], "test_macro_f1": test_f1,
        "target_f1": c["target_f1"], "time_to_target_seconds": target_seconds,
        "target_reached": target_seconds is not None, "censor_seconds": train_seconds,
        "train_wall_seconds": train_seconds, "setup_seconds": setup_seconds,
        "step_seconds_mean": float(np.mean(step_times)),
        "steady_step_seconds_median": float(np.median(stable_times)) if stable_times else None,
        "step_seconds_p95": float(np.percentile(step_times, 95)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "initial_trainable_parameters": initial_parameters,
        "final_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "gradient_anomaly_steps": len(anomalies), "quantized_linear_count": quantized}
    atomic_json(output / "result.json", result)
    return result
