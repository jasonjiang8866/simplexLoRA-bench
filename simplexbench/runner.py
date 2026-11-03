import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from .common import atomic_json, digest, file_hash
from .simplex import SimplexController, rebuild_optimizer
from .clipping import parameter_group, clip_gradients, validate_clip_evidence, trainable_signature


def synchronize():
    torch.cuda.synchronize()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class ClassificationProbe:
    """Observe the existing head; no extra forward, RNG draw or gradient edits."""
    def __init__(self, model):
        heads = [m for n, m in model.named_modules()
                 if n.endswith("score.modules_to_save.default")]
        if len(heads) != 1:
            raise ValueError("Expected one active classification head")
        self.head = heads[0]
        self.inputs = None
        self.handle = self.head.register_forward_pre_hook(self.capture)

    def capture(self, module, args):
        # Keep only a detached view until this step's backward completes.
        self.inputs = args[0].detach()

    @torch.no_grad()
    def inspect(self, batch, output, pad_id):
        logits, labels = output.logits.detach().float(), batch["labels"]
        mask = batch["attention_mask"].bool()
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = positions.masked_fill(~mask, -1).max(-1).values
        by_id = positions.masked_fill(batch["input_ids"] == pad_id, -1).max(-1).values
        if (last < 0).any() or not torch.equal(last, by_id):
            raise RuntimeError("Attention-mask and classifier padding disagree")
        hidden = self.inputs[torch.arange(len(labels), device=labels.device), last].float()
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
        if not torch.allclose(ce, output.loss.detach().float(), atol=1e-6, rtol=1e-5):
            raise RuntimeError("Loss differs from batch-mean cross entropy")
        reconstructed = torch.nn.functional.linear(hidden, self.head.weight.detach().float(),
            self.head.bias.detach().float() if self.head.bias is not None else None)
        if not torch.allclose(reconstructed, logits, atol=1e-4, rtol=1e-4):
            raise RuntimeError("Classifier pooling/logits mismatch")
        residual = logits.softmax(-1) - torch.nn.functional.one_hot(labels, logits.shape[-1])
        expected = residual.T @ hidden / len(labels)
        actual = self.head.weight.grad.float()
        relative = float((expected - actual).norm() / expected.norm().clamp_min(1e-12))
        if relative > 1e-4:
            raise RuntimeError("Head gradient differs from analytic mean-CE gradient")
        return {"mean_ce": float(ce), "logit_abs_max": float(logits.abs().max()),
                "logit_std": float(logits.std(unbiased=False)),
                "pooled_hidden_norm_mean": float(hidden.norm(dim=-1).mean()),
                "analytic_head_grad_norm": float(expected.norm()),
                "analytic_head_grad_relative_error": relative,
                "nonpadding_tokens_mean": float(mask.sum(-1).float().mean()),
                "label_counts": torch.bincount(labels, minlength=logits.shape[-1]).tolist(),
                "batch_sha256": digest({k: v.detach().cpu().tolist() for k, v in batch.items()})}


def optimizer_inventory(model, optimizer, require_state=False):
    live = {id(p) for p in model.parameters() if p.requires_grad}
    entries = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(entries) != len(set(entries)) or set(entries) != live:
        raise RuntimeError("Missing, duplicate or stale optimizer parameters")
    states = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            state = optimizer.state.get(p, {})
            if require_state and not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
                raise RuntimeError("Missing AdamW state for trained parameter")
            g = states.setdefault(parameter_group(name), {"parameters_with_state": 0, "bytes": 0})
            g["parameters_with_state"] += bool(state)
            g["bytes"] += sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v))
    return {"parameter_tensors": len(entries), "state_tensors": len(optimizer.state),
            "state_by_group": states, "state_bytes": sum(g["bytes"] for g in states.values())}


def gradient_stats(named_parameters, spike_norm):
    missing, nonfinite, zero = [], [], []
    squared = 0.0
    groups = {k: 0.0 for k in ("head", "adapter_A", "adapter_B", "magnitude", "importance", "other")}
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
            group = parameter_group(name)
            groups[group] += n * n
            if n == 0:
                zero.append(name)
    norm = squared ** 0.5
    return {"norm": norm, "missing": missing, "nonfinite": nonfinite,
            "zero": zero, "spike": norm > spike_norm,
            "group_norms": {k: v ** 0.5 for k, v in groups.items()}}


def learning_rate(c, step, method):
    search = c["rank_interval"] * c["rank_updates"]
    if method == "simplex" and step <= search:
        return c["lr"] * 0.1
    offset = search if method == "simplex" else 0
    local, duration = step - offset, c.get("schedule_steps", c["steps"]) - offset
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


def load_base(c):
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
    return model, quantized


def load_model(c, method="lora"):
    model, quantized = load_base(c)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=c["rank"], lora_alpha=c["alpha"],
        lora_dropout=0.0, target_modules=c["target_modules"], bias="none",
        modules_to_save=["score"], use_dora=method == "dora"))
    if any(p.requires_grad for n, p in model.named_parameters()
           if "lora_" not in n and "modules_to_save.default" not in n):
        raise RuntimeError("Unexpected trainable base parameters")
    assert_trainable(model, method)
    return model, quantized


def assert_trainable(model, method):
    groups = parameter_inventory(model)
    expected = {"head", "adapter_A", "adapter_B"} | ({"magnitude"} if method == "dora" else set())
    if set(groups) != expected:
        raise RuntimeError("Unexpected trainable groups or accidentally frozen parameter group")
    for _, layer in model.named_modules():
        if hasattr(layer, "lora_A") and "default" in layer.lora_A:
            params = [layer.lora_A["default"].weight, layer.lora_B["default"].weight]
            if method == "dora":
                params.append(layer.lora_magnitude_vector["default"].weight)
            if not all(p.requires_grad for p in params):
                raise RuntimeError("Accidentally frozen adapter parameter")


def parameter_inventory(model):
    groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            g = groups.setdefault(parameter_group(name), {"numel": 0, "bytes": 0, "tensors": 0})
            g["numel"] += p.numel(); g["bytes"] += p.numel() * p.element_size(); g["tensors"] += 1
    return groups


def frozen_signature(model):
    import hashlib
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if p.requires_grad or "lora_" in n or "modules_to_save" in n:
            continue
        h.update(n.replace(".base_layer.", ".").encode())
        h.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def magnitude_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_magnitude_vector" in n}


@torch.no_grad()
def magnitude_stats(model, initial):
    if not initial:
        return None
    current = dict(model.named_parameters())
    values = torch.cat([current[n].detach().flatten() for n in initial])
    delta = torch.cat([(current[n].detach() - v).flatten() for n, v in initial.items()])
    if not torch.isfinite(values).all():
        raise RuntimeError("Nonfinite DoRA magnitude")
    return {"numel": values.numel(), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "norm": float(values.norm()),
            "delta_l2": float(delta.norm()), "delta_abs_max": float(delta.abs().max())}


@torch.no_grad()
def verify_reload(model, base, destination, batch):
    model.eval()
    expected = model(**batch).logits.detach()
    loaded = PeftModel.from_pretrained(base(), destination).eval()
    actual = loaded(**batch).logits.detach()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    restored = dict(loaded.named_parameters())
    checked = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n not in restored or not torch.equal(p, restored[n]):
                raise RuntimeError("Checkpoint parameter missing or changed: " + n)
            checked.append(n)
    return {"passed": True, "max_logit_difference": float((expected-actual).abs().max()),
            "atol": 2e-3, "rtol": 2e-3, "exact_parameter_names": checked,
            "head_verified": any("modules_to_save.default" in n for n in checked),
            "magnitude_verified": any("lora_magnitude_vector" in n for n in checked)}


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
    model, quantized = load_model(c, method)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()
    controller = SimplexController(model, c["rank"], c["rank_interval"], c["rank_updates"],
                                   c["weight_lr"], c["regularization"]) if method == "simplex" else None
    optimizer = rebuild_optimizer(model, None, c["lr"], c["weight_decay"])
    probe = ClassificationProbe(model)
    head_initial = {"std": float(probe.head.weight.detach().std()),
                    "norm": float(probe.head.weight.detach().norm()),
                    "sha256": __import__("hashlib").sha256(
                        probe.head.weight.detach().cpu().numpy().tobytes()).hexdigest()}
    ablation = "clipping_strategy" in c
    initial_signature = trainable_signature(model, controller) if ablation else None
    optimizer_inventory(model, optimizer)
    frozen_initial = frozen_signature(model)
    magnitude_initial = magnitude_snapshot(model)
    initial_inventory = parameter_inventory(model)
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
        "quantized_linear_count": quantized, "setup_seconds": setup_seconds,
        "head_initial": head_initial, "loss_reduction": "mean", "clip_max_norm": 1.0,
        "pad_token_id": tokenizer.pad_token_id, "initial_parameter_inventory": initial_inventory,
        "magnitude_initial": magnitude_stats(model, magnitude_initial),
        **({"clipping_strategy": c["clipping_strategy"],
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "initial_trainable_sha256": initial_signature,
            "initial_frozen_sha256": frozen_initial} if ablation else {})})
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
            device_batch = {k: v.cuda() for k, v in batch.items()}
            forward = model(**device_batch)
            loss = forward.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            observation = probe.inspect(device_batch, forward, tokenizer.pad_token_id)
            params = list(model.named_parameters())
            if controller and not controller.frozen:
                params.append(("simplex_importance", controller.weights))
            stats = gradient_stats(params, c["gradient_spike_norm"])
            if stats["nonfinite"] or stats["missing"] or stats["spike"]:
                anomalies.append({"step": step, **stats})
                atomic_json(output / "anomalies.json", anomalies)
            if stats["nonfinite"] or stats["missing"]:
                raise RuntimeError(f"Invalid gradients at step {step}: {stats}")
            if ablation:
                synchronize()
                clip_start = time.perf_counter()
                stats["clipping"] = clip_gradients(params, c["clipping_strategy"])
                synchronize()
                clip_seconds = time.perf_counter() - clip_start
                if c["clipping_strategy"] == "global":
                    stats["clip_coefficient"] = stats["clipping"]["groups"]["global"]["coefficient"]
                stats["post_clip"] = gradient_stats(params, c["gradient_spike_norm"])
                validate_clip_evidence(stats, c["clipping_strategy"])
            else:
                clip_norm = torch.nn.utils.clip_grad_norm_([p for _, p in params if p.requires_grad],
                                              max_norm=1.0, error_if_nonfinite=True)
                stats["clip_coefficient"] = min(1.0, 1.0 / (float(clip_norm) + 1e-6))
                stats["post_clip"] = gradient_stats(params, c["gradient_spike_norm"])
                if stats["post_clip"]["norm"] > 1.00001:
                    raise RuntimeError("Gradient clipping failed")
            importance_active = ablation and controller is not None and not controller.frozen
            optimizer.step()
            optimizer_inventory(model, optimizer, require_state=True)
            head_state = optimizer.state.get(probe.head.weight)
            ranks_before = {n: m.r["default"] for n, m in controller.layers.items()} if controller else None
            changed = controller.step(step, telemetry=ablation) if controller else False
            importance = controller.last_update if importance_active else None
            if changed:
                optimizer = rebuild_optimizer(model, optimizer, lr, c["weight_decay"])
                if optimizer.state.get(probe.head.weight) is not head_state:
                    raise RuntimeError("Classifier optimizer moments lost on rank update")
            inventory = optimizer_inventory(model, optimizer)
            magnitude = magnitude_stats(model, magnitude_initial)
            synchronize()
            seconds = time.perf_counter() - before
            step_times.append(seconds)
            if step > c["timing_warmup_steps"] and (not controller or
                    step > c["rank_interval"] * c["rank_updates"] + c["timing_warmup_steps"]):
                stable_times.append(seconds)
            log.write(json.dumps({"step": step, "loss": float(loss.detach()), "lr": lr,
                                  "seconds": seconds, "rank_update": changed, "gradients": stats,
                                  "observations": observation, "optimizer": inventory, "magnitude": magnitude,
                                  "ranks_before": ranks_before if changed else None,
                                  "ranks_after": controller.history[-1] if changed else None,
                                  **({"clip_seconds": clip_seconds, "importance": importance}
                                     if ablation else {})},
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
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final_optimizer = optimizer_inventory(model, optimizer, require_state=True)
    frozen_final = frozen_signature(model)
    if frozen_final != frozen_initial:
        raise RuntimeError("Frozen base changed during training")
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "adapter")
    def reload_base():
        base, _ = load_base(c)
        base.config.pad_token_id = tokenizer.pad_token_id
        return base
    reload_start = time.perf_counter()
    reload_evidence = verify_reload(model, reload_base, output / "adapter", device_batch)
    reload_evidence["frozen_base_sha256"] = frozen_final
    reload_evidence["frozen_base_unchanged"] = True
    reload_evidence["seconds"] = time.perf_counter() - reload_start
    atomic_json(output / "reload.json", reload_evidence)
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
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "initial_parameter_inventory": initial_inventory, "final_parameter_inventory": parameter_inventory(model),
        "optimizer_final": final_optimizer, "magnitude_final": magnitude_stats(model, magnitude_initial),
        "checkpoint_reload": reload_evidence,
        "initial_trainable_parameters": initial_parameters,
        "final_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "gradient_anomaly_steps": len(anomalies), "quantized_linear_count": quantized,
        **({"clipping_strategy": c["clipping_strategy"]} if ablation else {})}
    atomic_json(output / "result.json", result)
    return result
