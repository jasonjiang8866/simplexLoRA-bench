import hashlib
import json
import math
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_config(c):
    for key, value in c.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Nonfinite configuration: {key}")
    for key in ("steps", "eval_every", "batch_size", "max_length", "train_samples",
                "validation_samples", "test_samples", "num_labels", "rank_interval", "rank_updates"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["rank"] != 8 or c["bits"] != 8:
        raise ValueError("This comparison requires rank=8 and bits=8")
    if len(c["seeds"]) != 3 or len(set(c["seeds"])) != 3:
        raise ValueError("Exactly three distinct seeds required")
    if any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in c["seeds"]):
        raise ValueError("Seeds must be uint32 integers")
    if c["rank_interval"] * c["rank_updates"] >= c["steps"]:
        raise ValueError("Rank search must finish before the training budget")
    if not 0 < c["target_f1"] <= 1:
        raise ValueError("Invalid target macro-F1")
    for key in ("lr", "weight_lr", "gradient_spike_norm"):
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if c["warmup_steps"] < 0 or c["weight_decay"] < 0 or c["regularization"] < 0:
        raise ValueError("Negative optimizer setting")
