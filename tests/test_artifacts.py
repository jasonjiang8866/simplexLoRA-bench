import copy
import json

import pytest
import torch
from safetensors.torch import save_file

from simplexbench.cli import read_manifest, summarize, verify_run
from simplexbench.common import atomic_json, digest, file_hash, validate_config


@pytest.fixture
def manifest():
    from pathlib import Path
    config = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())
    value = {"config": config, "source_hash": "fixture", "requested_config": config}
    value["id"] = digest(value)
    return value


def bundle(directory, manifest, method, seed):
    directory.mkdir()
    (directory / "adapter").mkdir()
    c = manifest["config"]
    result = dict(status="passed", method=method, seed=seed, manifest_id=manifest["id"], steps=c["steps"],
                  data_hashes={"train": "a", "validation": "b", "test": "c"},
                  validation_macro_f1=.6, test_macro_f1=.5, quantized_linear_count=2,
                  time_to_target_seconds=None, target_reached=False, censor_seconds=10,
                  step_seconds_mean=.1, steady_step_seconds_median=.1,
                  peak_allocated_bytes=100, peak_reserved_bytes=200, gradient_anomaly_steps=0)
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "provenance.json", {"gpu": "fixture", "cuda": "fixture", "packages": {}})
    (directory / "steps.jsonl").write_text("".join(json.dumps({"step": n}) + "\n"
                                                 for n in range(1, c["steps"] + 1)))
    atomic_json(directory / "evaluations.json", [{"step": c["steps"]}])
    atomic_json(directory / "ranks.json", [{"adapter_parameters": 8, "budget": 8}] * c["rank_updates"])
    atomic_json(directory / "anomalies.json", [])
    atomic_json(directory / "adapter/adapter_config.json", {})
    save_file({"layer.lora_A.weight": torch.zeros(8, 16),
               "layer.lora_B.weight": torch.zeros(16, 8), "score.weight": torch.zeros(4, 16)},
              directory / "adapter/adapter_model.safetensors")
    seal(directory, manifest, method, seed)


def seal(directory, manifest, method, seed):
    atomic_json(directory / "status.json", {
        "status": "passed", "manifest_id": manifest["id"], "method": method, "seed": seed,
        "artifacts": {str(p.relative_to(directory)): file_hash(p)
                      for p in directory.rglob("*") if p.is_file() and p.name != "status.json"}})


def test_six_run_summary_and_censoring(tmp_path, manifest):
    atomic_json(tmp_path / "manifest.json", manifest)
    for seed in manifest["config"]["seeds"]:
        for method in ("lora", "simplex"):
            bundle(tmp_path / f"{method}-{seed}", manifest, method, seed)
    summary = summarize(tmp_path)
    assert len(summary["runs"]) == 6
    assert summary["aggregates"]["simplex"]["time_to_target"]["reached_only_mean_seconds"] is None
    assert (tmp_path / "summary.md").is_file()
    # A preexisting passed status cannot hide a missing checkpoint.
    (tmp_path / "simplex-17/adapter/adapter_model.safetensors").unlink()
    with pytest.raises(OSError):
        summarize(tmp_path)


@pytest.mark.parametrize("mutation", ["checksum", "steps", "rank_budget", "seed", "missing_head"])
def test_false_success_rejected(tmp_path, manifest, mutation):
    directory = tmp_path / "run"
    bundle(directory, manifest, "simplex", 17)
    if mutation == "checksum":
        (directory / "result.json").write_text("{}")
    elif mutation == "steps":
        (directory / "steps.jsonl").write_text('{"step":1}\n')
        seal(directory, manifest, "simplex", 17)
    elif mutation == "rank_budget":
        atomic_json(directory / "ranks.json", [{"adapter_parameters": 9, "budget": 8}] * 2)
        seal(directory, manifest, "simplex", 17)
    elif mutation == "seed":
        seal(directory, manifest, "simplex", 42)
    else:
        save_file({"layer.lora_A.weight": torch.zeros(8, 16)},
                  directory / "adapter/adapter_model.safetensors")
        seal(directory, manifest, "simplex", 17)
    with pytest.raises(ValueError):
        verify_run(directory, manifest, "simplex", 17)


def test_manifest_tampering(tmp_path, manifest):
    manifest["config"]["steps"] = 100
    atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(tmp_path / "manifest.json")


@pytest.mark.parametrize("key,value", [("bits", 4), ("rank", 16), ("seeds", [1, 1, 2]),
                                      ("steps", 1), ("target_f1", float("nan"))])
def test_invalid_config(manifest, key, value):
    config = copy.deepcopy(manifest["config"])
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)
