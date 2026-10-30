import argparse
import importlib.metadata
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .common import atomic_json, digest, file_hash, validate_config

METHODS = ("lora", "simplex")


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest["id"] != digest({k: v for k, v in manifest.items() if k != "id"}):
        raise ValueError("Manifest checksum mismatch")
    validate_config(manifest["config"])
    return manifest


def verify_run(directory, manifest, method, seed):
    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text())
    if status["status"] != "passed":
        raise ValueError(f"Incomplete run: {directory}")
    for key, expected in (("manifest_id", manifest["id"]), ("method", method), ("seed", seed)):
        if status.get(key) != expected:
            raise ValueError(f"Run identity mismatch: {key}")
    artifacts = status.get("artifacts", {})
    required = {"result.json", "provenance.json", "steps.jsonl", "evaluations.json", "ranks.json",
                "anomalies.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"}
    if not required <= artifacts.keys():
        raise ValueError("Missing required artifacts")
    for relative, expected in artifacts.items():
        p = directory / relative
        if not p.resolve().is_relative_to(directory.resolve()) or file_hash(p) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    result = json.loads((directory / "result.json").read_text())
    for key, expected in (("manifest_id", manifest["id"]), ("method", method), ("seed", seed),
                          ("status", "passed"), ("steps", manifest["config"]["steps"])):
        if result.get(key) != expected:
            raise ValueError(f"Result mismatch: {key}")
    rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    if [row["step"] for row in rows] != list(range(1, result["steps"] + 1)):
        raise ValueError("Incomplete/duplicate training steps")
    evals = json.loads((directory / "evaluations.json").read_text())
    if not evals or evals[-1]["step"] != result["steps"]:
        raise ValueError("Missing final evaluation")
    if method == "simplex":
        ranks = json.loads((directory / "ranks.json").read_text())
        if len(ranks) != manifest["config"]["rank_updates"]:
            raise ValueError("Missing rank-search history")
        if any(row["adapter_parameters"] > row["budget"] for row in ranks):
            raise ValueError("Parameter budget violation")
    if result["quantized_linear_count"] <= 0:
        raise ValueError("Missing 8-bit base evidence")
    for key in ("validation_macro_f1", "test_macro_f1"):
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if result["target_reached"] != (result["time_to_target_seconds"] is not None):
        raise ValueError("Inconsistent target result")
    if result["target_reached"] and not 0 <= result["time_to_target_seconds"] <= result["censor_seconds"]:
        raise ValueError("Invalid time-to-target")
    from safetensors import safe_open
    with safe_open(directory / "adapter/adapter_model.safetensors", framework="pt") as weights:
        keys = list(weights.keys())
        if not any("lora_A" in k for k in keys) or not any("score" in k for k in keys):
            raise ValueError("Adapter/classifier checkpoint incomplete")
        for key in keys:
            if "lora_A" in key:
                bkey = key.replace("lora_A", "lora_B")
                if bkey not in keys or weights.get_slice(key).get_shape()[0] != weights.get_slice(bkey).get_shape()[1]:
                    raise ValueError("Invalid adapter shapes")
    return result


def summarize(output):
    output = Path(output)
    atomic_json(output / "status.json", {"status": "validating"})
    try:
        return _summarize(output)
    except BaseException as exc:
        atomic_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise


def _summarize(output):
    output = Path(output)
    manifest = read_manifest(output / "manifest.json")
    results = [verify_run(output / f"{method}-{seed}", manifest, method, seed)
               for seed in manifest["config"]["seeds"] for method in METHODS]
    if len({digest(r["data_hashes"]) for r in results}) != 1:
        raise ValueError("Data differs between runs")
    provenances = [json.loads((output / f"{r['method']}-{r['seed']}" / "provenance.json").read_text())
                   for r in results]
    if len({digest({k: p[k] for k in ("packages", "gpu", "cuda")}) for p in provenances}) != 1:
        raise ValueError("Hardware/software differs between runs")
    summary = {"manifest_id": manifest["id"], "runs": results, "aggregates": {}, "paired": []}
    metrics = ("validation_macro_f1", "test_macro_f1", "step_seconds_mean",
               "steady_step_seconds_median", "peak_allocated_bytes", "peak_reserved_bytes",
               "gradient_anomaly_steps")
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        group = {}
        for key in metrics:
            values = [r[key] for r in rows if r[key] is not None]
            group[key] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                          "std": statistics.stdev(values) if len(values) > 1 else None}
        times = [r["time_to_target_seconds"] for r in rows if r["target_reached"]]
        group["time_to_target"] = {"reached": len(times), "total": len(rows),
                                   "reached_only_mean_seconds": statistics.mean(times) if times else None}
        summary["aggregates"][method] = group
    for seed in manifest["config"]["seeds"]:
        a, b = [next(r for r in results if r["method"] == m and r["seed"] == seed) for m in METHODS]
        summary["paired"].append({"seed": seed, "test_f1_delta_simplex_minus_lora":
                                   b["test_macro_f1"] - a["test_macro_f1"]})
    atomic_json(output / "summary.json", summary)
    lines = ["# LoRA r=8 vs SimplexLoRA", "",
             "Three paired seeds; mean ± sample standard deviation. No significance claim.",
             "Time-to-target includes rank search and validation, excludes loading/tokenization; unreached runs are censored.",
             "", "| Method | Test macro-F1 | Step mean (s) | Peak allocated GiB | Target reached |",
             "|---|---:|---:|---:|---:|"]
    for method, a in summary["aggregates"].items():
        fmt = lambda key: f"{a[key]['mean']:.4f} ± {a[key]['std']:.4f}"
        lines.append(f"| {method} | {fmt('test_macro_f1')} | {fmt('step_seconds_mean')} | "
                     f"{a['peak_allocated_bytes']['mean'] / 2**30:.3f} | "
                     f"{a['time_to_target']['reached']}/3 |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    atomic_json(output / "status.json", {"status": "passed", "manifest_id": manifest["id"],
                "summary_sha256": file_hash(output / "summary.json"),
                "markdown_sha256": file_hash(output / "summary.md")})
    return summary


def source_identity():
    root = Path(__file__).resolve().parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


def resolve(config):
    from huggingface_hub import HfApi
    c = dict(config)
    api = HfApi()
    c["model_revision"] = api.model_info(c["model"], revision=c["model_revision"]).sha
    c["dataset_revision"] = api.dataset_info(c["dataset"], revision=c["dataset_revision"]).sha
    if not all(re.fullmatch(r"[0-9a-f]{40}", c[k]) for k in ("model_revision", "dataset_revision")):
        raise ValueError("Failed to resolve immutable Hub revisions")
    return c


def matrix(args):
    import fcntl
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".matrix.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another matrix process owns this output directory") from None
        return _matrix(args)


def _matrix(args):
    c = json.loads(Path(args.config).read_text())
    validate_config(c)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output exists; use --resume to validate and skip completed runs")
        manifest = read_manifest(manifest_path)
        if manifest["requested_config"] != c or manifest["source_hash"] != source_identity():
            raise ValueError("Config/source changed: use a new output directory")
    else:
        manifest = {"requested_config": c, "config": resolve(c), "source_hash": source_identity(),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        manifest["id"] = digest(manifest)
        atomic_json(manifest_path, manifest)
    atomic_json(output / "status.json", {"status": "running", "manifest_id": manifest["id"]})
    # Fail before six paid experiments if the actual quantized adapter path is broken.
    try:
        subprocess.run([sys.executable, "-m", "simplexbench.compat", "--cuda",
                        "--output", str(output / "compatibility.json")], check=True)
        for i, seed in enumerate(c["seeds"]):
            # Alternate ordering to reduce systematic thermal/cache ordering bias.
            for method in (METHODS if i % 2 == 0 else METHODS[::-1]):
                directory = output / f"{method}-{seed}"
                if directory.exists():
                    try:
                        verify_run(directory, manifest, method, seed)
                        continue
                    except (ValueError, KeyError, OSError):
                        # Retain failed evidence; restart this run from step zero.
                        archive = output / "incomplete"
                        archive.mkdir(exist_ok=True)
                        directory.rename(archive / f"{directory.name}-{time.time_ns()}")
                subprocess.run([sys.executable, "-m", "simplexbench.cli", "worker",
                                "--manifest", str(manifest_path), "--method", method,
                                "--seed", str(seed), "--output", str(directory)], check=True)
                verify_run(directory, manifest, method, seed)
        summarize(output)
    except BaseException as exc:
        atomic_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("matrix")
    p.add_argument("--config", default="configs/benchmark.json")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p = commands.add_parser("worker")
    p.add_argument("--manifest", required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("summarize")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "matrix":
        matrix(args)
    elif args.command == "summarize":
        summarize(args.output)
    else:
        from .runner import train
        manifest = read_manifest(args.manifest)
        validate_config(manifest["config"])
        if args.seed not in manifest["config"]["seeds"]:
            raise ValueError("Seed not in manifest")
        train(manifest["config"], args.method, args.seed, args.output, manifest["id"])


if __name__ == "__main__":
    main()
