"""Paired clipping experiment, separate from historical method comparisons."""
import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .common import atomic_json, digest, execution_order, file_hash, validate_config
from .cli import read_manifest, resolve, source_identity, summarize, verify_run

STRATEGIES = ("global", "grouped")


def validate_ablation(c):
    validate_config(c)
    if c.get("clipping_strategies") != list(STRATEGIES) or "clipping_strategy" in c:
        raise ValueError("Expected exactly global/grouped clipping strategies")
    if c.get("methods") != ["lora", "simplex", "dora"] or c["seeds"] != [17, 42, 123]:
        raise ValueError("Preserve original methods and paired seeds")


def arm_config(c, strategy):
    return {**{k: v for k, v in c.items() if k != "clipping_strategies"}, "clipping_strategy": strategy}


def paired_order(c):
    return [{**item, "strategy": strategy} for i, item in enumerate(execution_order(c))
            for strategy in (STRATEGIES if i % 2 == 0 else STRATEGIES[::-1])]


def plan(c):
    validate_ablation(c)
    return {"runs": 18, "optimizer_steps": 18 * c["steps"], "steps_per_run": c["steps"],
            "execution_order": paired_order(c), "config_sha256": digest(c),
            "clipping": {"global": {"all": 1.0}, "grouped": {"head": 1.0, "adapter": 1.0}},
            "paid_execution": "not authorized by this plan"}


def read_json(path):
    return json.loads(Path(path).read_text())


def rows_at(path):
    return [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def telemetry(path, method, config):
    rows = rows_at(path)
    search = config["rank_interval"] * config["rank_updates"]
    phases = {"all": rows, "search": rows[:search] if method == "simplex" else [],
              "post_search": rows[search:] if method == "simplex" else rows}
    result = {}
    for phase, selected in phases.items():
        norms = {}
        for group in ("head", "adapter_A", "adapter_B", "magnitude", "importance"):
            # Importance exists only during search; no dilution with post-search zeros.
            active = [r for r in selected if group != "importance" or r["importance"] is not None]
            norms[group] = {position: describe([r["gradients"]["group_norms"][group] if position == "pre"
                else r["gradients"]["post_clip"]["group_norms"][group] for r in active]) for position in ("pre", "post")}
        clip_groups = ("global",) if config["clipping_strategy"] == "global" else ("head", "adapter")
        result[phase] = {"gradient_norms": norms,
            "clip_seconds": describe([r["clip_seconds"] for r in selected]),
            "head_pre_squared_norm_share": describe([r["gradients"]["group_norms"]["head"] ** 2 / r["gradients"]["norm"] ** 2
                for r in selected if r["gradients"]["norm"] > 0]),
            "global_pre_norm": describe([r["gradients"]["norm"] for r in selected]),
            "global_post_norm_max": max((r["gradients"]["post_clip"]["norm"] for r in selected), default=None),
            "clipped_steps": {g: sum(r["gradients"]["clipping"]["groups"][g]["coefficient"] < 1 for r in selected)
                              for g in clip_groups},
            "coefficients": {g: describe([r["gradients"]["clipping"]["groups"][g]["coefficient"] for r in selected])
                             for g in clip_groups}}
    updates = [r["importance"] for r in rows if r["importance"] is not None]
    result["importance_updates"] = {k: describe([u[k] for u in updates]) for k in
        ("gradient_delta_l2", "regularizer_delta_l2", "actual_sgd_delta_l2", "actual_sgd_delta_abs_max")}
    result["importance_max_abs_deviation_from_one_before_reset"] = max(
        (abs(v - 1) for u in updates for v in (u["projected"] if u["projected"] is not None else u["after_sgd"])), default=None)
    result["rank_history"] = read_json(path / "ranks.json")
    result["clip_seconds_total"] = sum(r["clip_seconds"] for r in rows)
    return result


def summarize_ablation(output):
    output = Path(output)
    atomic_json(output / "status.json", {"status": "validating"})
    try:
        return _summarize_ablation(output)
    except BaseException as exc:
        atomic_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise


def _summarize_ablation(output):
    manifest = read_json(output / "ablation.json")
    if manifest["id"] != digest({k: v for k, v in manifest.items() if k != "id"}):
        raise ValueError("Ablation manifest checksum mismatch")
    c = manifest["config"]
    validate_ablation(c)
    if manifest["execution_order"] != paired_order(c):
        raise ValueError("Invalid paired execution order")
    arm_manifests = {s: read_manifest(output / s / "manifest.json") for s in STRATEGIES}
    for s, arm in arm_manifests.items():
        if arm["config"] != arm_config(c, s) or arm["source_hash"] != manifest["source_hash"]:
            raise ValueError("Arm config/source differs beyond clipping")
    # Verify complete runs before using any evidence. No historical run substitution.
    summaries = {s: summarize(output / s) for s in STRATEGIES}
    pairs, measurements, platforms = [], {}, set()
    for item in execution_order(c):
        method, seed = item["method"], item["seed"]
        paths = [output / s / f"{method}-{seed}" for s in STRATEGIES]
        provenances = [read_json(p / "provenance.json") for p in paths]
        for p in provenances:
            platforms.add(digest({k: p[k] for k in ("packages", "gpu", "gpu_uuid", "cuda", "python")}))
        for key in ("initial_trainable_sha256", "initial_frozen_sha256", "head_initial", "data_hashes"):
            if not provenances[0].get(key) or provenances[0][key] != provenances[1].get(key):
                raise ValueError("Unpaired initialization/data: " + key)
        rows = [rows_at(p) for p in paths]
        for strategy, rr in zip(STRATEGIES, rows):
            for row in rr:
                if row["gradients"].get("clipping", {}).get("strategy") != strategy:
                    raise ValueError("Missing clipping telemetry")
                if not math.isfinite(row["clip_seconds"]) or row["clip_seconds"] < 0:
                    raise ValueError("Invalid clipping timing")
                active = method == "simplex" and row["step"] <= c["rank_interval"] * c["rank_updates"]
                if active != (row.get("importance") is not None):
                    raise ValueError("Missing/unexpected importance telemetry")
        signatures = [[(r["observations"]["batch_sha256"], r["lr"]) for r in rr] for rr in rows]
        if signatures[0] != signatures[1]:
            raise ValueError("Unpaired batch order or LR schedule")
        results = [read_json(p / "result.json") for p in paths]
        pair = {"method": method, "seed": seed, "delta_direction": "grouped-minus-global"}
        for key in ("test_macro_f1", "validation_macro_f1", "train_wall_seconds", "step_seconds_mean",
                    "step_seconds_p95", "steady_step_seconds_median", "peak_allocated_bytes", "peak_reserved_bytes"):
            pair[key] = results[1][key] - results[0][key] if all(r[key] is not None for r in results) else None
        pair["step_time_percent_change"] = 100 * (results[1]["step_seconds_mean"] / results[0]["step_seconds_mean"] - 1)
        pair["time_to_target"] = {s: {k: r[k] for k in ("target_reached", "time_to_target_seconds", "censor_seconds")}
                                  for s, r in zip(STRATEGIES, results)}
        pair["time_to_target_delta_seconds"] = (results[1]["time_to_target_seconds"] - results[0]["time_to_target_seconds"]
                                                if all(r["target_reached"] for r in results) else None)
        pair["first_target_step"] = {s: next((e["step"] for e in read_json(path / "evaluations.json")
            if e["target_eligible"] and e["macro_f1"] >= c["target_f1"]), None) for s, path in zip(STRATEGIES, paths)}
        pairs.append(pair)
        for s, path in zip(STRATEGIES, paths):
            measurements[f"{s}/{method}-{seed}"] = telemetry(path, method, arm_config(c, s))
    if len(platforms) != 1:
        raise ValueError("Hardware/software differs across clipping arms")
    metrics = ("test_macro_f1", "validation_macro_f1", "step_seconds_mean", "step_time_percent_change",
               "train_wall_seconds", "time_to_target_delta_seconds")
    gate_path = output / "compatibility.json"
    gate = read_json(gate_path) if gate_path.exists() else {}
    cuda_verified = (gate.get("status") == "passed" and gate.get("device") == "cuda" and gate.get("bits") == 8
                     and gate.get("model_type") == "qwen2" and set(gate.get("methods", {})) == set(c["methods"]))
    summary = {"cuda_int8_gate_passed": cuda_verified, "experiment": "clipping-optimization-ablation-not-paper-replication", "manifest_id": manifest["id"],
        "execution_order": manifest["execution_order"], "runs": [r for s in STRATEGIES for r in summaries[s]["runs"]],
        "pairs": pairs, "measurements": measurements,
        "paired_deltas": {m: {k: describe([p[k] for p in pairs if p["method"] == m and p[k] is not None])
                               for k in metrics} for m in c["methods"]},
        "limitations": ["Three seeds; descriptive paired differences, no general superiority claim.",
            "Grouped total L2 bound is sqrt(2), global bound is 1; clipping geometry and total budget both change.",
            "Instrumentation is identical in both arms; timing includes monitoring.",
            "Paired TTT delta uses both-reached pairs only; inspect censoring records."]}
    atomic_json(output / "summary.json", summary)
    lines = ["# Paired gradient-clipping ablation", "", "Grouped minus global; three paired seeds. Optimization ablation, not direct paper replication.", "",
             "| Method | Test F1 delta (mean ± sample SD) | Step time change (%) | Both reached target |", "|---|---:|---:|---:|"]
    for method, stats in summary["paired_deltas"].items():
        f = stats["test_macro_f1"]
        lines.append(f"| {method} | {f['mean']:+.8f} ± {f['sample_sd']:.8f} | {stats['step_time_percent_change']['mean']:+.3f} | {stats['time_to_target_delta_seconds']['n']}/3 |")
    lines += ["", f"CUDA/int8 gate verified: {cuda_verified}", "", *summary["limitations"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    atomic_json(output / "status.json", {"status": "passed", "manifest_id": manifest["id"],
                "summary_sha256": file_hash(output / "summary.json"), "markdown_sha256": file_hash(output / "summary.md")})
    return summary


def matrix(config, output, resume=False):
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".ablation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = read_json(config)
        validate_ablation(c)
        path = output / "ablation.json"
        if path.exists():
            manifest = read_json(path)
            if (not resume or manifest["requested_config"] != c or manifest["source_hash"] != source_identity()
                    or manifest["id"] != digest({k: v for k, v in manifest.items() if k != "id"})):
                raise ValueError("Existing output: require unchanged config/source and --resume")
        else:
            manifest = {"requested_config": c, "config": resolve(c), "source_hash": source_identity(),
                        "execution_order": paired_order(c), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            manifest["id"] = digest(manifest)
            atomic_json(path, manifest)
        for s in STRATEGIES:
            arm = {"config": arm_config(manifest["config"], s), "source_hash": manifest["source_hash"],
                   "ablation_id": manifest["id"], "execution_order": execution_order(c)}
            arm["id"] = digest(arm)
            root = output / s
            root.mkdir(exist_ok=True)
            if (root / "manifest.json").exists() and read_json(root / "manifest.json") != arm:
                raise ValueError("Arm manifest changed")
            atomic_json(root / "manifest.json", arm)
        atomic_json(output / "status.json", {"status": "running", "manifest_id": manifest["id"]})
        try:
            subprocess.run([sys.executable, "-m", "simplexbench.compat", "--cuda", "--output", str(output / "compatibility.json")], check=True)
            for item in manifest["execution_order"]:
                s, method, seed = item["strategy"], item["method"], item["seed"]
                root = output / s
                arm = read_manifest(root / "manifest.json")
                directory = root / f"{method}-{seed}"
                if directory.exists():
                    # Fail closed on corruption; don't silently replace paired evidence.
                    verify_run(directory, arm, method, seed)
                    continue
                subprocess.run([sys.executable, "-m", "simplexbench.cli", "worker", "--manifest", str(root / "manifest.json"),
                    "--method", method, "--seed", str(seed), "--output", str(directory)], check=True)
                verify_run(directory, arm, method, seed)
            return summarize_ablation(output)
        except BaseException as exc:
            atomic_json(output / "status.json", {"status": "failed", "error": str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "matrix", "summarize"))
    parser.add_argument("--config", default="configs/benchmark-clipping.json")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(read_json(args.config)), indent=2))
    else:
        if not args.output:
            parser.error("--output is required")
        if args.command == "matrix":
            matrix(args.config, args.output, args.resume)
        else:
            summarize_ablation(args.output)


if __name__ == "__main__":
    main()
