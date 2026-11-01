"""Actions-only smoke + independent cross-region Global Volume restore."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import signal
import time
import urllib.request

from provider import (IMAGE, MIG, REPOSITORY, SOURCE, NoRedirect, Runpod,
                      cpu_placement, finite, identity, preflight, request)

OUTPUT = Path("runpod-results")


class PhaseDeadline(Exception):
    """Never reinterpret a cancellation signal as a retryable HTTP timeout."""


def save(name, value):
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_result(result, phase, meta, profile=None):
    profile = profile or {"mode": "smoke", "steps": 8, "image": IMAGE, "source": SOURCE, "max_bytes": 256 * 1024**2}
    r = result.get("report", {})
    if result.get("status") != "passed":
        raise ValueError("Worker did not pass; see result artifact")
    expected = {"image": profile["image"], "source_hash": profile["source"], "mode": phase,
                "run_id": meta["run_id"], "global_volume_id": meta["volume_id"],
                "pod_id": meta["pod_id"], "controller_sha": meta["controller_sha"]}
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError("Worker identity mismatch")
    if "image_source_sha" in profile and r.get("image_source_sha") != profile["image_source_sha"]:
        raise ValueError("Image commit evidence mismatch")
    gate = r.get("compatibility", {})
    if (gate.get("status"), gate.get("device"), gate.get("bits")) != ("passed", "cuda", 8):
        raise ValueError("Missing CUDA/int8 evidence")
    for method in ("lora", "simplex"):
        if any(gate.get("methods", {}).get(method, {}).get(k) is not True
               for k in ("forward_backward", "frozen_base", "save_reload")):
            raise ValueError("Missing real PEFT checks")
    runs = r.get("summary", {}).get("runs", [])
    if len(runs) != 6 or {(x.get("method"), x.get("seed")) for x in runs} != {
            (m, s) for m in ("lora", "simplex") for s in (17, 42, 123)}:
        raise ValueError("Incomplete matrix")
    for run in runs:
        if run.get("status") != "passed" or run.get("steps") != profile["steps"] or run.get("quantized_linear_count", 0) <= 0:
            raise ValueError("Invalid smoke run")
        for key in ("test_macro_f1", "validation_macro_f1"):
            if not finite(run.get(key)) or run[key] > 1:
                raise ValueError("Invalid F1")
        for key in ("peak_allocated_bytes", "step_seconds_mean", "train_wall_seconds"):
            if not finite(run.get(key), .000001):
                raise ValueError("Missing performance measurement")
        if run.get("target_reached") != (run.get("time_to_target_seconds") is not None):
            raise ValueError("Invalid time-to-target censoring")
    if not re.fullmatch(r"[a-f0-9]{64}", r.get("snapshot_sha256", "")) or r.get("file_count", 0) < 48:
        raise ValueError("Missing persisted result evidence")
    bundle = r.get("bundle", {})
    if not isinstance(bundle.get("bytes"), int) or not 0 < bundle["bytes"] <= profile["max_bytes"] or not re.fullmatch(r"[a-f0-9]{64}", bundle.get("sha256", "")):
        raise ValueError("Invalid archive descriptor")
    if phase == "restore" and (r.get("restored_checksums_verified") is not True or
                               r.get("summary_revalidated") is not True or
                               r.get("writer_pod_id") != meta["writer_pod_id"]):
        raise ValueError("Missing independent restore validation")


def cleanup(api):
    """Only this Actions attempt's exact recorded resources; retain Global Volume."""
    identity()
    for path in sorted(OUTPUT.glob("provision-*.json")):
        meta = json.loads(path.read_text())
        if meta.get("owner_run_id", meta["run_id"]) != "simplex-" + identity():
            raise RuntimeError("Cleanup identity mismatch")
        matches = [p for p in api.listing("/pods", "pods") if p.get("name") == meta["name"]]
        if len(matches) > 1:
            raise RuntimeError("Ambiguous Pod identity")
        if not matches and meta.get("cleanup_confirmed"):
            continue
        for pod in matches:
            mounts = pod.get("mounts") or {}
            if (pod.get("env", {}).get("SIMPLEX_MANAGED_BY") != REPOSITORY or
                    pod.get("env", {}).get("BENCH_RUN_ID") != meta["run_id"] or
                    mounts.get("global") != [{"volumeId": meta["volume_id"], "path": "/global"}] or
                    mounts.get("network") or mounts.get("persistent") or
                    not re.fullmatch(r"[A-Za-z0-9_-]+", pod.get("id", "")) or
                    (meta.get("pod_id") and pod["id"] != meta["pod_id"])):
                raise RuntimeError("Refusing cleanup of mismatched resource")
            api.call("/pods/" + pod["id"], "DELETE")
        for _ in range(6):
            if not any(p.get("name") == meta["name"] for p in api.listing("/pods", "pods")):
                meta["cleanup_confirmed"] = True
                meta["cleanup_at_epoch"] = time.time()
                elapsed = meta["cleanup_at_epoch"] - meta["started_epoch"]
                meta["elapsed_seconds"] = elapsed
                meta["compute_cost_estimate_usd"] = elapsed * meta.get("rate", 0) / 3600
                save(path.name, meta)
                print("Compute cleanup confirmed: " + meta["name"], flush=True)
                break
            time.sleep(5)
        else:
            raise RuntimeError("Compute disappearance not confirmed")


def download(url, token, descriptor, path):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token,
                                             "User-Agent": "simplexlora-bench-actions/0.1"})
    count, h = 0, hashlib.sha256()
    with urllib.request.build_opener(NoRedirect).open(req, timeout=60) as src, path.open("wb") as dst:
        while chunk := src.read(1024**2):
            count += len(chunk)
            if count > descriptor["bytes"]:
                raise ValueError("Oversized result export")
            dst.write(chunk)
            h.update(chunk)
    if count != descriptor["bytes"] or h.hexdigest() != descriptor["sha256"]:
        raise ValueError("Downloaded result archive checksum mismatch")


def launch_phase(api, config, phase, *, prior=None, profile=None):
    profile = profile or {"mode": "smoke", "steps": 8, "image": IMAGE, "source": SOURCE,
                          "gpu_seconds": 2700, "max_bytes": 256 * 1024**2}
    owner_run_id = "simplex-" + identity()
    run_id = prior["run_id"] if phase == "restore" else owner_run_id
    sha = os.environ.get("CONTROLLER_SHA", os.environ["GITHUB_SHA"])
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise RuntimeError("Invalid controller SHA")
    name = owner_run_id + "-" + phase
    token = secrets.token_urlsafe(32)
    print("::add-mask::" + token, flush=True)
    worker = (Path(__file__).with_name("global_worker.py")).read_bytes()
    code = "import base64;exec(compile(base64.b64decode(" + repr(base64.b64encode(worker).decode()) + "),'/tmp/simplex-worker.py','exec'))"
    meta = {"name": name, "run_id": run_id, "phase": phase, "controller_sha": sha,
            "owner_run_id": owner_run_id,
            "volume_id": config["global_volume"]["id"], "image": profile["image"],
            "profile": profile,
            "worker_sha256": hashlib.sha256(worker).hexdigest(), "started_epoch": time.time()}
    body = {"name": name, "image": profile["image"], "cloud": "SECURE", "disk": 40,
            "mounts": {"global": [{"volumeId": meta["volume_id"], "path": "/global"}]},
            "startSsh": False, "ports": ["8888/http"],
            "entrypoint": ["/bin/bash", "-lc"], "cmd": ["exec /opt/bench-venv/bin/python -u -c " + shlex.quote(code)],
            "env": {"SIMPLEX_MANAGED_BY": REPOSITORY, "SIMPLEX_RESULT_TOKEN": token,
                    "SIMPLEX_PHASE": phase, "SIMPLEX_IMAGE": profile["image"], "SIMPLEX_SOURCE_HASH": profile["source"],
                    "SIMPLEX_GLOBAL_VOLUME_ID": meta["volume_id"], "SIMPLEX_CONTROLLER_SHA": sha,
                    "BENCH_MODE": profile["mode"], "BENCH_RUN_ID": run_id,
                    "BENCH_TIMEOUT_SECONDS": str(profile["gpu_seconds"] - 400),
                    "SIMPLEX_MAX_BYTES": str(profile["max_bytes"]),
                    "SIMPLEX_IMAGE_SOURCE_SHA": profile.get("image_source_sha", ""),
                    "HF_HOME": "/workspace/huggingface", "PYTHONUNBUFFERED": "1"}}
    if phase == "smoke":
        body.update(gpu={"id": MIG, "count": 1, "minRamPerGpu": 16,
                         "minVcpuCountPerGpu": 4, "minCudaVersion": "12.8"},
                    dataCenterIds=config["gpu"]["regions"])
        cap, deadline = .69, time.time() + profile["gpu_seconds"]
    else:
        cpu = cpu_placement(api, exclude=prior["data_center"])
        # Small CPU flavors reject the GPU's 40 GB writable disk request.
        # Restore stages only the bounded completed-result archive and its files.
        body.update(cpu={"id": cpu["id"], "vcpuCount": 2}, disk=10,
                    dataCenterIds=[cpu["region"]])
        body["env"]["SIMPLEX_SNAPSHOT_SHA256"] = prior["report"]["snapshot_sha256"]
        meta["writer_pod_id"] = prior["pod_id"]
        meta["requested_cpu"] = cpu
        cap, deadline = .20, time.time() + 900
    save("provision-" + phase + ".json", meta)

    def cancelled(*_):
        raise PhaseDeadline("Phase deadline or cancellation")

    previous = {s: signal.signal(s, cancelled) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.alarm(max(1, int(deadline - time.time())))
    try:
        # Exactly one create attempt; even an ambiguous timeout enters name-based cleanup.
        pod = api.call("/pods", "POST", body)
        pod_id = pod["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
            raise ValueError("Invalid Pod ID")
        meta["pod_id"] = pod_id
        save("provision-" + phase + ".json", meta)
        detail = api.call("/pods/" + pod_id)
        rate = detail.get("cost")
        if not finite(rate, .0001) or rate > cap:
            raise ValueError("Actual rate is outside cap")
        if phase == "smoke" and detail.get("gpu", {}).get("id") != MIG:
            raise ValueError("Incorrect allocated GPU")
        if phase == "restore" and (not detail.get("cpu") or detail.get("dataCenterId") == prior["data_center"]):
            raise ValueError("Restore must use a CPU in another data center")
        meta.update(rate=rate, data_center=detail["dataCenterId"])
        save("provision-" + phase + ".json", meta)
        print(f"Created {phase} Pod {pod_id}: {meta['data_center']}, ${rate}/hour", flush=True)
        url = f"https://{pod_id}-8888.proxy.runpod.net"
        last = None
        while time.time() < deadline - 60:
            try:
                result = request(url + "/result", token=token)
            except RuntimeError:
                time.sleep(15)
                continue
            if result.get("phase") != last:
                last = result.get("phase")
                print("Worker phase: " + str(last), flush=True)
            if result.get("status") in {"passed", "failed"}:
                save("result-" + phase + ".json", result)
                validate_result(result, phase, meta, profile)
                if phase == "smoke":
                    download(url + "/archive", token, result["report"]["bundle"], OUTPUT / "smoke-evidence.tar")
                else:
                    for k in ("snapshot_sha256", "bundle", "file_count", "summary"):
                        if result["report"][k] != prior["report"][k]:
                            raise ValueError("Restored evidence differs from writer")
                return {**meta, "report": result["report"]}
            time.sleep(15)
        raise TimeoutError("Worker polling deadline")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        cleanup(api)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "smoke", "restore", "cleanup"])
    mode = parser.parse_args().mode
    api = Runpod()
    if mode == "cleanup":
        cleanup(api)
        return
    config = preflight(api, require_gpu=mode != "restore")
    save("preflight.json", config)
    print(json.dumps(config, indent=2), flush=True)
    if mode == "preflight":
        return
    if os.environ.get("GITHUB_REF") != "refs/heads/main" or os.environ.get("CONFIRM_PAID_SMOKE") != "true":
        raise ValueError("Paid smoke requires explicit main workflow dispatch")
    if mode == "restore":
        previous_run = os.environ.get("RESTORE_FROM_RUN", "")
        if not re.fullmatch(r"[0-9]+", previous_run):
            raise ValueError("Invalid prior Actions run")
        previous = Path("restore-input")
        meta = json.loads((previous / "provision-smoke.json").read_text())
        result = json.loads((previous / "result-smoke.json").read_text())
        if (meta.get("run_id") != "simplex-" + previous_run + "-1" or
                meta.get("cleanup_confirmed") is not True or
                meta.get("volume_id") != config["global_volume"]["id"]):
            raise ValueError("Prior smoke ownership/cleanup/volume mismatch")
        validate_result(result, "smoke", meta)
        bundle = previous / "smoke-evidence.tar"
        h = hashlib.sha256()
        with bundle.open("rb") as stream:
            while chunk := stream.read(1024**2):
                h.update(chunk)
        if {"bytes": bundle.stat().st_size, "sha256": h.hexdigest()} != result["report"]["bundle"]:
            raise ValueError("Prior smoke export corrupted")
        first = {**meta, "report": result["report"]}
        shutil.copyfile(previous / "result-smoke.json", OUTPUT / "result-smoke.json")
        shutil.copyfile(previous / "provision-smoke.json", OUTPUT / "source-smoke-provision.json")
        print("Verified prior smoke artifact; renting CPU only", flush=True)
    else:
        first = launch_phase(api, config, "smoke")
    restored = launch_phase(api, config, "restore", prior=first)
    save("acceptance.json", {"status": "passed", "image": IMAGE, "source_hash": SOURCE,
        "run_id": first["run_id"], "global_volume_id": config["global_volume"]["id"],
        "smoke_pod_id": first["pod_id"], "restore_pod_id": restored["pod_id"],
        "writer_region": first["data_center"], "reader_region": restored["data_center"],
        "different_regions": first["data_center"] != restored["data_center"],
        "snapshot_sha256": first["report"]["snapshot_sha256"],
        "file_count": first["report"]["file_count"], "global_path": first["report"]["global_path"],
        "six_smoke_runs_verified": True, "cross_pod_restore_verified": True,
        "performance_benchmark": False})
    print("PASS: six GPU smoke runs and independent cross-region Global Volume restore", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("controller-error.json", {"type": type(exc).__name__, "message": str(exc)})
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1) from None
