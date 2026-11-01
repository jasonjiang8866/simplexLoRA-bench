"""Test the published image unchanged, then export/restore on a Global Volume.

This stdlib harness is injected by the Actions controller; training code and
dependencies remain those baked into the immutable image. Only completed files
are copied to object storage, with a completion manifest written last.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RESULT = {"status": "running", "phase": "starting"}
TOKEN = os.environ.get("SIMPLEX_RESULT_TOKEN", "")
LOCAL = Path("/workspace")
ARCHIVE = LOCAL / "evidence.tar"
MAX_BYTES = int(os.environ.get("SIMPLEX_MAX_BYTES", 256 * 1024**2))
MODE = os.environ.get("BENCH_MODE", "smoke")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def spec(path):
    return {"bytes": Path(path).stat().st_size, "sha256": sha(path)}


def copy_checked(source, destination, expected):
    if not 0 <= expected["bytes"] <= MAX_BYTES:
        raise ValueError("Copy exceeds bound")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        remaining = expected["bytes"]
        while remaining:
            chunk = src.read(min(4 * 1024**2, remaining))
            if not chunk:
                raise ValueError("Truncated input")
            dst.write(chunk)
            remaining -= len(chunk)
        if src.read(1):
            raise ValueError("Oversized input")
    if spec(destination) != expected:
        raise ValueError("Read-back checksum mismatch")


def safe_extract(archive, destination):
    with tarfile.open(archive, "r:") as tar:
        names, total = set(), 0
        for item in tar:
            p = PurePosixPath(item.name)
            if (not item.isfile() or p.is_absolute() or ".." in p.parts or
                    str(p) != item.name or item.name in names):
                raise ValueError("Unsafe archive member")
            names.add(item.name)
            total += item.size
            if total > MAX_BYTES or len(names) > 1000:
                raise ValueError("Archive exceeds bound")
            target = destination / p
            if target.is_symlink() or any(x.is_symlink() for x in target.parents):
                raise ValueError("Symlink destination rejected")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024**2)


def run(command, log, timeout):
    with log.open("ab") as stream:
        subprocess.run(command, cwd="/opt/bench", stdout=stream, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)


def validate_tree(root):
    sys.path.insert(0, "/opt/bench")
    from simplexbench.cli import summarize
    old_summary = sha(root / "summary.json")
    summary = summarize(root)
    if sha(root / "summary.json") != old_summary:
        raise ValueError("Summary changed after revalidation")
    gate = json.loads((root / "compatibility.json").read_text())
    if gate.get("status") != "passed" or gate.get("bits") != 8 or gate.get("device") != "cuda":
        raise ValueError("Missing real CUDA/int8 gate")
    for method in ("lora", "simplex"):
        if any(gate["methods"][method].get(k) is not True for k in ("forward_backward", "frozen_base", "save_reload")):
            raise ValueError("Incomplete compatibility evidence")
    if {(r["method"], r["seed"]) for r in summary["runs"]} != {
            (m, s) for m in ("lora", "simplex") for s in (17, 42, 123)} or len(summary["runs"]) != 6:
        raise ValueError("Incomplete paired matrix")
    expected = json.loads((Path("/opt/bench/configs") / (MODE + ".json")).read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if MODE != "smoke" and (manifest["config"] != expected or manifest["requested_config"] != expected):
        raise ValueError("Unexpected experiment configuration")
    if any(r["steps"] != expected["steps"] for r in summary["runs"]):
        raise ValueError("Incomplete experiment profile")
    return gate, summary


def train(target, root, report):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expected one CUDA device")
    props = torch.cuda.get_device_properties(0)
    report["gpu"] = {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
                     "total_memory_bytes": props.total_memory, "cuda": torch.version.cuda}
    if not (22 * 1024**3 <= props.total_memory <= 25 * 1024**3 and
            report["gpu"]["capability"] == [12, 0] and "RTX PRO 6000" in props.name):
        raise ValueError("Unexpected MIG hardware")
    RESULT["phase"] = "global-volume-write-read"
    target.mkdir(parents=True, exist_ok=False)
    probe = LOCAL / "probe.bin"
    probe.write_bytes(os.urandom(8 * 1024**2))
    before = time.monotonic()
    copy_checked(probe, target / "probe.bin", spec(probe))
    report["probe"] = {**spec(probe), "write_and_readback_seconds": time.monotonic() - before}
    RESULT["phase"] = "cuda-gate-and-six-smoke-workers"
    # Run the image's original entrypoint, preserving gate-before-workers ordering.
    run(["/bin/bash", "/opt/bench/runpod/start.sh"], LOCAL / "harness.log", int(os.environ.get("BENCH_TIMEOUT_SECONDS", "2300")) + 40)
    report["compatibility"], report["summary"] = validate_tree(root)
    RESULT["phase"] = "publish-completed-results"
    files = {str(p.relative_to(root)): spec(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if sum(x["bytes"] for x in files.values()) > MAX_BYTES // 2:
        raise ValueError("Result tree exceeds export bound")
    with tarfile.open(ARCHIVE, "w") as tar:
        for name in files:
            tar.add(root / name, arcname=name, recursive=False)
    bundle = spec(ARCHIVE)
    copy_checked(ARCHIVE, target / "evidence.tar", bundle)
    manifest = {"format": 1, "run_id": report["run_id"], "image": report["image"],
                "source_hash": report["source_hash"], "global_volume_id": report["global_volume_id"],
                "writer_pod_id": report["pod_id"], "bundle": bundle, "probe": spec(probe), "files": files}
    raw = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    (target / "snapshot.json").write_bytes(raw)  # Completion marker last; no rename/flock.
    if (target / "snapshot.json").read_bytes() != raw:
        raise ValueError("Completion marker read-back failed")
    report.update(bundle=bundle, snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                  file_count=len(files), global_path=str(target))


def restore(target, root, report):
    RESULT["phase"] = "fresh-pod-global-restore"
    raw = (target / "snapshot.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ["SIMPLEX_SNAPSHOT_SHA256"]:
        raise ValueError("Snapshot marker mismatch")
    manifest = json.loads(raw)
    for key in ("run_id", "image", "source_hash", "global_volume_id"):
        if manifest[key] != report[key]:
            raise ValueError("Snapshot identity mismatch")
    if manifest["writer_pod_id"] == report["pod_id"]:
        raise ValueError("Restore must use a different Pod")
    copy_checked(target / "probe.bin", LOCAL / "probe.bin", manifest["probe"])
    copy_checked(target / "evidence.tar", ARCHIVE, manifest["bundle"])
    safe_extract(ARCHIVE, root)
    files = {str(p.relative_to(root)): spec(p) for p in root.rglob("*") if p.is_file()}
    if files != manifest["files"]:
        raise ValueError("Restored file inventory/hash mismatch")
    report["compatibility"], report["summary"] = validate_tree(root)
    report.update(bundle=manifest["bundle"], snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                  file_count=len(files), writer_pod_id=manifest["writer_pod_id"],
                  probe=manifest["probe"], restored_checksums_verified=True,
                  summary_revalidated=True, global_path=str(target))


def work():
    os.environ["PATH"] = "/opt/bench-venv/bin:" + os.environ.get("PATH", "")
    report = {"mode": os.environ["SIMPLEX_PHASE"], "image": os.environ["SIMPLEX_IMAGE"],
              "run_id": os.environ["BENCH_RUN_ID"], "global_volume_id": os.environ["SIMPLEX_GLOBAL_VOLUME_ID"],
              "pod_id": os.environ["RUNPOD_POD_ID"], "profile": MODE, "controller_sha": os.environ["SIMPLEX_CONTROLLER_SHA"]}
    started = time.monotonic()
    try:
        LOCAL.mkdir(exist_ok=True)
        expected_sha = os.environ.get("SIMPLEX_IMAGE_SOURCE_SHA")
        if expected_sha:
            report["image_source_sha"] = Path("/opt/source-sha.txt").read_text().strip()
            if report["image_source_sha"] != expected_sha:
                raise ValueError("Baked image source commit mismatch")
        if not os.path.ismount("/global") or os.path.islink("/global"):
            raise ValueError("Global Volume is not mounted")
        sys.path.insert(0, "/opt/bench")
        from simplexbench.cli import source_identity
        report["source_hash"] = source_identity()
        if report["source_hash"] != os.environ["SIMPLEX_SOURCE_HASH"]:
            raise ValueError("Image source differs from published revision")
        target = Path("/global/simplexlora-bench") / report["run_id"]
        root = LOCAL / "results" / report["run_id"]
        if report["mode"] == "smoke":
            train(target, root, report)
        elif report["mode"] == "restore":
            restore(target, root, report)
        else:
            raise ValueError("Unknown mode")
        report["seconds"] = time.monotonic() - started
        RESULT.update(status="passed", phase="complete", report=report)
    except Exception as exc:
        (LOCAL / "worker-error.log").write_text(traceback.format_exc())
        tail = (LOCAL / "harness.log").read_text(errors="replace")[-24000:] if (LOCAL / "harness.log").exists() else ""
        RESULT.update(status="failed", report=report, error=type(exc).__name__,
                      detail=str(exc), log_tail=tail, traceback=traceback.format_exc())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if not TOKEN or not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + TOKEN):
            self.send_error(401)
            return
        if self.path == "/result":
            data = json.dumps(RESULT, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/archive" and RESULT.get("status") == "passed":
            self.send_response(200)
            self.send_header("Content-Length", str(ARCHIVE.stat().st_size))
            self.end_headers()
            with ARCHIVE.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, 1024**2)
        else:
            self.send_error(404)


if __name__ == "__main__":
    if len(TOKEN) < 32:
        raise SystemExit("Missing result access token")
    threading.Thread(target=work, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 8888), Handler).serve_forever()
