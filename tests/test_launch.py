import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "runpod/launch.py"
spec = importlib.util.spec_from_file_location("launch", PATH)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def args():
    return argparse.Namespace(image="ghcr.io/test/bench@sha256:" + "a" * 64,
                              run_id="test-run", gpu="NVIDIA GeForce RTX 5090",
                              data_center="TEST", network_volume="existing-volume",
                              mode="smoke", timeout=60)


def test_v2_payload_and_immutable_image():
    body = launcher.payload(args())
    assert body["gpu"]["count"] == 1
    assert body["mounts"]["network"][0]["path"] == "/workspace"
    assert "RUNPOD_API_KEY" not in body["env"]
    invalid = args()
    invalid.image = "ghcr.io/test/bench:latest"
    with pytest.raises(ValueError):
        launcher.payload(invalid)


def test_dry_run_needs_no_credentials():
    env = dict(os.environ)
    env.pop("RUNPOD_API_KEY", None)
    run = subprocess.run([sys.executable, str(PATH), "--image", args().image, "--run-id", "test",
                          "--network-volume", "test", "--data-center", "test"],
                         env=env, capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)["dry_run"]


def test_launch_never_retries_ambiguous_failure(monkeypatch):
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            raise TimeoutError()
    monkeypatch.setenv("RUNPOD_API_KEY", "test-only")
    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(RuntimeError, match="outcome unknown"):
        launcher.launch(launcher.payload(args()))
    assert len(calls) == 1
