import copy
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
import hashlib
import sys
import urllib.error
from unittest.mock import patch

import global_smoke as controller
import global_worker as worker
import provider


def metadata():
    return {"run_id": "simplex-123-1", "pod_id": "newpod", "volume_id": "global1",
            "controller_sha": "b" * 40, "writer_pod_id": "oldpod"}


def result():
    meta = metadata()
    return {"status": "passed", "report": {
        "image": provider.IMAGE, "source_hash": provider.SOURCE, "mode": "restore",
        "run_id": meta["run_id"], "global_volume_id": meta["volume_id"],
        "pod_id": meta["pod_id"], "controller_sha": meta["controller_sha"],
        "writer_pod_id": "oldpod", "restored_checksums_verified": True, "summary_revalidated": True,
        "compatibility": {"status": "passed", "device": "cuda", "bits": 8,
            "methods": {m: {k: True for k in ("forward_backward", "frozen_base", "save_reload")}
                        for m in ("lora", "simplex")}},
        "snapshot_sha256": "a" * 64, "file_count": 50, "bundle": {"bytes": 1024, "sha256": "c" * 64},
        "summary": {"runs": [{"method": m, "seed": s, "status": "passed", "steps": 8,
            "quantized_linear_count": 12, "test_macro_f1": .2, "validation_macro_f1": .3,
            "peak_allocated_bytes": 1024, "step_seconds_mean": .1, "train_wall_seconds": 1,
            "target_reached": False, "time_to_target_seconds": None}
            for m in ("lora", "simplex") for s in (17, 42, 123)]}}}


class EvidenceTests(unittest.TestCase):
    def test_complete_restore(self):
        controller.validate_result(result(), "restore", metadata())

    def test_incomplete_and_wrong_evidence_rejected(self):
        for change in (lambda r: r.update(status="failed"),
                       lambda r: r["report"].update(image="tag:latest"),
                       lambda r: r["report"].update(restored_checksums_verified=False),
                       lambda r: r["report"]["compatibility"].update(device="cpu"),
                       lambda r: r["report"]["compatibility"]["methods"]["simplex"].update(save_reload=False),
                       lambda r: r["report"]["summary"]["runs"].pop(),
                       lambda r: r["report"]["summary"]["runs"][0].update(test_macro_f1=float("nan")),
                       lambda r: r["report"]["summary"]["runs"][0].update(quantized_linear_count=0),
                       lambda r: r["report"].update(writer_pod_id="wrong")):
            with self.subTest(change=change):
                r = result()
                change(r)
                with self.assertRaises(ValueError):
                    controller.validate_result(r, "restore", metadata())

    def test_hostile_archives_and_duplicate_members_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind in ("traversal", "absolute", "symlink", "duplicate"):
                with tarfile.open(root / "test.tar", "w") as tar:
                    item = tarfile.TarInfo({"traversal": "../escape", "absolute": "/escape"}.get(kind, "file"))
                    if kind == "symlink":
                        item.type, item.linkname = tarfile.SYMTYPE, "/etc/passwd"
                    tar.addfile(item, io.BytesIO(b""))
                    if kind == "duplicate":
                        tar.addfile(item, io.BytesIO(b""))
                with self.assertRaises(ValueError):
                    worker.safe_extract(root / "test.tar", root / kind)

    def test_truncated_copy_does_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source", Path(tmp) / "dest"
            source.write_bytes(b"abc")
            expected = worker.spec(source)
            worker.copy_checked(source, dest, expected)
            self.assertEqual(dest.read_bytes(), b"abc")
            source.write_bytes(b"ab")
            with self.assertRaises(ValueError):
                worker.copy_checked(source, dest, expected)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GITHUB_REPOSITORY": provider.REPOSITORY,
            "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "b" * 40})
        self.env.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.output = patch.object(controller, "OUTPUT", Path(self.tmp.name))
        self.output.start()

    def tearDown(self):
        self.output.stop()
        self.tmp.cleanup()
        self.env.stop()

    def test_ambiguous_create_attempted_once_then_cleaned_by_name(self):
        class API:
            calls = []
            def call(self, path, method="GET", body=None):
                self.calls.append((path, method))
                raise RuntimeError("ambiguous timeout")
        api = API()
        with patch.object(controller, "cleanup") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                controller.launch_phase(api, {"global_volume": {"id": "global1"},
                    "gpu": {"regions": ["EU-TEST-1"]}}, "smoke")
            cleanup.assert_called_once_with(api)
        self.assertEqual(api.calls, [("/pods", "POST")])
        self.assertTrue((Path(self.tmp.name) / "provision-smoke.json").exists())

    def test_signal_deadline_is_not_swallowed_as_http_timeout(self):
        class Opener:
            def open(self, *_args, **_kwargs):
                raise controller.PhaseDeadline("cancelled")
        with patch.object(provider.urllib.request, "build_opener", return_value=Opener()):
            with self.assertRaises(controller.PhaseDeadline):
                provider.request("https://api.runpod.io/v2/pods", token="test-only")

    def test_http_client_has_explicit_identity(self):
        seen = []
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self, _):
                return b'{}'
        class Opener:
            def open(self, req, **_):
                seen.append(req)
                return Response()
        with patch.object(provider.urllib.request, "build_opener", return_value=Opener()):
            provider.request("https://example.test/result", token="test-only")
        self.assertEqual(seen[0].get_header("User-agent"), "simplexlora-bench-actions/0.1")

    def test_provider_errors_never_echo_raw_details(self):
        class Opener:
            def open(self, *_args, **_kwargs):
                raise urllib.error.HTTPError('https://example.test', 400, 'bad', {},
                    io.BytesIO(b'{"detail":"CPU disk minimum; private-token-123"}'))
        with patch.object(provider.urllib.request, "build_opener", return_value=Opener()):
            with self.assertRaises(RuntimeError) as caught:
                provider.request("https://example.test", token="test-only")
        self.assertIn("cpu,disk,minimum", str(caught.exception))
        self.assertNotIn("private-token", str(caught.exception))

    def test_restore_reuses_verified_smoke_and_never_launches_gpu(self):
        r = result()
        r['report'].update(mode='smoke', run_id='simplex-555-1', bundle={
            'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()})
        meta = {**metadata(), 'run_id': 'simplex-555-1', 'cleanup_confirmed': True,
                'data_center': 'EU-TEST-1'}
        base = Path(self.tmp.name)
        previous = base / 'restore-input'
        previous.mkdir()
        (previous / 'result-smoke.json').write_text(json.dumps(r))
        (previous / 'provision-smoke.json').write_text(json.dumps(meta))
        (previous / 'smoke-evidence.tar').write_bytes(b'good')
        cwd = Path.cwd()
        try:
            os.chdir(base)
            with patch.dict(os.environ, {'GITHUB_REF': 'refs/heads/main', 'CONFIRM_PAID_SMOKE': 'true',
                                          'RESTORE_FROM_RUN': '555'}), \
                 patch.object(sys, 'argv', ['controller', 'restore']), \
                 patch.object(controller, 'Runpod'), \
                 patch.object(controller, 'preflight', return_value={'global_volume': {'id': 'global1'}}), \
                 patch.object(controller, 'launch_phase', return_value={
                     'pod_id': 'cpu', 'data_center': 'AP-TEST-1', 'report': r['report']}) as launch:
                r['report']['global_path'] = '/global/simplexlora-bench/simplex-555-1'
                (previous / 'result-smoke.json').write_text(json.dumps(r))
                controller.main()
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(launch.call_args.args[2], 'restore')
        finally:
            os.chdir(cwd)

    def test_cleanup_retains_volume_and_unrelated_pods(self):
        meta = {**metadata(), "name": "simplex-123-1-smoke", "started_epoch": 1, "rate": .69}
        controller.save("provision-smoke.json", meta)
        pod = {"id": "newpod", "name": meta["name"],
               "env": {"SIMPLEX_MANAGED_BY": provider.REPOSITORY, "BENCH_RUN_ID": meta["run_id"]},
               "mounts": {"global": [{"volumeId": "global1", "path": "/global"}]}}
        class API:
            calls = []
            pods = [pod, {"id": "other", "name": "other-project"}]
            def listing(self, *_):
                return self.pods
            def call(self, path, method="GET", body=None):
                self.calls.append((path, method))
                self.pods = [self.pods[1]]
        api = API()
        controller.cleanup(api)
        self.assertEqual(api.calls, [("/pods/newpod", "DELETE")])
        self.assertEqual(api.pods[0]["id"], "other")

    def test_cleanup_rejects_unrecognized_mount(self):
        controller.save("provision-smoke.json", {**metadata(), "name": "simplex-123-1-smoke", "started_epoch": 1})
        class API:
            def listing(self, *_):
                return [{"id": "newpod", "name": "simplex-123-1-smoke",
                    "env": {"SIMPLEX_MANAGED_BY": provider.REPOSITORY, "BENCH_RUN_ID": "simplex-123-1"},
                    "mounts": {"global": [{"volumeId": "wrong", "path": "/global"}]}}]
            def call(self, *_):
                raise AssertionError("Must not delete")
        with self.assertRaises(RuntimeError):
            controller.cleanup(API())

    def test_budget_fails_closed_without_returning_account_data(self):
        api = type("API", (), {"key": "test-only"})()
        account = {"clientBalance": 2, "currentSpendPerHr": 0, "dailyCharges": [{"amount": 3}]}
        with patch.object(provider, "request", return_value={"data": {"myself": account}}):
            result = provider.budget(api)
            self.assertNotIn("clientBalance", result)
            account["dailyCharges"] = [{"amount": 501}]
            with self.assertRaises(RuntimeError):
                provider.budget(api)


if __name__ == "__main__":
    unittest.main()
