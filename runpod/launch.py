"""RunPod REST v2 launcher. Dry run by default; --launch creates one paid Pod."""
import argparse
import json
import os
import re
import urllib.error
import urllib.request

API = "https://api.runpod.io/v2/pods"


def payload(args):
    if not re.fullmatch(r"[a-zA-Z0-9./_-]+@sha256:[a-f0-9]{64}", args.image):
        raise ValueError("Use a published immutable image@sha256:digest")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Invalid run ID")
    if args.timeout < 1:
        raise ValueError("Timeout must be positive")
    return {
        "name": args.run_id, "image": args.image, "cloud": "SECURE",
        "gpu": {"id": args.gpu, "count": 1, "minRamPerGpu": 32},
        "dataCenterIds": [args.data_center], "disk": 30, "startSsh": False,
        "mounts": {"network": [{"volumeId": args.network_volume, "path": "/workspace"}]},
        "env": {"BENCH_MODE": args.mode, "BENCH_RUN_ID": args.run_id,
                "BENCH_TIMEOUT_SECONDS": str(args.timeout)}
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def launch(body):
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("Set RUNPOD_API_KEY locally; do not put it in the payload")
    request = urllib.request.Request(API, data=json.dumps(body).encode(),
                                    headers={"Authorization": f"Bearer {token}",
                                             "Content-Type": "application/json"}, method="POST")
    # Never automatically retry a potentially successful create request.
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"RunPod create returned HTTP {exc.code}; inspect console before retrying") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Create outcome unknown; inspect RunPod console before retrying") from None
    return {"id": result["id"], "status": result.get("status"),
            "console": "https://www.runpod.io/console/pods"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--network-volume", required=True)
    parser.add_argument("--data-center", required=True)
    parser.add_argument("--mode", choices=("smoke", "benchmark"), default="smoke")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    body = payload(args)
    print(json.dumps(launch(body) if args.launch else {"dry_run": True, "url": API, "body": body}, indent=2))


if __name__ == "__main__":
    main()
