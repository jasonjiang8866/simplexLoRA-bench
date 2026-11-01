"""Read-only diagnostics for this repository's managed Pods; never echo credentials."""
import json
import signal
import urllib.request

from provider import API, REPOSITORY, NoRedirect, Runpod, request


def main():
    api = Runpod()
    for pod in api.listing("/pods", "pods"):
        if pod.get("env", {}).get("SIMPLEX_MANAGED_BY") != REPOSITORY:
            continue
        detail = api.call("/pods/" + pod["id"])
        print(json.dumps({k: detail.get(k) for k in
              ("id", "name", "status", "dataCenterId", "cost", "gpu", "cpu", "mounts")}), flush=True)
        token = detail.get("env", {}).get("SIMPLEX_RESULT_TOKEN")
        if token:
            try:
                result = request(f"https://{pod['id']}-8888.proxy.runpod.net/result", token=token)
                print("Worker: " + json.dumps(result), flush=True)
            except RuntimeError as exc:
                print(str(exc), flush=True)
        secrets = [api.key] + [str(v) for k, v in detail.get("env", {}).items()
            if any(s in k.upper() for s in ("TOKEN", "KEY", "PASSWORD", "SECRET")) and v]
        req = urllib.request.Request(API + "/pods/" + pod["id"] + "/logs?tail=30",
                                     headers={"Authorization": "Bearer " + api.key})
        def timeout(*_):
            raise TimeoutError()
        old = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(20)
        try:
            with urllib.request.build_opener(NoRedirect).open(req, timeout=10) as stream:
                for index, raw in enumerate(stream):
                    line = raw.decode(errors="replace")
                    for value in secrets:
                        line = line.replace(value, "[REDACTED]")
                    if line.startswith("data:"):
                        print(line[:3000].rstrip(), flush=True)
                    if index > 200:
                        break
        except Exception as exc:
            print("Log read ended: " + type(exc).__name__, flush=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


if __name__ == "__main__":
    main()
