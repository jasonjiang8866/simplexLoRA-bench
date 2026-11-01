"""Bounded RunPod API access. Account credentials never enter the Pod."""
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "https://api.runpod.io/v2"
REPOSITORY = "jasonjiang8866/simplexLoRA-bench"
GLOBAL_NAME = "spt-1436489970-step90-inputs-v1"
MIG = "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"
IMAGE = "ghcr.io/jasonjiang8866/simplexlora-bench@sha256:cb5ce33250b8ced35892adbbcdc8c531eee67d5bab99192907a8fba4d701d342"
SOURCE = "4c9eaac04f2925e3bc51ac415cba8806a8b63ccb9a76a2cae1d5683ec141aaaa"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request(url, *, token, method="GET", body=None, limit=8_000_000):
    req = urllib.request.Request(url, method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                 "User-Agent": "simplexlora-bench-actions/0.1"},
        data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response:
            data = response.read(limit + 1)
        if len(data) > limit:
            raise RuntimeError("Response exceeds limit")
        return json.loads(data) if data else None
    except urllib.error.HTTPError as exc:
        # Only fixed diagnostic categories leave this handler, never provider text.
        raw = exc.read(16000).decode(errors="replace").lower()
        categories = [label for word, label in (
            ("capacity", "capacity"), ("stock", "capacity"), ("disk", "disk"),
            ("cpu", "cpu"), ("vcpu", "vcpu"), ("image", "image"),
            ("global", "global-volume"), ("mount", "mount"), ("port", "port"),
            ("entrypoint", "entrypoint"), ("command", "command"), ("argument", "argument"),
            ("not supported", "unsupported"), ("minimum", "minimum"), ("maximum", "maximum"),
            ("unavailable", "unavailable"), ("not found", "not-found")) if word in raw]
        raise RuntimeError(f"Remote HTTP {exc.code}; categories={','.join(sorted(set(categories))) or 'unclassified'}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Remote request timeout/unavailable; outcome may be unknown") from None


def finite(value, minimum=0):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= minimum


class Runpod:
    def __init__(self):
        self.key = os.environ.get("RUNPOD_API_KEY", "")
        if not self.key:
            raise RuntimeError("RUNPOD_API_KEY secret is missing")

    def call(self, path, method="GET", body=None):
        return request(API + path, token=self.key, method=method, body=body)

    def listing(self, path, key):
        items, seen = [], set()
        base = path
        while True:
            response = self.call(path)
            if not isinstance(response.get(key), list):
                raise RuntimeError("Invalid inventory")
            items.extend(response[key])
            page = response.get("pagination", {})
            if page.get("hasNextPage") is False:
                return items
            cursor = page.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise RuntimeError("Incomplete inventory")
            seen.add(cursor)
            path = base + "?" + urllib.parse.urlencode({"cursor": cursor})


def identity():
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise RuntimeError("Unexpected Actions repository")
    run, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not run.isdigit() or not attempt.isdigit():
        raise RuntimeError("Invalid Actions identity")
    return run + "-" + attempt


def budget(api):
    now = datetime.now(timezone.utc)
    result = request("https://api.runpod.io/graphql", token=api.key, method="POST", body={
        "query": "query SmokeBudget($start: DateTime!, $end: DateTime!) { myself { clientBalance currentSpendPerHr dailyCharges(startDate: $start, endDate: $end) { amount } } }",
        "variables": {"start": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(), "end": now.isoformat()}})
    if result.get("errors"):
        raise RuntimeError("Budget query failed")
    account = result.get("data", {}).get("myself") or {}
    balance, spend, charges = account.get("clientBalance"), account.get("currentSpendPerHr"), account.get("dailyCharges")
    if not finite(balance) or not finite(spend) or not isinstance(charges, list):
        raise RuntimeError("Missing budget information")
    amounts = [c.get("amount") for c in charges if isinstance(c, dict)]
    if len(amounts) != len(charges) or not all(finite(x, -math.inf) for x in amounts):
        raise RuntimeError("Invalid billing history")
    if balance < 1 or sum(abs(x) for x in amounts) + spend + 1 > 500:
        raise RuntimeError("Insufficient $1 reserve or $500 monthly budget headroom")
    return {"reserve_usd": 1, "monthly_budget_usd": 500, "balance_sufficient": True,
            "month_headroom_sufficient": True, "month_basis": "UTC", "checked_utc": now.isoformat(),
            "scope": "reported charges; billing may lag; storage/request fees additional"}


def preflight(api, *, require_gpu=True):
    identity()
    volumes = [v for v in api.listing("/global-volumes", "globalVolumes") if v.get("name") == GLOBAL_NAME]
    if len(volumes) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", volumes[0].get("id", "")):
        raise RuntimeError("Existing Global Volume is missing or ambiguous; none created")
    pods = api.listing("/pods", "pods")
    if any(p.get("env", {}).get("SIMPLEX_MANAGED_BY") == REPOSITORY for p in pods):
        raise RuntimeError("A managed Simplex Pod already exists; inspect before launch")
    gpu = api.call("/catalog/gpus/" + urllib.parse.quote(MIG, safe="") +
                   "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8")
    rate = gpu.get("price", {}).get("secure")
    if not finite(rate, 0.001) or rate > .69:
        raise RuntimeError("MIG catalog rate missing or above $0.69/hour")
    regions = sorted(d["id"] for d in gpu.get("dataCenters", [])
                     if d.get("availability") in {"LOW", "MEDIUM", "HIGH"})
    if require_gpu and not regions:
        raise RuntimeError("No MIG capacity")
    cpu = cpu_placement(api)
    return {"status": "passed", "read_only": True, "image": IMAGE,
            "global_volume": {"id": volumes[0]["id"], "name": GLOBAL_NAME},
            "gpu": {"id": MIG, "rate": rate, "regions": regions}, "cpu": cpu,
            "other_pod_count": len(pods), "budget": budget(api)}


def cpu_placement(api, exclude=None):
    choices = []
    catalog = api.call("/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=2")
    for cpu in catalog["cpus"]:
        rate = cpu.get("price", {}).get("securePerVcpu")
        if not finite(rate, .0001) or rate * 2 > .20 or not cpu["vcpu"]["min"] <= 2 <= cpu["vcpu"]["max"]:
            continue
        for dc in cpu.get("dataCenters", []):
            if dc.get("availability") in {"LOW", "MEDIUM", "HIGH"} and dc["id"] != exclude:
                choices.append((rate * 2, cpu["id"], dc["id"]))
    if not choices:
        raise RuntimeError("No CPU capacity within $0.20/hour in a different data center")
    rate, flavor, region = min(choices)
    return {"id": flavor, "region": region, "rate": rate}
