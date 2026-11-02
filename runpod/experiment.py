"""Bounded diagnostic/full matrices using the proven smoke Pod lifecycle.

No paid default. Dry-run records a concrete plan; dispatch must supply a cap.
The old eight-step acceptance remains the default of global_smoke.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import global_smoke as lifecycle
from provider import Runpod, budget, finite, identity, preflight


def plan(mode, image, source_sha, gpu_minutes, max_cost):
    if mode not in {"diagnostic", "benchmark", "diagnostic-dora", "benchmark-dora"}:
        raise ValueError("Invalid experiment mode")
    if not re.fullmatch(r"ghcr.io/jasonjiang8866/simplexlora-bench@sha256:[a-f0-9]{64}", image):
        raise ValueError("An immutable repository image is required")
    if not re.fullmatch(r"[a-f0-9]{40}", source_sha):
        raise ValueError("Image source commit is required")
    limit = 45 if mode.startswith("diagnostic") else 180
    if isinstance(gpu_minutes, bool) or not 15 <= gpu_minutes <= limit:
        raise ValueError("GPU deadline outside bounded profile")
    # Reserve 5m cleanup per phase beyond the training/polling deadline.
    ceiling = .69 * (gpu_minutes + 5) / 60 + .20 * 20 / 60
    if not finite(max_cost, ceiling) or max_cost > 10:
        raise ValueError(f"Cost cap must cover ${ceiling:.4f} compute ceiling and be <=$10")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from simplexbench.cli import source_identity
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs' / f'{mode}.json').read_text())
    return {"mode": mode, "steps": config['steps'], "image": image,
            "methods": config.get("methods", ["lora", "simplex"]), "expected_runs": 3 * len(config.get("methods", ["lora", "simplex"])),
            "source": source_identity(), "image_source_sha": source_sha,
            "gpu_seconds": gpu_minutes * 60, "cpu_seconds": 900,
            "max_bytes": 1024**3, "max_compute_usd": max_cost,
            "worst_case_compute_usd": ceiling, "config": config,
            "storage_request_fees": "additional; existing volume retained"}


def verify_prior(directory, profile, config):
    meta = json.loads((directory / 'provision-smoke.json').read_text())
    result = json.loads((directory / 'result-smoke.json').read_text())
    prior_profile = dict(meta.get('profile', {}))
    for key in ('methods', 'expected_runs'):
        if key not in prior_profile and 'dora' not in profile.get('methods', []):
            prior_profile[key] = profile[key]
    if prior_profile != profile or meta.get('cleanup_confirmed') is not True:
        raise ValueError('Prior profile or GPU cleanup mismatch')
    if meta['volume_id'] != config['global_volume']['id']:
        raise ValueError('Prior volume mismatch')
    lifecycle.validate_result(result, 'smoke', meta, profile)
    path = directory / 'smoke-evidence.tar'
    h = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    if {'bytes': path.stat().st_size, 'sha256': h} != result['report']['bundle']:
        raise ValueError('Prior archive corrupted')
    return {**meta, 'report': result['report']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dry-run', 'run', 'restore', 'cleanup'])
    args = p.parse_args()
    if args.action == 'cleanup':
        lifecycle.cleanup(Runpod())
        return
    profile = plan(os.environ['EXPERIMENT_MODE'], os.environ['EXPERIMENT_IMAGE'],
                   os.environ['IMAGE_SOURCE_SHA'], int(os.environ['GPU_MINUTES']),
                   float(os.environ['MAX_COMPUTE_USD']))
    if os.environ.get('CONTROLLER_SHA') != profile['image_source_sha']:
        raise ValueError('Checkout must be the exact image source revision')
    lifecycle.save('plan.json', profile)
    api = Runpod()
    config = preflight(api, require_gpu=args.action != 'restore')
    if config['global_volume']['id'] != 'cmvwmviz9000007jteci4ajn8':
        raise ValueError('Global Volume identity changed')
    config['budget'] = budget(api, reserve=profile['max_compute_usd'])
    config['image'] = profile['image']
    lifecycle.save('preflight.json', config)
    print(json.dumps({'plan': profile, 'preflight': config}, indent=2), flush=True)
    if args.action == 'dry-run':
        return
    if os.environ.get('CONFIRM_PAID_EXPERIMENT') != 'true':
        raise ValueError('Explicit paid dispatch required')
    if profile['mode'].startswith('benchmark') and args.action != 'restore':
        # Written only after reviewing diagnostic evidence; must match this image.
        review = json.loads(Path('diagnosis-review.json').read_text())
        if (review.get('decision') != 'proceed' or review.get('image') != profile['image'] or
                review.get('source_hash') != profile['source'] or not review.get('diagnostic_run')):
            raise ValueError('Missing reviewed diagnosis for this image/source')
    if profile['mode'] == 'benchmark-dora' and args.action != 'restore':
        if (review.get('methods') != profile['methods'] or review.get('completed_runs') != 9 or
                review.get('diagnostic_steps') != 50):
            raise ValueError('Nine-run DoRA diagnosis review required')
    if args.action == 'restore':
        first = verify_prior(Path('restore-input'), profile, config)
        lifecycle.save('result-smoke.json', {'status': 'passed', 'report': first['report']})
        lifecycle.save('source-smoke-provision.json', {k: v for k, v in first.items() if k != 'report'})
    else:
        first = lifecycle.launch_phase(api, config, 'smoke', profile=profile)
    restored = lifecycle.launch_phase(api, config, 'restore', prior=first, profile=profile)
    lifecycle.save('acceptance.json', {'status': 'passed', 'profile': profile,
        'run_id': first['run_id'], 'global_path': first['report']['global_path'],
        'global_volume_id': first['volume_id'], 'gpu_pod_id': first['pod_id'],
        'cpu_pod_id': restored['pod_id'], 'writer_region': first['data_center'],
        'reader_region': restored['data_center'], 'cross_pod_restore_verified': True,
        'snapshot_sha256': first['report']['snapshot_sha256'],
        'file_count': first['report']['file_count'],
        'performance_benchmark': profile['mode'].startswith('benchmark')})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        lifecycle.save('controller-error.json', {'type': type(exc).__name__, 'message': str(exc)})
        print(f'ERROR: {type(exc).__name__}: {exc}', flush=True)
        raise SystemExit(1) from None
