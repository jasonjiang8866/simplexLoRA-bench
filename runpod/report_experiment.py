"""Reproducible, stdlib-only analysis of completed experiment archives.

Reads files without extracting the tar. Reports censored targets as null.
Does not turn a gradient-spike count or three seeds into a significance claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def describe(xs):
    return {'n': len(xs), 'min': min(xs), 'max': max(xs),
            'mean': statistics.mean(xs),
            'std': statistics.stdev(xs) if len(xs) > 1 else None}


def report(archive):
    with tarfile.open(archive, 'r:') as tar:
        def read(name):
            item = tar.getmember(name)
            if not item.isfile() or item.size > 64 * 1024**2:
                raise ValueError('Invalid evidence member')
            return tar.extractfile(item).read()
        manifest = json.loads(read('manifest.json'))
        summary = json.loads(read('summary.json'))
        expected = manifest['config']['steps']
        steps, provenance, runs = {}, {}, []
        for result in summary['runs']:
            key = f"{result['method']}-{result['seed']}"
            rows = [json.loads(line) for line in read(key + '/steps.jsonl').splitlines()]
            if [r['step'] for r in rows] != list(range(1, expected + 1)):
                raise ValueError('Incomplete step sequence')
            steps[key] = rows
            provenance[key] = json.loads(read(key + '/provenance.json'))
            grads = [r['gradients'] for r in rows]
            obs = [r['observations'] for r in rows]
            fractions = [g['group_norms']['head']**2 / g['norm']**2 for g in grads if g['norm']]
            ranks = json.loads(read(key + '/ranks.json'))
            runs.append({**result, 'gradient_summary': {
                'pre_clip_norm': describe([g['norm'] for g in grads]),
                'post_clip_norm': describe([g['post_clip']['norm'] for g in grads]),
                'clip_coefficient': describe([g['clip_coefficient'] for g in grads]),
                'head_squared_norm_fraction': describe(fractions),
                'group_norms': {k: describe([g['group_norms'][k] for g in grads])
                                for k in grads[0]['group_norms']},
                'spike_steps': sum(g['spike'] for g in grads),
                'nonfinite_steps': sum(bool(g['nonfinite']) for g in grads),
                'missing_steps': sum(bool(g['missing']) for g in grads)},
                'observations': {
                    'loss': describe([r['loss'] for r in rows]),
                    'loss_first_10_mean': statistics.mean(r['loss'] for r in rows[:10]),
                    'loss_last_10_mean': statistics.mean(r['loss'] for r in rows[-10:]),
                    'hidden_norm': describe([o['pooled_hidden_norm_mean'] for o in obs]),
                    'logit_abs_max': max(o['logit_abs_max'] for o in obs),
                    'head_gradient_relative_error_max': max(o['analytic_head_grad_relative_error'] for o in obs)},
                'rank_events': [{'step': r['step'], 'total_rank': sum(r['ranks'].values()),
                    'min_rank': min(r['ranks'].values()), 'max_rank': max(r['ranks'].values()),
                    'rank_counts': {str(v): list(r['ranks'].values()).count(v) for v in sorted(set(r['ranks'].values()))},
                    'adapter_parameters': r['adapter_parameters'], 'budget': r['budget']}
                    for r in ranks],
                'evaluations': json.loads(read(key + '/evaluations.json')),
                'head_initial': provenance[key]['head_initial']})
        paired = []
        for seed in (17, 42, 123):
            a, b = [steps[f'{m}-{seed}'] for m in ('lora', 'simplex')]
            if len(a) != len(b):
                raise ValueError('Unpaired lengths')
            paired.append({'seed': seed,
                'all_batch_hashes_equal': all(x['observations']['batch_sha256'] == y['observations']['batch_sha256'] for x,y in zip(a,b)),
                'initial_head_hash_equal': provenance[f'lora-{seed}']['head_initial']['sha256'] == provenance[f'simplex-{seed}']['head_initial']['sha256'],
                'initial_loss_delta': b[0]['loss'] - a[0]['loss'],
                'initial_norm_delta': b[0]['gradients']['norm'] - a[0]['gradients']['norm']})
        checks = {
            'six_runs': len(runs) == 6 and {(r['method'],r['seed']) for r in runs} == {(m,s) for m in ('lora','simplex') for s in (17,42,123)},
            'paired_batches_and_initial_heads': all(p['all_batch_hashes_equal'] and p['initial_head_hash_equal'] for p in paired),
            'finite_present_gradients': all(r['gradient_summary']['missing_steps'] == r['gradient_summary']['nonfinite_steps'] == 0 for r in runs),
            'clipping_verified': all(r['gradient_summary']['post_clip_norm']['max'] <= 1.00001 for r in runs),
            'analytic_head_gradient_verified': all(r['observations']['head_gradient_relative_error_max'] <= 1e-4 for r in runs)}
        deltas = [x['test_f1_delta_simplex_minus_lora'] for x in summary['paired']]
        return {'archive_sha256': hashlib.file_digest(Path(archive).open('rb'), 'sha256').hexdigest(),
                'manifest': manifest, 'checks': checks, 'runs': runs, 'paired_checks': paired,
                'aggregates': summary['aggregates'], 'paired_f1': summary['paired'],
                'paired_f1_delta_summary': describe(deltas)}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True)
    args=p.parse_args();Path(args.output).write_text(json.dumps(report(args.archive),indent=2,allow_nan=False)+'\n')
