"""Summarize immutable step evidence without treating spikes as failed runs."""
import argparse
import json
from pathlib import Path
import statistics
import tarfile


def analyze(path):
    runs = {}
    with tarfile.open(path, 'r:') as tar:
        for member in tar:
            if not member.name.endswith('/steps.jsonl'):
                continue
            if member.size > 32 * 1024**2 or not member.isfile():
                raise ValueError('Unexpected step evidence')
            rows = [json.loads(x) for x in tar.extractfile(member)]
            norm = [x['gradients']['norm'] for x in rows]
            groups = [x['gradients'].get('group_norms', {}) for x in rows]
            fractions = [g['head']**2 / n**2 for g, n in zip(groups, norm) if g and n]
            runs[member.name.split('/')[0]] = {
                'steps': len(rows), 'norm_min': min(norm), 'norm_max': max(norm),
                'spikes': sum(x['gradients']['spike'] for x in rows),
                'nonfinite': sum(bool(x['gradients']['nonfinite']) for x in rows),
                'missing': sum(bool(x['gradients']['missing']) for x in rows),
                'head_squared_norm_fraction_mean': statistics.mean(fractions) if fractions else None,
                'post_clip_norm_max': max((x['gradients'].get('post_clip', {}).get('norm', 0) for x in rows)),
                'mean_step_seconds': statistics.mean(x['seconds'] for x in rows),
                'first': rows[0], 'last': rows[-1],
                'rank_boundaries': [x for x in rows if x['rank_update'] or x['step'] in (11, 21, 31)],
                'analytic_head_relative_error_max': max(x.get('observations', {}).get('analytic_head_grad_relative_error', 0) for x in rows)}
    methods = sorted({key.rsplit('-',1)[0] for key in runs})
    paired = []
    for seed in (17, 42, 123):
        for method in methods:
            if method == 'lora': continue
            a, b = [runs[f'{m}-{seed}']['first'] for m in ('lora', method)]
            paired.append({'seed': seed, 'comparison': method + '-lora', 'first_loss_delta': b['loss'] - a['loss'],
                           'first_norm_delta': b['gradients']['norm'] - a['gradients']['norm'],
                           'first_batch_matches': a.get('observations', {}).get('batch_sha256') == b.get('observations', {}).get('batch_sha256')})
    return {'runs': runs, 'paired': paired,
            'interpretation': 'Observations only. Human review required before full benchmark; norm >100 is not an automatic failure.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('archive'); p.add_argument('--output', required=True)
    args = p.parse_args()
    Path(args.output).write_text(json.dumps(analyze(args.archive), indent=2) + '\n')
