"""Render measured experiment tables; no inference from null target times."""
import argparse
import json
from pathlib import Path
import statistics


def render(a):
    runs = a['runs']
    def pm(key, method, scale=1, digits=4):
        xs = [r[key] / scale for r in runs if r['method'] == method]
        return f'{statistics.mean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f}'
    lines = [
        '## Final fixed-budget accuracy', '',
        'Mean ± sample standard deviation across three paired seeds. Test data is',
        'evaluated once at the final budget; no best-validation checkpoint selection.', '',
        '| Method | Validation macro-F1 | Test macro-F1 |', '|---|---:|---:|']
    for method in ('lora', 'simplex'):
        lines.append(f"| {method} | {pm('validation_macro_f1', method)} | {pm('test_macro_f1', method)} |")
    lines += ['', '| Seed | LoRA test F1 | Simplex test F1 | Simplex − LoRA |', '|---|---:|---:|---:|']
    for pair in a['paired_f1']:
        x,y = [next(r for r in runs if r['method'] == m and r['seed'] == pair['seed']) for m in ('lora','simplex')]
        lines.append(f"| {pair['seed']} | {x['test_macro_f1']:.6f} | {y['test_macro_f1']:.6f} | {y['test_macro_f1']-x['test_macro_f1']:+.6f} |")
    d=a['paired_f1_delta_summary']
    lines += ['', f"Paired difference mean ± sample SD: **{d['mean']:+.6f} ± {d['std']:.6f}** (F1 units).",
              'Three seeds do not establish a general significance or method superiority claim.', '',
              '## Time-to-target and censoring', '',
              'Target: validation macro-F1 ≥ 0.85 at evaluations every 25 steps.',
              'Time begins at training, includes validation, monitoring and rank search,',
              'and excludes model/data setup and final test evaluation. Simplex search-stage',
              'evaluations are ineligible because the adapters are subsequently reinitialized.',
              'The milestone is the first observed eligible crossing, not a sustained threshold.', '',
              '| Method | Seed | Reached | First eligible step | Time to target (s) | End of observation (s) |',
              '|---|---:|---|---:|---:|---:|']
    for r in runs:
        times=[x for x in r['evaluations'] if x['target_eligible'] and x['macro_f1']>=r['target_f1']]
        step=times[0]['step'] if times else '—'
        t=f"{r['time_to_target_seconds']:.3f}" if r['target_reached'] else 'null (right-censored)'
        lines.append(f"| {r['method']} | {r['seed']} | {r['target_reached']} | {step} | {t} | {r['censor_seconds']:.3f} |")
    lines.append('')
    for m in ('lora','simplex'):
        selected=[r for r in runs if r['method']==m]
        reached=[r['time_to_target_seconds'] for r in selected if r['target_reached']]
        timing=(f'{statistics.mean(reached):.3f} s' if reached else 'null')
        if len(reached)>1: timing+=f' ± {statistics.stdev(reached):.3f} s (sample SD)'
        lines.append(f"- {m}: reached {len(reached)}/{len(selected)}; right-censored {len(selected)-len(reached)}; reached-only mean {timing}.")
    lines += ['', 'Unreached times remain null; no zero substitution or mean over censored values.', '',
              '## Runtime and memory', '',
              'Step times synchronize CUDA and include probes/optimizer/rank changes, excluding',
              'validation. Steady median excludes startup/search; P95 below is per-run P95.',
              'Peak memory includes the loaded model, training and evaluation; reserved memory',
              'is allocator reservation, not additional tensor memory.', '',
              '| Method | Step mean (ms) | Steady median (ms) | Per-run P95 (ms) | Peak allocated (GiB) | Peak reserved (GiB) |',
              '|---|---:|---:|---:|---:|---:|']
    for m in ('lora','simplex'):
        lines.append(f"| {m} | {pm('step_seconds_mean',m,.001,3)} | {pm('steady_step_seconds_median',m,.001,3)} | {pm('step_seconds_p95',m,.001,3)} | {pm('peak_allocated_bytes',m,2**30,4)} | {pm('peak_reserved_bytes',m,2**30,4)} |")
    lines += ['', '| Method | Seed | Training + validation (s) | Setup (s) | Final trainable parameters |', '|---|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['method']} | {r['seed']} | {r['train_wall_seconds']:.3f} | {r['setup_seconds']:.3f} | {r['final_trainable_parameters']} |")
    lines += ['', '## Gradient evidence', '',
              'Original norm >100 threshold and clip max_norm=1.0 remain unchanged.',
              'Expected zero A gradients when B is zero are not treated as missing gradients.', '',
              '| Method | Seed | Norm >100 steps | Missing / nonfinite | Pre-clip range | Mean head squared-norm share | Maximum post-clip norm |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        g=r['gradient_summary']; n=g['pre_clip_norm']
        lines.append(f"| {r['method']} | {r['seed']} | {g['spike_steps']}/{r['steps']} | {g['missing_steps']} / {g['nonfinite_steps']} | {n['min']:.3f}–{n['max']:.3f} | {100*g['head_squared_norm_fraction']['mean']:.4f}% | {g['post_clip_norm']['max']:.9f} |")
    err=max(r['observations']['head_gradient_relative_error_max'] for r in runs)
    lines += ['', f'Largest analytic head-gradient relative error: {err:.3e}.',
              'Group norms (head, A, B, importance, other), clipping coefficients, losses,',
              'hidden/logit scales, all pairing checks and full rank weights are retained in JSON.', '',
              '## Rank allocation', '',
              '| Seed | Step | Total rank / 192 | Layer rank counts | Adapter parameters / 344064 |',
              '|---|---:|---:|---|---:|']
    for r in runs:
        if r['method']!='simplex': continue
        for h in r['rank_events']:
            counts=', '.join(f"r{k}: {v}" for k,v in h['rank_counts'].items())
            lines.append(f"| {r['seed']} | {h['step']} | {h['total_rank']} / 192 | {counts} | {h['adapter_parameters']} / {h['budget']} |")
    lines += ['', 'LoRA remains r=8 at all 24 q_proj layers. The classifier has 3584 parameters.',
              'Floor rounding leaves unused adapter capacity; rank search and resets are charged',
              'to the same 500-step budget. The JSON preserves all layer-specific ranks and weights.', '']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('analysis');p.add_argument('--output',required=True)
    args=p.parse_args();Path(args.output).write_text(render(json.loads(Path(args.analysis).read_text())))
