"""Strict transport-level identity checks for the nested clipping matrix.

Deep tensor, initialization, gradient and batch verification is performed by
summarize_ablation on both writer and independent CPU reader.
"""


def validate_summary(summary, config):
    methods = ['lora', 'simplex', 'dora']
    strategies = ['global', 'grouped']
    seeds = [17, 42, 123]
    if (config.get('methods') != methods or config.get('seeds') != seeds or
            config.get('clipping_strategies') != strategies):
        raise ValueError('Unexpected clipping profile')
    expected = {(strategy, method, seed) for strategy in strategies
                for method in methods for seed in seeds}
    runs = summary.get('runs', [])
    if (len(runs) != 18 or {(r.get('clipping_strategy'), r.get('method'), r.get('seed'))
                           for r in runs} != expected):
        raise ValueError('Incomplete clipping matrix')
    if any(r.get('steps') != config['steps'] for r in runs):
        raise ValueError('Incomplete clipping steps')
    pairs = summary.get('pairs', [])
    if (len(pairs) != 9 or {(p.get('method'), p.get('seed')) for p in pairs} !=
            {(m, s) for m in methods for s in seeds} or
            any(p.get('delta_direction') != 'grouped-minus-global' for p in pairs)):
        raise ValueError('Incomplete clipping pairs')
    order = summary.get('execution_order', [])
    expected_order = []
    for i, seed in enumerate(seeds):
        for j, method in enumerate(methods[i:] + methods[:i]):
            arm_order = strategies if (i * 3 + j) % 2 == 0 else strategies[::-1]
            expected_order.extend({'method': method, 'seed': seed, 'strategy': s}
                                  for s in arm_order)
    if order != expected_order:
        raise ValueError('Clipping execution order mismatch')
    if set(summary.get('measurements', {})) != {f'{s}/{m}-{seed}' for s, m, seed in expected}:
        raise ValueError('Incomplete clipping telemetry')
    if (summary.get('cuda_int8_gate_passed') is not True or
            summary.get('experiment') != 'clipping-optimization-ablation-not-paper-replication'):
        raise ValueError('Missing clipping CUDA evidence')
