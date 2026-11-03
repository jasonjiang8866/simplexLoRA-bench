import copy
import unittest

import experiment
import global_smoke
import provider
from test_global_smoke import metadata, result


class ClippingLifecycleTests(unittest.TestCase):
    def profile(self, mode='benchmark-clipping'):
        return experiment.plan(mode, provider.IMAGE, 'b' * 40,
                               150 if mode.startswith('benchmark') else 45,
                               1.85 if mode.startswith('benchmark') else .65)

    def evidence(self):
        from simplexbench.ablation import paired_order
        p = self.profile()
        p['source'] = provider.SOURCE
        r = result()
        report = r['report']
        report['image_source_sha'] = 'b' * 40
        gate = report['compatibility']
        gate['model_type'] = 'qwen2'
        gate['methods']['dora'] = {**gate['methods']['lora'], 'optimizer_coverage': True,
                                   'reload': {'magnitude_verified': True}}
        prototype = report['summary']['runs'][0]
        order = paired_order(p['config'])
        report['summary'] = {
            'runs': [{**prototype, 'method': x['method'], 'seed': x['seed'],
                      'clipping_strategy': x['strategy'], 'steps': 500,
                      'checkpoint_reload': {'passed': True, 'magnitude_verified': True},
                      'magnitude_final': {'delta_l2': .001}} for x in order],
            'pairs': [{'method': m, 'seed': s, 'delta_direction': 'grouped-minus-global'}
                      for m in p['methods'] for s in (17, 42, 123)],
            'measurements': {f"{x['strategy']}/{x['method']}-{x['seed']}": {} for x in order},
            'execution_order': order, 'cuda_int8_gate_passed': True,
            'experiment': 'clipping-optimization-ablation-not-paper-replication'}
        return p, r

    def test_profile_and_compute_ceiling(self):
        diagnostic = self.profile('diagnostic-clipping')
        formal = self.profile()
        self.assertEqual(diagnostic['expected_runs'], 18)
        self.assertEqual(diagnostic['steps'], 50)
        self.assertEqual(formal['expected_runs'], 18)
        self.assertEqual(formal['steps'], 500)
        self.assertEqual(formal['max_bytes'], 2 * 1024**3)
        self.assertLess(formal['worst_case_compute_usd'], 1.85)
        with self.assertRaises(ValueError):
            experiment.plan('benchmark-clipping', provider.IMAGE, 'b'*40, 150, 1)

    def test_complete_nested_matrix_and_restore(self):
        p, r = self.evidence()
        global_smoke.validate_result(r, 'restore', metadata(), p)

    def test_missing_duplicate_wrong_policy_short_or_unrestored_rejected(self):
        p, original = self.evidence()
        mutations = [
            lambda r: r['report']['summary']['runs'].pop(),
            lambda r: r['report']['summary']['runs'].__setitem__(1, r['report']['summary']['runs'][0]),
            lambda r: r['report']['summary']['runs'][0].update(clipping_strategy='other'),
            lambda r: r['report']['summary']['runs'][0].update(steps=50),
            lambda r: r['report']['summary']['pairs'].pop(),
            lambda r: r['report']['summary']['execution_order'].reverse(),
            lambda r: r['report']['summary']['measurements'].pop('global/lora-17'),
            lambda r: r['report']['summary'].update(cuda_int8_gate_passed=False),
            lambda r: r['report'].update(restored_checksums_verified=False),
            lambda r: r['report']['summary']['runs'][0].update(quantized_linear_count=0),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                r = copy.deepcopy(original)
                mutate(r)
                with self.assertRaises(ValueError):
                    global_smoke.validate_result(r, 'restore', metadata(), p)

    def test_formal_review_binds_policies_config_and_recovery(self):
        from simplexbench.common import digest
        p = self.profile()
        review = {'decision': 'proceed', 'image': p['image'], 'source_hash': p['source'],
                  'diagnostic_run': 'fixture-123', 'methods': p['methods'],
                  'completed_runs': 18, 'diagnostic_steps': 50,
                  'clipping_strategies': ['global', 'grouped'],
                  'diagnostic_config_sha256': digest(self.profile('diagnostic-clipping')['config']),
                  'paired_validation_passed': True, 'cross_pod_restore_verified': True,
                  'cleanup_verified': True}
        experiment.validate_review(review, p)
        for key in review:
            broken = dict(review)
            broken.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.validate_review(broken, p)
        for change in ({'completed_runs': 9}, {'clipping_strategies': ['global']},
                       {'diagnostic_config_sha256': 'a'*64}, {'cleanup_verified': False}):
            with self.assertRaises(ValueError):
                experiment.validate_review({**review, **change}, p)
