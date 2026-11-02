import copy
import unittest
from unittest.mock import patch

import experiment
import global_smoke
import provider
from test_global_smoke import result, metadata


class ProfileTests(unittest.TestCase):
    def test_budget_and_immutable_image_fail_closed(self):
        for args in [('benchmark', provider.IMAGE, 'b'*40, 180, .65),
                     ('diagnostic', provider.IMAGE, 'b'*40, 46, 1),
                     ('diagnostic', 'latest', 'b'*40, 45, 1),
                     ('diagnostic', provider.IMAGE, 'main', 45, 1)]:
            with self.assertRaises(ValueError):
                experiment.plan(*args)
        p = experiment.plan('diagnostic', provider.IMAGE, 'b'*40, 45, .65)
        self.assertEqual(p['steps'], 50)
        self.assertLess(p['worst_case_compute_usd'], .65)

    def test_full_profile_rejects_smoke_even_if_six_runs_pass(self):
        p = experiment.plan('benchmark', provider.IMAGE, 'b'*40, 120, 2)
        p['source'] = provider.SOURCE
        r = result()
        r["report"]["image_source_sha"] = "b" * 40
        with self.assertRaises(ValueError):
            global_smoke.validate_result(r, 'restore', metadata(), p)
        for row in r['report']['summary']['runs']:
            row['steps'] = 500
        global_smoke.validate_result(r, 'restore', metadata(), p)

    def test_formal_reserve_is_not_one_dollar(self):
        api = type('API', (), {'key': 'fixture'})()
        with patch.object(provider, 'request', return_value={'data': {'myself': {
                'clientBalance': 1.5, 'currentSpendPerHr': 0, 'dailyCharges': []}}}):
            with self.assertRaises(RuntimeError):
                provider.budget(api, reserve=2)

    def test_dora_profile_requires_nine_500_step_runs_and_magnitude(self):
        p = experiment.plan('benchmark-dora', provider.IMAGE, 'b'*40, 120, 2)
        self.assertEqual(p['expected_runs'], 9)
        self.assertEqual(p['steps'], 500)
        p['source'] = provider.SOURCE
        r = result(); report = r['report']
        report['image_source_sha'] = 'b'*40
        gate = report['compatibility']; gate['model_type'] = 'qwen2'
        gate['methods']['dora'] = {**gate['methods']['lora'], 'optimizer_coverage': True,
                                   'reload': {'magnitude_verified': True}}
        rows = report['summary']['runs']
        for row in rows:
            row.update(steps=500, checkpoint_reload={'passed': True})
        # A completed historical six-run matrix cannot satisfy this profile.
        with self.assertRaisesRegex(ValueError, 'Incomplete matrix'):
            global_smoke.validate_result(r, 'restore', metadata(), p)
        for seed in (17, 42, 123):
            rows.append({**rows[0], 'method': 'dora', 'seed': seed,
                         'checkpoint_reload': {'passed': True, 'magnitude_verified': True},
                         'magnitude_final': {'delta_l2': .001}})
        global_smoke.validate_result(r, 'restore', metadata(), p)
        for mutate in (lambda r: r['report']['summary']['runs'][-1].update(steps=50),
                       lambda r: r['report']['summary']['runs'][-1]['checkpoint_reload'].update(magnitude_verified=False),
                       lambda r: r['report']['compatibility'].update(model_type='llama')):
            broken = copy.deepcopy(r); mutate(broken)
            with self.assertRaises(ValueError):
                global_smoke.validate_result(broken, 'restore', metadata(), p)
