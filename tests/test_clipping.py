import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from simplexbench.clipping import clip_gradients
from simplexbench.common import atomic_json, digest, execution_order, file_hash, validate_config
from simplexbench.ablation import arm_config, paired_order, plan, summarize_ablation
from simplexbench.cli import source_identity
from simplexbench.runner import gradient_stats, learning_rate
from simplexbench.clipping import validate_clip_evidence
from test_runner import run_cpu_fixture

ROOT = Path(__file__).parents[1]


def parameters():
    result = []
    for name, value in [("score.modules_to_save.default.weight", 100.), ("q.lora_A.default.weight", 0.),
                        ("q.lora_B.default.weight", 3.), ("q.lora_magnitude_vector.default.weight", 4.),
                        ("simplex_importance", .01)]:
        p = torch.nn.Parameter(torch.ones(1))
        p.grad = torch.tensor([value])
        result.append((name, p))
    return result


def test_global_is_bitwise_legacy_and_grouped_includes_importance_magnitude():
    original, actual = parameters(), parameters()
    torch.nn.utils.clip_grad_norm_([p for _, p in original], 1., error_if_nonfinite=True)
    clip_gradients(actual, "global")
    for (_, expected), (_, got) in zip(original, actual):
        assert torch.equal(expected.grad, got.grad)
    grouped = parameters()
    stats = gradient_stats(grouped, 100.)
    stats["clipping"] = clip_gradients(grouped, "grouped")
    stats["post_clip"] = gradient_stats(grouped, 100.)
    validate_clip_evidence(stats, "grouped")
    assert stats["post_clip"]["norm"] > 1.4  # Two caps, explicitly not a total cap of 1.
    coef = 1 / ((25 + .0001) ** .5 + 1e-6)
    assert grouped[-1][1].grad.item() == pytest.approx(.01 * coef)
    assert grouped[-2][1].grad.item() == pytest.approx(4 * coef)
    assert grouped[1][1].grad.item() == 0
    assert stats["clipping"]["groups"]["adapter"]["tensors"] == 4
    bad = copy.deepcopy(stats); bad["clipping"]["groups"]["adapter"]["coefficient"] = 1
    with pytest.raises(ValueError, match="coefficient"):
        validate_clip_evidence(bad, "grouped")


@pytest.mark.parametrize("failure", ["unknown", "duplicate", "missing", "nonfinite", "empty"])
def test_invalid_clipping_fails_closed(failure):
    params = parameters()
    if failure == "unknown": params[0] = ("unrecognized", params[0][1])
    if failure == "duplicate": params.append(params[0])
    if failure == "missing": params[0][1].grad = None
    if failure == "nonfinite": params[0][1].grad.fill_(float("nan"))
    if failure == "empty": params = params[1:]
    with pytest.raises((ValueError, RuntimeError)):
        clip_gradients(params, "grouped")


def test_profiles_preserve_baselines_and_diagnostic_prefix():
    configs = {}
    for kind in ("diagnostic", "benchmark"):
        old = json.loads((ROOT / f"configs/{kind}-dora.json").read_text())
        new = json.loads((ROOT / f"configs/{kind}-clipping.json").read_text())
        assert {k: v for k, v in new.items() if k != "clipping_strategies"} == old
        assert plan(new)["optimizer_steps"] == 18 * old["steps"]
        configs[kind] = new
        order = paired_order(new)
        assert len({tuple(sorted(x.items())) for x in order}) == 18
        assert [x["strategy"] for x in order[:4]] == ["global", "grouped", "grouped", "global"]
    for method in configs['benchmark']['methods']:
        assert [learning_rate(configs['diagnostic'], s, method) for s in range(1, 51)] == [
            learning_rate(configs['benchmark'], s, method) for s in range(1, 51)]
    with pytest.raises(ValueError, match='clipping strategy'):
        validate_config({**configs['benchmark'], 'clipping_strategy': 'typo'})


@pytest.mark.parametrize("method", ["lora", "simplex", "dora"])
def test_instrumented_global_preserves_legacy_training(tmp_path, monkeypatch, method):
    run_cpu_fixture(tmp_path / 'legacy', monkeypatch, method)
    run_cpu_fixture(tmp_path / 'global', monkeypatch, method, config_updates={'clipping_strategy': 'global'})
    paths = [tmp_path / arm / f'{method}-17' for arm in ('legacy', 'global')]
    weights = [load_file(p / 'adapter/adapter_model.safetensors') for p in paths]
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    assert (paths[0] / 'ranks.json').read_bytes() == (paths[1] / 'ranks.json').read_bytes()
    rows = [[json.loads(line) for line in (p / 'steps.jsonl').read_text().splitlines()] for p in paths]
    assert [(r['loss'], r['lr']) for r in rows[0]] == [(r['loss'], r['lr']) for r in rows[1]]
    if method == 'simplex':
        assert all(r['importance'] is not None for r in rows[1][:4])
        assert all(r['importance'] is None for r in rows[1][4:])
        assert rows[1][3]['importance']['reset_to_one']


def test_full_eighteen_run_pairing_and_tampering(tmp_path, monkeypatch):
    c = json.loads((ROOT / 'configs/smoke.json').read_text())
    c.update(methods=['lora','simplex','dora'], evidence_version=2, clipping_strategies=['global','grouped'])
    from simplexbench import ablation
    config_path = tmp_path / 'input.json'
    atomic_json(config_path, c)
    output = tmp_path / 'output'
    calls = []
    def fake_subprocess(command, check):
        assert check
        calls.append(command)
        if command[2] == 'simplexbench.compat':
            assert '--cuda' in command
            return
        arm = json.loads(Path(command[command.index('--manifest') + 1]).read_text())
        directory = Path(command[command.index('--output') + 1])
        method = command[command.index('--method') + 1]
        seed = int(command[command.index('--seed') + 1])
        run_cpu_fixture(directory.parent, monkeypatch, method, seed,
            config_updates={'clipping_strategy': arm['config']['clipping_strategy']}, manifest_override=arm)
    monkeypatch.setattr(ablation, 'resolve', lambda c: dict(c))
    monkeypatch.setattr(ablation.subprocess, 'run', fake_subprocess)
    summary = ablation.matrix(config_path, output)
    assert len(calls) == 19
    observed = [{'method': cmd[cmd.index('--method') + 1], 'seed': int(cmd[cmd.index('--seed') + 1]),
                 'strategy': Path(cmd[cmd.index('--output') + 1]).parent.name} for cmd in calls[1:]]
    assert observed == paired_order(c)
    with pytest.raises(ValueError, match='Existing output'):
        ablation.matrix(config_path, output)
    ablation.matrix(config_path, output, resume=True)
    assert len(calls) == 20  # Only the gate repeats; sealed workers are skipped.
    tmp_path = output
    assert len(summary['runs']) == 18 and len(summary['pairs']) == 9
    assert not summary['cuda_int8_gate_passed']
    assert {r['clipping_strategy'] for r in summary['runs']} == {'global', 'grouped'}
    assert all(v['test_macro_f1']['n'] == 3 for v in summary['paired_deltas'].values())
    assert summary['measurements']['grouped/simplex-17']['search']['gradient_norms']['importance']['pre']['n'] == 4
    assert summary['measurements']['grouped/simplex-17']['post_search']['gradient_norms']['importance']['pre']['n'] == 0
    assert all(v['time_to_target_delta_seconds']['n'] == 0 for v in summary['paired_deltas'].values())
    # Resealed batch/LR/device changes must still fail paired validation.
    path = tmp_path / 'grouped/lora-17'
    for filename, mutate, error in [
        ('steps.jsonl', lambda rows: rows[0]['observations'].update(batch_sha256='bad'), 'Unpaired batches'),
        ('steps.jsonl', lambda rows: rows[0].update(lr=123.), 'Unpaired batch order'),
        ('provenance.json', lambda p: p.update(gpu_uuid='another-device'), 'Hardware/software')]:
        original_bytes = (path / filename).read_bytes()
        value = ([json.loads(line) for line in original_bytes.splitlines()] if filename.endswith('jsonl')
                 else json.loads(original_bytes))
        mutate(value)
        if filename.endswith('jsonl'):
            (path / filename).write_text(''.join(json.dumps(row) + '\n' for row in value))
        else:
            atomic_json(path / filename, value)
        status = json.loads((path / 'status.json').read_text())
        status['artifacts'][filename] = file_hash(path / filename)
        atomic_json(path / 'status.json', status)
        with pytest.raises(ValueError, match=error):
            summarize_ablation(tmp_path)
        (path / filename).write_bytes(original_bytes)
        status['artifacts'][filename] = file_hash(path / filename)
        atomic_json(path / 'status.json', status)
    # Resealing a changed init hash must NOT bypass semantic pairing checks.
    path = tmp_path / 'grouped/lora-17'
    prov = json.loads((path / 'provenance.json').read_text())
    original = copy.deepcopy(prov)
    prov['initial_trainable_sha256'] = '0' * 64
    atomic_json(path / 'provenance.json', prov)
    status = json.loads((path / 'status.json').read_text())
    status['artifacts']['provenance.json'] = file_hash(path / 'provenance.json')
    atomic_json(path / 'status.json', status)
    with pytest.raises(ValueError, match='Unpaired initialization'):
        summarize_ablation(tmp_path)
    atomic_json(path / 'provenance.json', original)
    status['artifacts']['provenance.json'] = file_hash(path / 'provenance.json')
    atomic_json(path / 'status.json', status)
    (tmp_path / 'grouped/dora-123/status.json').unlink()
    with pytest.raises(FileNotFoundError):
        summarize_ablation(tmp_path)
