"""Freeze a new experiment after validation, without consuming test predictions.

This protocol is separate from the historical epoch-14 fixture. Hashes bind the
chosen weights, dataset, generation and current implementation, not a magic epoch.
It records chronology but cannot prove that a human never examined the test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_hashes():
    paths = sorted((ROOT / 'captioning').glob('*.py'))
    paths += [ROOT / p for p in ('model/model_encoder.py', 'model/model_decoder.py', 'data/RSCC/rscc.py',
                                'eval_func/bleu/bleu.py', 'eval_func/bleu/bleu_scorer.py',
                                'eval_func/rouge/rouge.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze(run, prepared, validation, output, generation_limit):
    run, prepared, validation, output = map(Path, (run, prepared, validation, output))
    if output.exists():
        raise ValueError('Protocol already exists')
    plan = json.loads((validation / 'protocol.json').read_text())
    scores = json.loads((validation / 'metrics.json').read_text())
    if plan.get('code_sha256') != code_hashes():
        raise ValueError('Validation implementation differs from current code')
    rows = json.loads((validation / 'best-predictions.json').read_text())
    manifest_rows = [json.loads(l) for l in (prepared / 'manifest.jsonl').read_text().splitlines() if l.strip()]
    val_ids = {r['id'] for r in manifest_rows if r['split'] == 'val'}
    if not rows or len({r['id'] for r in rows}) != len(rows) or {r['id'] for r in rows} != val_ids:
        raise ValueError('Complete validation coverage required before freeze')
    if any(r['split'] != 'val' for r in rows) or plan.get('split', 'val') != 'val' or plan['scope'] != 'all':
        raise ValueError('Freeze uses validation only')
    if scores['status'] != 'completed' or scores['test_evaluated'] or scores['pairs'] != len(rows):
        raise ValueError('Validation incomplete or wrong split')
    expected = {'checkpoint_sha256': digest(run / 'best.pth'),
                'manifest_sha256': digest(prepared / 'manifest.jsonl'),
                'vocab_sha256': digest(prepared / 'vocab.json')}
    if (plan['models']['best']['sha256'] != expected['checkpoint_sha256'] or
        plan['manifest_sha256'] != expected['manifest_sha256'] or
        plan['vocab_sha256'] != expected['vocab_sha256'] or plan['generation_limit'] != generation_limit):
        raise ValueError('Validation does not match selected checkpoint/data/generation')
    value = dict(expected, schema=1, frozen_at=datetime.now(timezone.utc).isoformat(),
                 generation_limit=generation_limit, decoding='greedy', code_sha256=code_hashes(),
                 epoch=plan['models']['best']['epoch'], test_pairs=sum(r['split'] == 'test' for r in manifest_rows),
                 validation_predictions_sha256=digest(validation / 'best-predictions.json'),
                 history_sha256=digest(run / 'history.json'), audit_sha256=digest(prepared / 'audit.json'))
    value['sha256'] = signature(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(value, f, indent=2)
    return value


def check(path, run, prepared, generation_limit):
    value = json.loads(Path(path).read_text())
    signed = value.pop('sha256')
    if signed != signature(value) or value['schema'] != 1:
        raise ValueError('Invalid protocol signature/schema')
    if value['code_sha256'] != code_hashes():
        raise ValueError('Code changed since freeze; do not tune on test')
    for key, filename in [('checkpoint_sha256', Path(run)/'best.pth'),
                          ('history_sha256', Path(run)/'history.json'),
                          ('manifest_sha256', Path(prepared)/'manifest.jsonl'),
                          ('vocab_sha256', Path(prepared)/'vocab.json'),
                          ('audit_sha256', Path(prepared)/'audit.json')]:
        if digest(filename) != value[key]:
            raise ValueError('Frozen artifact changed: ' + key)
    if value['generation_limit'] != generation_limit or value['decoding'] != 'greedy':
        raise ValueError('Generation differs from frozen validation')
    return dict(value, sha256=signed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'prepared', 'validation', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--generation-limit', type=int, default=551)
    a = p.parse_args()
    print(json.dumps(freeze(a.run, a.prepared, a.validation, a.output, a.generation_limit), indent=2))


if __name__ == '__main__':
    main()
