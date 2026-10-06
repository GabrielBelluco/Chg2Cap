"""Offline CPU-only METEOR-NLTK for saved RSCC predictions; no model imports.

Uses word_tokenize and single_meteor_score with the defaults used by
Hugging Face Evaluate's METEOR implementation. Not Java METEOR 1.5.
Never overwrites an existing report or downloads resources implicitly.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import nltk
from nltk.translate.meteor_score import single_meteor_score


def evaluate(rows, split='val', allow_test=False):
    if split not in ('val', 'test') or (split == 'test' and not allow_test):
        raise ValueError('Test requires explicit authorization')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty JSON list of predictions')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each row must be an object')
        if not isinstance(row.get('id'), str) or row['id'] in seen:
            raise ValueError('Missing or duplicate id')
        seen.add(row['id'])
        if row.get('split') != split:
            raise ValueError('Rows do not match requested split')
        if not isinstance(row.get('reference'), str) or not row['reference'].strip():
            raise ValueError('Missing reference')
        if not isinstance(row.get('prediction'), str):
            raise ValueError('Missing prediction')
    scores = []
    for row in rows:
        score = single_meteor_score(
            nltk.word_tokenize(row['reference']),
            nltk.word_tokenize(row['prediction']),
            alpha=0.9, beta=3, gamma=0.5,
        )
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid METEOR score')
        scores.append({'id': row['id'], 'meteor': score})
    return {'pairs': len(scores),
            'METEOR_NLTK': sum(s['meteor'] for s in scores) / len(scores),
            'per_pair': scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--allow-test', action='store_true')
    parser.add_argument('--resources', type=Path,
                        default=Path(sys.prefix) / 'nltk_data')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new report name')
    # Do not silently borrow resources from the user's other environments.
    nltk.data.path[:] = [str(args.resources.resolve())]
    raw = args.predictions.read_bytes()
    report = evaluate(json.loads(raw), args.split, args.allow_test)
    report.update({
        'implementation': 'nltk.translate.meteor_score.single_meteor_score',
        'nltk_version': nltk.__version__, 'python_version': sys.version,
        'tokenizer': 'nltk.word_tokenize; language=english; preserve_line=False',
        'parameters': {'alpha': 0.9, 'beta': 3, 'gamma': 0.5},
        'aggregation': 'arithmetic mean of per-pair scores',
        'source': str(args.predictions.resolve()),
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'resource_sha256': {str(p.relative_to(args.resources)):
                            hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(args.resources.rglob('*.zip'))},
        'test_evaluated': args.split == 'test',
        'split': args.split,
        'warning': 'Not Java METEOR 1.5. Text similarity is not visual correctness. '
                   'This alone does not reproduce the RSCC paper protocol.',
    })
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ['pairs', 'METEOR_NLTK', 'nltk_version']}))


if __name__ == '__main__':
    main()
