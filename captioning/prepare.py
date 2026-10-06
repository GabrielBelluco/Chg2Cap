"""Manual full-data preparation from already extracted images. Never downloads.

EBD-only and all-RSCC are distinct scopes. All-RSCC requires an explicit official
test ID list; never infer the official split from JSONL/archive ordering.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from data.RSCC.rscc import read_annotations, sha256_file, resolve_image, tokenize, build_vocab, validate_splits
from captioning.common import group_records, assert_disjoint, ranked
ANNOTATIONS_SHA256 = '96be5cd62f51910caabee011e00ce973fd817dbd256adb338cca154a7924421e'
from captioning.common import write_json


def assign_full_splits(rows, previous, official_test_ids, seed=17):
    known = {r['id'] for r in rows}
    if not official_test_ids.issubset(known):
        raise ValueError('Official test list contains IDs absent from this scope')
    reserved = {}
    for row in previous:
        if row['id'] not in known:
            continue
        if row['id'] in reserved and reserved[row['id']] != row['split']:
            raise ValueError('Previous manifests disagree')
        reserved[row['id']] = row['split']
    for pair_id in official_test_ids:
        if pair_id in reserved and reserved[pair_id] != 'test':
            raise ValueError('Official test overlaps a previous train/val assignment')
        reserved[pair_id] = 'test'
    output = []
    for group in group_records(rows):
        fixed = {reserved[r['id']] for r in group if r['id'] in reserved}
        if len(fixed) > 1:
            raise ValueError('Scene/pixel group connects reserved splits; resolve provenance before training')
        group_id = min(r['id'] for r in group)
        bucket = int(ranked('full:' + group_id, seed), 16) % 10
        split = next(iter(fixed)) if fixed else 'test' if bucket == 0 else 'val' if bucket == 1 else 'train'
        output.extend(dict(r, group_id=group_id, split=split,
                           official_test=r['id'] in official_test_ids) for r in group)
    validate_splits(output)
    assert_disjoint(output)
    if {r['split'] for r in output} != {'train', 'val', 'test'}:
        raise ValueError('All three splits must be nonempty')
    return sorted(output, key=lambda r: (r['split'], r['id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--scope', choices=['EBD', 'all'], required=True)
    parser.add_argument('--preserve-manifest', type=Path, nargs='*', default=[])
    parser.add_argument('--official-test-ids', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh preparation directory')
    if sha256_file(args.annotations) != ANNOTATIONS_SHA256:
        parser.error('This preparation is validated for the pinned RSCC_qvq.jsonl revision only')
    official = set()
    if args.scope == 'all':
        if not args.official_test_ids:
            parser.error('Full RSCC requires --official-test-ids (author-verified canonical IDs, one per line)')
        official_lines = [l.strip() for l in args.official_test_ids.read_text().splitlines() if l.strip()]
        official = set(official_lines)
        if len(official) != 988 or len(official_lines) != 988:
            parser.error('Expected 988 distinct author-verified test pair IDs; do not guess from image count/order')
    elif args.official_test_ids:
        parser.error('Official RSCC test is xBD; do not supply that list for EBD-only scope')
    all_rows = read_annotations(args.annotations)
    rows = [r for r in all_rows if args.scope == 'all' or r['source'] == 'EBD']
    expected = 62351 if args.scope == 'all' else 18215
    if len(rows) != expected:
        parser.error('Unexpected number of source annotations')
    previous = []
    for path in args.preserve_manifest:
        previous.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    # Check presence before decoding thousands of files; never skip missing images.
    missing = []
    for row in rows:
        for field, phase in [('before', 'pre'), ('after', 'post')]:
            try:
                resolve_image(args.image_root, row[field], phase)
            except FileNotFoundError:
                missing.append(row[field])
    if missing:
        parser.error(f'Missing {len(missing)} images; no partial full dataset will be produced. First paths: {missing[:5]}')
    for i, row in enumerate(rows):
        for field, phase in [('before', 'pre'), ('after', 'post')]:
            path = resolve_image(args.image_root, row[field], phase)
            row[field + '_sha256'] = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                if image.mode != 'RGB' or image.size != (512, 512):
                    raise ValueError('Expected published 512x512 RGB pair; verify original crop mapping: ' + str(path))
                row[field + '_pixel_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
        if (i + 1) % 500 == 0:
            print('Verified image pairs:', i + 1, '/', len(rows), flush=True)
    prepared = assign_full_splits(rows, previous, official, args.seed)
    vocab = build_vocab([r for r in prepared if r['split'] == 'train'])
    args.output.mkdir(parents=True)
    with (args.output / 'manifest.jsonl').open('x') as stream:
        for row in prepared:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(args.output / 'vocab.json', vocab)
    report = {'scope': args.scope, 'pairs': len(prepared), 'seed': args.seed,
              'annotations_sha256': sha256_file(args.annotations),
              'max_length': max(len(tokenize(r['caption'])) for r in all_rows),
              'vocab_size': len(vocab), 'splits': dict(Counter(r['split'] for r in prepared)),
              'events': dict(Counter(r['event'] for r in prepared)),
              'preserve_manifests': {str(p): sha256_file(p) for p in args.preserve_manifest},
              'official_test_ids_sha256': sha256_file(args.official_test_ids) if args.official_test_ids else None,
              'official_test_provenance': 'User must verify authorship of supplied ID list; counts alone do not prove provenance.',
              'split_policy': 'Preserve prior/official assignments by connected scene/pixel group; remaining hash buckets ~80/10/10.',
              'warning': 'Internal splits, not exact reproduction of official training split. No geographic-independence guarantee. No training performed.'}
    write_json(args.output / 'audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
