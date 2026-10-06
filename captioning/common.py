"""Shared I/O and split checks, independent of pilot executables."""
import json
import hashlib
from collections import defaultdict
from data.RSCC.rscc import sha256_file

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)

def check_image_hashes(rows, image_root):
    from data.RSCC.rscc import resolve_image
    for row in rows:
        for field, phase in (('before', 'pre'), ('after', 'post')):
            path = resolve_image(image_root, row[field], phase)
            if sha256_file(path) != row[field + '_sha256']:
                raise ValueError('Image checksum mismatch: ' + str(path))

def assert_disjoint(rows):
    owners = {}
    for r in rows:
        for key in [r['scene_id'], r['group_id'], r['before_pixel_sha256'], r['after_pixel_sha256']]:
            previous = owners.setdefault(key, r['split'])
            if previous != r['split']:
                raise ValueError('Scene/group/image shared across pilot splits')

def group_records(records):
    """Join scenes and identical image pixels, including cross-temporal matches."""
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, r in enumerate(records):
        keys = [('scene', r['scene_id'])]
        keys += [('pixels', r[k + '_pixel_sha256']) for k in ('before', 'after')]
        for key in keys:
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row)
    return list(groups.values())

def ranked(value, seed):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
