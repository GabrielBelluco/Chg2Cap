"""RSCC parsing and image loading, isolated from the existing LEVIR pipeline."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

SPECIAL_TOKENS = {'<NULL>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
SOURCE_PREFIX = '/home/datasets/'


def tokenize(caption):
    """Keep the baseline punctuation convention without changing the raw caption."""
    text = caption
    for mark in (';', ','):
        text = text.replace(mark, ' ' + mark)
    for mark in ('?', '.'):
        text = text.replace(mark, '')
    return ['<START>', *text.split(), '<END>']


def relative_image_path(value):
    if not isinstance(value, str) or not value.startswith(SOURCE_PREFIX):
        raise ValueError('Expected the published /home/datasets/ image prefix')
    relative = PurePosixPath(value[len(SOURCE_PREFIX):])
    if '..' in relative.parts or not relative.parts or relative.is_absolute():
        raise ValueError('Unsafe image path')
    if relative.parts[0] not in ('EBD', 'xbd') or relative.suffix.lower() != '.png':
        raise ValueError('Unsupported RSCC image path')
    return relative.as_posix()


def normalize_record(row):
    for key in ('pre_image', 'post_image', 'change_caption'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError('Missing or empty field: ' + key)
    before = PurePosixPath(relative_image_path(row['pre_image']))
    after = PurePosixPath(relative_image_path(row['post_image']))
    if '_pre_disaster' not in before.name or '_post_disaster' not in after.name:
        raise ValueError('Missing temporal image marker')
    if before.name.replace('_pre_disaster', '_post_disaster') != after.name:
        raise ValueError('The before/after image names do not form a pair')
    # Support both the published images/ layout and the authors' converter layout.
    if before.parent.name not in ('images', 'images_pre'):
        raise ValueError('Unexpected pre-image directory')
    if after.parent.name not in ('images', 'images_post'):
        raise ValueError('Unexpected post-image directory')
    if before.parent.parent != after.parent.parent:
        raise ValueError('Images belong to different events')
    event = before.parent.parent.name
    scene = re.sub(r'_pre_disaster(?:_part\d+)?$', '', before.stem)
    if scene == before.stem:
        raise ValueError('Unrecognized image naming convention')
    return {
        'id': str(before.parent.parent / before.stem),
        'source': before.parts[0],
        'event': event,
        'scene_id': str(before.parent.parent / scene),
        'before': before.as_posix(),
        'after': after.as_posix(),
        'caption': row['change_caption'],
    }


def read_annotations(path):
    records, seen = [], set()
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = normalize_record(json.loads(line))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f'Invalid annotation at line {line_number}: {exc}') from exc
            if record['id'] in seen:
                raise ValueError(f'Duplicate pair at line {line_number}: {record["id"]}')
            seen.add(record['id'])
            records.append(record)
    if not records:
        raise ValueError('No annotations found')
    return records


def audit(records):
    lengths = sorted(len(tokenize(r['caption'])) for r in records)
    words = [len(r['caption'].split()) for r in records]
    return {
        'pairs': len(records),
        'sources': dict(sorted(Counter(r['source'] for r in records).items())),
        'events': dict(sorted(Counter(r['event'] for r in records).items())),
        'scene_count': len({r['scene_id'] for r in records}),
        'caption_words_mean': sum(words) / len(words),
        'tokenization': 'Chg2Cap punctuation convention; whitespace split; START/END included',
        'tokens_min': lengths[0],
        'tokens_median': lengths[len(lengths) // 2],
        'tokens_p95': lengths[min(len(lengths) - 1, int(len(lengths) * .95))],
        'tokens_max': lengths[-1],
        'exceed_baseline_41_tokens': sum(n > 41 for n in lengths),
        'official_split_in_annotations': False,
        'warning': 'This JSONL contains no official train/validation/test assignments.',
    }


def build_vocab(records, min_count=1):
    if min_count < 1 or not records:
        raise ValueError('Vocabulary requires nonempty training records and min_count >= 1')
    splits = {r.get('split') for r in records}
    if splits not in ({'train'}, {'smoke'}):
        raise ValueError('Build vocabulary only from training data or a smoke-only sample')
    counts = Counter(t for r in records for t in tokenize(r['caption']))
    vocab = dict(SPECIAL_TOKENS)
    for token, count in sorted(counts.items()):
        if count >= min_count and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def validate_splits(records):
    """Reject accidental leakage between crops of the same original scene."""
    pairs, scenes = {}, {}
    for r in records:
        split = r.get('split')
        if split not in ('train', 'val', 'test', 'smoke'):
            raise ValueError('Each pair requires an explicit split')
        if r['id'] in pairs:
            raise ValueError('Duplicate pair in manifest: ' + r['id'])
        pairs[r['id']] = split
        previous = scenes.setdefault(r['scene_id'], split)
        if previous != split:
            raise ValueError('Scene shared across splits: ' + r['scene_id'])
    if 'smoke' in set(pairs.values()) and len(set(pairs.values())) > 1:
        raise ValueError('Do not mix smoke data with experimental splits')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_image(root, relative, phase):
    root = Path(root).resolve()
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest image path must be relative and safe')
    candidates = [root / relative]
    if relative.parent.name == 'images':
        candidates.append(root / relative.parent.parent / ('images_' + phase) / relative.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('Image path escapes the dataset root')
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(str(candidates[0]))


class RSCCDataset:
    """Map-style Torch dataset; returns the same seven fields as LEVIRCCDataset.

    Torch/Pillow imports are lazy so annotation auditing needs only Python's stdlib.
    RGB images are resized to 256x256 by default and ImageNet-normalized for the
    pretrained ResNet. This is specific to RSCC and does not alter LEVIR's loader.
    """

    def __init__(self, manifest, image_root, vocab, split, max_length, image_size=256):
        import torch
        if image_size not in (256, 512):
            raise ValueError('Supported image sizes: 256 or 512')
        records = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        validate_splits(records)
        self.records = [r for r in records if r['split'] == split]
        if not self.records:
            raise ValueError('Requested split is empty: ' + split)
        if any(vocab.get(k) != v for k, v in SPECIAL_TOKENS.items()):
            raise ValueError('Vocabulary has incompatible special token IDs')
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError('Vocabulary IDs must be unique and contiguous')
        self.vocab = vocab
        self.max_length = max_length
        self.image_size = image_size
        self.mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
        self.std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
        self.paths = []
        for r in self.records:
            if len(tokenize(r['caption'])) > max_length:
                raise ValueError('Caption exceeds max_length; refusing silent truncation: ' + r['id'])
            self.paths.append((resolve_image(image_root, r['before'], 'pre'),
                               resolve_image(image_root, r['after'], 'post')))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image
        images = []
        for path in self.paths[index]:
            with Image.open(path) as image:
                if image.mode != 'RGB':
                    raise ValueError('Expected an RGB image: ' + str(path))
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                array = np.array(image, dtype=np.float32) / 255.0
            images.append((torch.from_numpy(array).permute(2, 0, 1) - self.mean) / self.std)
        record = self.records[index]
        tokens = tokenize(record['caption'])
        encoded = torch.zeros(self.max_length, dtype=torch.long)
        encoded[:len(tokens)] = torch.tensor([self.vocab.get(t, SPECIAL_TOKENS['<UNK>']) for t in tokens])
        return (images[0], images[1], encoded.unsqueeze(0),
                torch.tensor([[len(tokens)]]), encoded, torch.tensor(len(tokens)), record['id'])
