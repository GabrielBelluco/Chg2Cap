import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from PIL import Image

import json
from data.RSCC.rscc import (RSCCDataset, audit, build_vocab, normalize_record,
                            read_annotations, relative_image_path, resolve_image,
                            tokenize, validate_splits)
from captioning.common import assert_disjoint, group_records
from captioning.prepare import assign_full_splits

from captioning.extract import JoinedParts, extract_allowed


class TestFullTools(unittest.TestCase):
    def test_joined_parts_match_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / str(i) for i in range(3)]
            for p, data in zip(paths, [b'abc', b'', b'defg']):
                p.write_bytes(data)
            with io.BufferedReader(JoinedParts(paths), buffer_size=2) as stream:
                self.assertEqual(stream.read(4), b'abcd')
                self.assertEqual(stream.read(), b'efg')

    def test_extract_only_allowlisted_images(self):
        png = io.BytesIO()
        Image.new('RGB', (512, 512), 'green').save(png, format='PNG')
        packed = io.BytesIO()
        with tarfile.open(fileobj=packed, mode='w:gz') as archive:
            for name in ['EBD/event/images/a.png', '../unsafe.txt']:
                data = png.getvalue() if name.endswith('.png') else b'ignored'
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'out'
            self.assertEqual(extract_allowed(io.BytesIO(packed.getvalue()), {'EBD/event/images/a.png'}, root), 1)
            self.assertFalse((Path(directory) / 'unsafe.txt').exists())
            with self.assertRaises(FileExistsError):
                extract_allowed(io.BytesIO(packed.getvalue()), {'EBD/event/images/a.png'}, root)


def annotation(caption='A building collapsed.', part=''):
    return {
        'pre_image': '/home/datasets/EBD/EVENT/images/EVENT_000001_pre_disaster' + part + '.png',
        'post_image': '/home/datasets/EBD/EVENT/images/EVENT_000001_post_disaster' + part + '.png',
        'change_caption': caption,
    }

class DatasetTests(unittest.TestCase):
    def setUp(self):
        global torch, DataLoader
        import torch
        from torch.utils.data import DataLoader
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, caption='A building collapsed.'):
        record = dict(normalize_record(annotation(caption)), split='smoke')
        for name, color in (('before', 'red'), ('after', 'blue')):
            path = self.root / record[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (512, 512), color).save(path)
        manifest = self.root / 'manifest.jsonl'
        manifest.write_text(json.dumps(record) + '\n')
        return record, manifest, build_vocab([record])

    def test_original_caption_preserved(self):
        text = '  One road, two buildings; no water.  '
        row = normalize_record(annotation(text))
        self.assertEqual(row['caption'], text)
        self.assertEqual(tokenize(text), ['<START>', 'One', 'road', ',', 'two', 'buildings', ';', 'no', 'water', '<END>'])

    def test_temporal_pair_mismatch_rejected(self):
        row = annotation()
        row['post_image'] = row['post_image'].replace('000001', '000002')
        with self.assertRaises(ValueError):
            normalize_record(row)

    def test_empty_caption_rejected(self):
        with self.assertRaises(ValueError):
            normalize_record(annotation('  '))

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):
            relative_image_path('/home/datasets/EBD/../../escape.png')
        with self.assertRaises(ValueError):
            resolve_image(self.root, '../escape.png', 'pre')

    def test_duplicate_pair_rejected(self):
        path = self.root / 'annotations.jsonl'
        path.write_text((json.dumps(annotation()) + '\n') * 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_annotations(path)

    def test_crops_cannot_leak_across_splits(self):
        first = dict(normalize_record(annotation(part='_part1')), split='train')
        second = dict(normalize_record(annotation(part='_part2')), split='test')
        with self.assertRaisesRegex(ValueError, 'Scene shared'):
            validate_splits([first, second])

    def test_explicit_split_required(self):
        with self.assertRaises(ValueError):
            validate_splits([normalize_record(annotation())])

    def test_vocabulary_rejects_validation_data(self):
        record = dict(normalize_record(annotation()), split='val')
        with self.assertRaises(ValueError):
            build_vocab([record])

    def test_loader_matches_seven_field_interface(self):
        record, manifest, vocab = self.fixture()
        dataset = RSCCDataset(manifest, self.root, vocab, 'smoke', 64)
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        self.assertEqual(len(batch), 7)
        self.assertEqual(tuple(batch[0].shape), (1, 3, 256, 256))
        self.assertEqual(tuple(batch[2].shape), (1, 1, 64))
        self.assertEqual(tuple(batch[3].shape), (1, 1, 1))
        self.assertEqual(tuple(batch[4].shape), (1, 64))
        self.assertEqual(batch[5].item(), 5)
        self.assertEqual(batch[4][0, 4].item(), vocab['<END>'])
        self.assertTrue(torch.isfinite(batch[0]).all())
        self.assertGreater(batch[0][0, 0].mean(), batch[1][0, 0].mean())

    def test_long_caption_is_not_silently_truncated(self):
        _, manifest, vocab = self.fixture('word ' * 80)
        with self.assertRaisesRegex(ValueError, 'refusing silent truncation'):
            RSCCDataset(manifest, self.root, vocab, 'smoke', 41)

    def test_missing_images_fail_early(self):
        record, manifest, vocab = self.fixture()
        (self.root / record['after']).unlink()
        with self.assertRaises(FileNotFoundError):
            RSCCDataset(manifest, self.root, vocab, 'smoke', 64)

    def test_converter_directory_layout(self):
        relative = Path('EBD/EVENT/images/example.png')
        target = self.root / 'EBD/EVENT/images_pre/example.png'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'fixture')
        self.assertEqual(resolve_image(self.root, str(relative), 'pre'), target)

    def test_audit_counts_long_captions(self):
        result = audit([normalize_record(annotation('word ' * 80))])
        self.assertEqual(result['tokens_max'], 82)
        self.assertEqual(result['exceed_baseline_41_tokens'], 1)
        self.assertFalse(result['official_split_in_annotations'])

def fixture():
    rows = [{'id': str(i), 'scene_id': 'scene' + str(i), 'source': 'EBD', 'event': 'event',
             'before_pixel_sha256': 'a' + str(i), 'after_pixel_sha256': 'b' + str(i)} for i in range(607)]
    previous = [dict(r, split='train' if i < 128 else 'val' if i < 144 else 'test', group_id=r['id'])
                for i, r in enumerate(rows[:160])]
    return rows, previous

def records(count=40):
    return [{'id': f'EBD/event/{i}', 'scene_id': f'event/scene{i}', 'event': f'event{i % 2}',
             'source': 'EBD', 'before_pixel_sha256': f'before{i}', 'after_pixel_sha256': f'after{i}',
             'caption': f'word{i} changed.'} for i in range(count)]

class PreparationTests(unittest.TestCase):
    def test_full_preparation_preserves_old_and_official_groups(self):
        rows, previous = fixture()
        rows[201]['before_pixel_sha256'] = rows[200]['before_pixel_sha256']
        output = {r['id']: r for r in assign_full_splits(rows, previous, {'200'})}
        self.assertEqual(output['200']['split'], 'test')
        self.assertEqual(output['201']['split'], 'test')
        for row in previous:
            self.assertEqual(row['split'], output[row['id']]['split'])

    def test_full_official_test_cannot_overlap_old_training(self):
        rows, previous = fixture()
        with self.assertRaises(ValueError):
            assign_full_splits(rows, previous, {'0'})

    def test_full_unknown_test_id_fails(self):
        rows, previous = fixture()
        with self.assertRaises(ValueError):
            assign_full_splits(rows, previous, {'unknown'})

    def test_identical_pixels_grouped_even_across_temporal_roles(self):
        rows = records()
        rows[1]['after_pixel_sha256'] = rows[0]['before_pixel_sha256']
        rows[2]['scene_id'] = rows[1]['scene_id']
        groups = group_records(rows)
        self.assertTrue(any({r['id'] for r in g} == {'EBD/event/0', 'EBD/event/1', 'EBD/event/2'} for g in groups))
        split = assign_full_splits(rows, [], set())
        affected = [r['split'] for r in split if r['id'] in {'EBD/event/0', 'EBD/event/1', 'EBD/event/2'}]
        self.assertEqual(len(set(affected)), 1)

    def test_vocab_has_no_validation_or_test_words(self):
        rows = assign_full_splits(records(), [], set())
        train = [r for r in rows if r['split'] == 'train']
        vocab = build_vocab(train)
        for r in rows:
            unique = tokenize(r['caption'])[1]
            self.assertEqual(unique in vocab, r['split'] == 'train')

    def test_splits_deterministic_and_input_order_independent(self):
        rows, previous = fixture()
        self.assertEqual(assign_full_splits(rows, previous, set()),
                         assign_full_splits(rows[::-1], previous[::-1], set()))
        assert_disjoint(assign_full_splits(rows, previous, set()))

    def test_conflicting_pixel_groups_rejected(self):
        rows, previous = fixture()
        rows[130]['before_pixel_sha256'] = rows[0]['after_pixel_sha256']
        with self.assertRaises(ValueError):
            assign_full_splits(rows, previous, set())


if __name__ == '__main__':
    unittest.main()
