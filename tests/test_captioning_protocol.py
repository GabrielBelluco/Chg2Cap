"""Synthetic artifacts test protocol safety without loading any checkpoint."""
import json
from pathlib import Path
import tempfile
import unittest
from captioning import protocol as p


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.run, self.data, self.val = [root / n for n in ('run', 'data', 'val')]
        for d in (self.run, self.data, self.val):
            d.mkdir()
        self.out = root / 'freeze.json'
        (self.run / 'best.pth').write_bytes(b'synthetic checkpoint')
        self.write(self.run / 'history.json', [])
        for name in ('audit.json', 'vocab.json'):
            self.write(self.data / name, {})
        (self.data / 'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in
            [{'id': 'v1', 'split': 'val'}, {'id': 't1', 'split': 'test'}]))
        self.plan = {'scope': 'all', 'split': 'val', 'generation_limit': 551,
                     'code_sha256': p.code_hashes(), 'models': {'best':
                     {'sha256': p.digest(self.run / 'best.pth'), 'epoch': 3}},
                     'manifest_sha256': p.digest(self.data / 'manifest.jsonl'),
                     'vocab_sha256': p.digest(self.data / 'vocab.json')}
        self.write(self.val / 'protocol.json', self.plan)
        self.write(self.val / 'metrics.json', {'status': 'completed', 'test_evaluated': False, 'pairs': 1})
        self.write(self.val / 'best-predictions.json', [{'id': 'v1', 'split': 'val'}])

    def write(self, path, obj):
        path.write_text(json.dumps(obj))

    def freeze(self):
        return p.freeze(self.run, self.data, self.val, self.out, 551)

    def test_arbitrary_epoch_and_check(self):
        self.assertEqual(self.freeze()['epoch'], 3)
        self.assertEqual(p.check(self.out, self.run, self.data, 551)['test_pairs'], 1)

    def test_no_overwrite(self):
        self.freeze()
        with self.assertRaises(ValueError):
            self.freeze()

    def test_changed_weights_rejected(self):
        self.freeze()
        (self.run / 'best.pth').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            p.check(self.out, self.run, self.data, 551)

    def test_changed_generation_rejected(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'Generation differs'):
            p.check(self.out, self.run, self.data, 99)

    def test_incomplete_validation_rejected(self):
        self.write(self.val / 'best-predictions.json', [])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.freeze()

    def test_test_predictions_not_validation(self):
        self.write(self.val / 'best-predictions.json', [{'id': 'v1', 'split': 'test'}])
        with self.assertRaisesRegex(ValueError, 'validation only'):
            self.freeze()

    def test_changed_code_before_freeze_rejected(self):
        self.plan['code_sha256'] = {}
        self.write(self.val / 'protocol.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'implementation differs'):
            self.freeze()

    def test_corrupted_protocol_rejected(self):
        frozen = self.freeze()
        frozen['epoch'] = 99
        self.write(self.out, frozen)
        with self.assertRaisesRegex(ValueError, 'signature'):
            p.check(self.out, self.run, self.data, 551)


if __name__ == '__main__':
    unittest.main()
