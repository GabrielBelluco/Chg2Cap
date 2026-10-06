import copy
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from collections import namedtuple


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, restore_training, validate_resume, require_disk_space
        import torch
        from captioning.resume import restore_training, validate_resume, require_disk_space

    def test_adam_and_rng_continue_identically(self):
        torch.manual_seed(7)
        models = [torch.nn.Linear(2, 2) for _ in range(3)]
        opt = torch.optim.Adam([p for m in models for p in m.parameters()], lr=.001)
        gen = torch.Generator().manual_seed(17)
        def step(ms, op):
            op.zero_grad()
            sum(m(torch.randn(2, 2)).sum() for m in ms).backward()
            op.step()
        step(models, opt)
        state = {k: copy.deepcopy(m.state_dict()) for k, m in zip(
            ('encoder_dict', 'encoder_trans_dict', 'decoder_dict'), models)}
        state.update(optimizer_dict=copy.deepcopy(opt.state_dict()), torch_rng=torch.get_rng_state(),
                     data_generator_rng=gen.get_state())
        step(models, opt)
        restored = [torch.nn.Linear(2, 2) for _ in range(3)]
        opt2 = torch.optim.Adam([p for m in restored for p in m.parameters()], lr=.5)
        gen2 = torch.Generator()
        restore_training(state, restored, opt2, gen2, cuda=False)
        step(restored, opt2)
        for a, b in zip(models, restored):
            for x, y in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(gen.get_state(), gen2.get_state()))
        self.assertEqual(opt2.param_groups[0]['lr'], .001)

    def test_config_and_partial_epoch_rejected(self):
        cfg = dict(manifest_sha256='a', vocab_sha256='b', max_length=551, batch_size=1, seed=7)
        state = {k: {} for k in ('encoder_dict', 'encoder_trans_dict', 'decoder_dict', 'optimizer_dict', 'torch_rng', 'cuda_rng', 'data_generator_rng')}
        state.update(config=cfg, vocab={'word': 0}, epoch=1, steps=10, validation_loss=3.)
        validate_resume(state, {'word': 0}, cfg, 30)
        with self.assertRaises(ValueError):
            validate_resume(state, {'word': 0}, dict(cfg, batch_size=2), 30)
        state['full_epoch'] = False
        with self.assertRaises(ValueError):
            validate_resume(state, {'word': 0}, cfg, 30)

    def test_low_disk_refused(self):
        usage = namedtuple('usage', 'total used free')(100, 99, 1)
        with patch('captioning.resume.shutil.disk_usage', return_value=usage):
            with self.assertRaises(OSError):
                require_disk_space('/tmp', 100)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, select_rows, render_html, validate_checkpoint, main, generate_rows
        import torch
        from captioning.evaluate import select_rows, render_html, validate_checkpoint, main
        from captioning.train import generate_rows
    def rows(self):
        return [dict(id=f'{event}-{i}', event=event, split=split)
                for event in ['A', 'B'] for i, split in enumerate(['train', 'val', 'val', 'val', 'test'])]

    def test_selection_only_val_deterministic(self):
        rows = self.rows()
        a = select_rows(rows, 'sample', 2, 17)
        self.assertEqual(a, select_rows(list(reversed(rows)), 'sample', 2, 17))
        self.assertEqual(len(a), 4)
        self.assertTrue(all(r['split'] == 'val' for r in a))
        self.assertEqual(len(select_rows(rows, 'all')), 6)

    def test_earlier_probes_union_no_duplicates(self):
        rows = self.rows()
        chosen = select_rows(rows, 'sample', 1, 17, True)
        ids = {r['id'] for r in chosen}
        self.assertEqual(len(chosen), len(ids))
        self.assertTrue({r['id'] for r in rows if r['split'] == 'val'} >= ids)
        self.assertTrue({r['id'] for r in [x for x in rows if x['split'] == 'val'][:4]} <= ids)

    def test_reserved_test_selection_and_guard(self):
        rows = self.rows()
        selected = select_rows(rows, 'all', split='test')
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(r['split'] == 'test' for r in selected))
        with self.assertRaises(ValueError):
            select_rows(rows, 'sample', split='test')
        with self.assertRaises(ValueError):
            select_rows(rows, 'all', include_earlier=True, split='test')
        with patch('sys.argv', ['captioning.evaluate', '--output', '/tmp/not-used', '--split', 'test']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main()

    def test_invalid_selection_rejected(self):
        with self.assertRaises(ValueError):
            select_rows(self.rows(), 'sample', 10)
        with self.assertRaises(ValueError):
            select_rows([dict(id='x', split='test')], 'all')

    def test_checkpoint_mismatch_and_partial_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'history.json').write_text(json.dumps([dict(epoch=14, full_epoch=True, steps=20, val_loss=2.)]))
            cfg = dict(manifest_sha256='m', vocab_sha256='v', max_length=10,
                       code_sha256={p: 'code' for p in ('model/model_encoder.py', 'model/model_decoder.py', 'data/RSCC/rscc.py')})
            state = dict(vocab={'a': 1}, config=cfg, epoch=14, steps=20, validation_loss=2.)
            with patch('captioning.evaluate.sha256_file', return_value='code'):
                validate_checkpoint(state, {'a': 1}, 'm', 'v', {'max_length': 10}, root)
                with self.assertRaises(ValueError):
                    validate_checkpoint(state, {'a': 1}, 'wrong', 'v', {'max_length': 10}, root)
                (root / 'history.json').write_text(json.dumps([dict(epoch=14, full_epoch=False, steps=20, val_loss=2.)]))
                with self.assertRaises(ValueError):
                    validate_checkpoint(state, {'a': 1}, 'm', 'v', {'max_length': 10}, root)

    def test_gallery_escapes_text_and_names_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'before.png').touch()
            (root / 'after.png').touch()
            row = dict(id='a', before='before.png', after='after.png', caption='<script>bad</script>')
            render_html(root / 'comparison.html', [row], {'best': [dict(id='a', prediction='<b>caption</b>')]},
                        {'best': {'epoch': 14}}, root)
            html = (root / 'comparison.html').read_text()
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertIn('época 14', html)
            self.assertIn('src="before.png"', html)

    def test_no_grad_generation_does_not_receive_reference_tokens(self):
        class Encoder:
            def eval(self): pass
            def __call__(self, a, b):
                assert not torch.is_grad_enabled()
                return a, b
        class Decoder:
            def eval(self): pass
            def sample(self, a, b, k):
                assert not torch.is_grad_enabled()
                assert k == 1
                return [2, 4, 3]
        class Dataset:
            records = [dict(split='val', before='a', after='b', caption='reference not passed to decoder')]
            def __getitem__(self, index):
                return torch.zeros(1), torch.zeros(1), None, None, None, None, 'x'
        with patch.object(torch.Tensor, 'cuda', lambda self: self):
            rows = generate_rows(Encoder(), Encoder(), Decoder(), Dataset(), [0], {'<NULL>':0, '<START>':2, '<END>':3, 'building':4})
        self.assertEqual(rows[0]['prediction'], 'building')
        self.assertTrue(rows[0]['ended_with_eos'])
        self.assertFalse(rows[0]['output_cap_reached'])

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.argv', ['captioning.evaluate', '--output', tmp]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, trim_padding
        import torch
        from captioning.losses import trim_padding
    def test_padding_trim_preserves_end_tokens_and_full_reference(self):
        tokens = torch.tensor([[2, 8, 3, 0, 0, 0], [2, 9, 10, 11, 3, 0]])
        lengths = torch.tensor([3, 5])
        actual = trim_padding(tokens, lengths)
        self.assertEqual(tuple(actual.shape), (2, 5))
        for i, length in enumerate(lengths):
            self.assertTrue(torch.equal(actual[i, :length], tokens[i, :length]))

    def test_invalid_lengths_rejected(self):
        with self.assertRaises(ValueError):
            trim_padding(torch.zeros(1, 4), torch.tensor([5]))