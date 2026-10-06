"""No GPU, data, weights, downloads or training required."""
import unittest
import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from captioning import cli


class CurrentCommands(unittest.TestCase):
    def setUp(self):
        self.config = cli.read(cli.ROOT / 'configs/captioning_ebd.json')

    def command(self, *args):
        return cli.generation_command(cli.parser().parse_args(args), self.config)

    def test_training_uses_package(self):
        cmd = self.command('train', '--output', '/tmp/new-training', '--hours', '2')
        self.assertIn('captioning.train', cmd)
        self.assertIn('7200.0', cmd)
        self.assertNotIn('scripts/train_rscc.py', cmd)

    def test_validation_explicit_run(self):
        cmd = self.command('validation', '--run', '/tmp/new-run', '--output', '/tmp/new-val')
        self.assertIn('captioning.evaluate', cmd)
        self.assertEqual(cmd[cmd.index('--best') + 1], '/tmp/new-run')
        self.assertNotIn('--allow-test', cmd)

    def test_test_passes_frozen_protocol(self):
        cmd = self.command('test', '--run', '/tmp/new-run', '--output', '/tmp/new-test',
                           '--protocol', '/tmp/frozen.json', '--allow-test')
        self.assertEqual(cmd[cmd.index('--protocol') + 1], '/tmp/frozen.json')

    def test_metrics_are_code_not_output(self):
        from pathlib import Path
        cmds = cli.metric_commands(self.config, Path('/tmp/input'), Path('/tmp/output'), ['best'], 'val')
        self.assertIn('captioning.meteor', cmds[0])
        self.assertIn('captioning.semantic', cmds[1])
        self.assertTrue(all('-m' in cmd for cmd in cmds))


    def test_training_budget_and_resume(self):
        a = cli.parser().parse_args(['train', '--output', 'new', '--hours', '2', '--resume', 'old/best.pth'])
        cmd = cli.generation_command(a, self.config)
        self.assertEqual(cmd[cmd.index('--max-train-seconds') + 1], '7200.0')
        self.assertIn('--skip-final-generation', cmd)
        self.assertIn('--resume', cmd)
        self.assertNotIn('--allow-test', cmd)

    def test_full_validation_default_best(self):
        a = cli.parser().parse_args(['validation', '--run', 'run', '--output', 'new'])
        cmd = cli.generation_command(a, self.config)
        self.assertEqual(cmd[cmd.index('--scope') + 1], 'all')
        self.assertEqual(cmd[cmd.index('--models') + 1], 'best')
        self.assertNotIn('--allow-test', cmd)

    def test_test_needs_authorization(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(['test', '--run', 'run', '--protocol', 'protocol.json', '--output', 'not-created', '--dry-run'])

    def test_dry_run_no_execution(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, 'run') as run:
            dest = Path(tmp) / 'new'
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli.main(['validation', '--run', 'run', '--output', str(dest), '--dry-run'])
            self.assertIn('captioning.semantic', stdout.getvalue())
            run.assert_not_called()
            self.assertFalse(dest.exists())

    def test_check_only_no_scores(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as stdout:
            cli.main(['validation', '--run', 'run', '--output', str(Path(tmp)/'new'), '--check-only', '--dry-run'])
        self.assertNotIn('captioning.semantic', stdout.getvalue())
        self.assertIn('--check-only', stdout.getvalue())

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(['train', '--output', tmp, '--hours', '2', '--dry-run'])

    def test_metrics_match_validation_configuration(self):
        cmds = cli.metric_commands(self.config, Path('/input'), Path('/output'), ['best'], 'test')
        self.assertEqual(len(cmds), 3)
        self.assertTrue(all('--allow-test' in cmd for cmd in cmds))
        self.assertIn('float16', cmds[1])
        self.assertIn('float32', cmds[2])
        self.assertEqual(cmds[2][cmds[2].index('--bert-layers') + 1], '17')

    def test_incomplete_evaluation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'metrics.json').write_text(json.dumps({'status': 'partial'}))
            (p/'protocol.json').write_text('{}')
            with self.assertRaises(ValueError):
                cli.evaluation_info(p)

    def test_summary_uses_saved_scores_and_rejects_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            evaluation, output = p/'evaluation', p/'scores'
            evaluation.mkdir(); output.mkdir()
            rows = [{'id': 'one', 'split': 'val', 'reference': 'road', 'prediction': 'road'}]
            raw = json.dumps(rows).encode()
            (evaluation/'best-predictions.json').write_bytes(raw)
            (evaluation/'protocol.json').write_text(json.dumps({'split':'val'}))
            (evaluation/'metrics.json').write_text(json.dumps({'status':'completed', 'pairs':1, 'models':{
                'best':{'BLEU_1_to_4':[1,1,1,1], 'ROUGE_L':1, 'mean_repeated_trigram_fraction':0}}}))
            for metric in ('meteor','st5-scs','bertscore'):
                report = {'source_sha256':hashlib.sha256(raw).hexdigest(), 'per_pair':[{'id':'one'}],
                          'test_evaluated':False, 'status':'completed', 'METEOR_NLTK':1,
                          'means':{'st5_scs':1,'f1':1}}
                (output/f'best-{metric}.json').write_text(json.dumps(report))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.summarize(evaluation, output)
            self.assertIn('| best | 1.000000', (output/'RESUMO.md').read_text())
            report['source_sha256'] = 'wrong'
            (output/'best-meteor.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                cli.summarize(evaluation, output)

if __name__ == '__main__':
    unittest.main()
