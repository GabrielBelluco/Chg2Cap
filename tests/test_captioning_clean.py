"""Help/contracts from a clean source copy with no installed site-packages."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CleanSource(unittest.TestCase):
    def test_without_local_environments_or_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            for folder in ('captioning', 'configs'):
                shutil.copytree(ROOT / folder, dest / folder,
                                ignore=shutil.ignore_patterns('__pycache__'))
            for name in ('data/RSCC/rscc.py', 'model/model_encoder.py',
                         'model/model_decoder.py', 'eval_func/bleu/bleu.py',
                         'eval_func/bleu/bleu_scorer.py', 'eval_func/rouge/rouge.py',
                         'tests/test_captioning_cli.py', 'tests/test_captioning_protocol.py',
                         'tests/test_captioning_qwen.py'):
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            commands = [['-m', 'captioning', '--help'],
                        ['-m', 'captioning', 'chg2cap', 'train', '--output', 'output/new',
                         '--hours', '2', '--dry-run'],
                        ['-m', 'captioning', 'qwen', 'validation', '--output', 'output/new', '--dry-run']]
            for pattern in ('test_captioning_cli.py', 'test_captioning_protocol.py', 'test_captioning_qwen.py'):
                commands.append(['-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern])
            for args in commands:
                result = subprocess.run([sys.executable, '-B', '-S', *args], cwd=dest,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((dest / 'output').exists())
            self.assertFalse((dest / '.venv').exists())


if __name__ == '__main__':
    unittest.main()
