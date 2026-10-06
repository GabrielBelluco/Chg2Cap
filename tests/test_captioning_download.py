import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from captioning.download import main


class Downloads(unittest.TestCase):
    def test_plan_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'models'
            with contextlib.redirect_stdout(io.StringIO()) as log:
                main(['--model', 'st5', '--root', str(root)])
            self.assertFalse(root.exists())
            self.assertIn('97f38f2', log.getvalue())

    def test_existing_weights_refused_before_network_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'roberta-large').mkdir()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(['--model', 'bert', '--root', tmp, '--execute'])


if __name__ == '__main__':
    unittest.main()
