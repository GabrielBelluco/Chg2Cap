"""Explicit opt-in downloads for pinned semantic evaluators, never inference."""
import argparse
import json
from pathlib import Path

MODELS = {
    'st5': ('sentence-transformers/sentence-t5-xxl',
            '97f38f2f69860f732f602829da6df76ee5c4c0f9', 'sentence-t5-xxl',
            ['*.json', 'spiece.model', 'model.safetensors', '2_Dense/model.safetensors', 'README.md']),
    'bert': ('FacebookAI/roberta-large',
             '722cf37b1afa9454edce342e7895e588b6ff1d59', 'roberta-large',
             ['*.json', 'vocab.json', 'merges.txt', 'model.safetensors', 'README.md']),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--root', type=Path, default=Path('.venv/rscc-semantic/models'))
    p.add_argument('--execute', action='store_true', help='Authorize network and disk writes')
    a = p.parse_args(argv)
    repo, revision, folder, patterns = MODELS[a.model]
    destination = a.root / folder
    plan = {'model': repo, 'revision': revision, 'destination': str(destination),
            'execute': a.execute, 'patterns': patterns}
    print(json.dumps(plan, indent=2))
    if not a.execute:
        return
    # A fresh target protects existing local evaluators and partial downloads.
    if destination.exists() or destination.is_symlink():
        p.error('Destination exists; inspect it and select a new --root')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, revision=revision, local_dir=destination,
                      allow_patterns=patterns, max_workers=2)
    required = ['model.safetensors', 'config.json']
    required += ['modules.json', '2_Dense/model.safetensors', 'spiece.model'] if a.model == 'st5' else ['vocab.json', 'merges.txt']
    if not all((destination / name).is_file() for name in required):
        raise RuntimeError('Incomplete download; preserve files and inspect before using')
    from captioning.protocol import digest
    inventory = dict(plan, files={str(f.relative_to(destination)): digest(f)
                                 for f in destination.rglob('*') if f.is_file() and '.cache' not in f.parts})
    with (destination / 'download-inventory.json').open('x') as f:
        json.dump(inventory, f, indent=2)


if __name__ == '__main__':
    main()
