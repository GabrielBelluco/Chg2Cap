"""Manually extract complete EBD downloads without creating a concatenated archive.

Verifies pinned part sizes/SHA256, extracts only annotated PNGs, rejects links and
unexpected images. Never downloads or overwrites an existing output directory.
"""

import argparse
import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from data.RSCC.rscc import read_annotations, sha256_file
ANNOTATIONS_SHA256 = '96be5cd62f51910caabee011e00ce973fd817dbd256adb338cca154a7924421e'

PARTS = [
    (3221225472, '676ee1d5c3947fd5d29e899770a4ef6c1a9e67f614c49ed901193a267fe85d17'),
    (3221225472, 'd37bf9b771eab3aef84b260829f37a0e58f933954d48aed64dabbf59efc1eb68'),
    (3221225472, '1e2a7b8a94446665063352dd809e3f0c52f8519094e7e1e67d46dcae420f32ee'),
    (3221225472, 'be99d6be9db082fe18b90e9db5c4efd6b1e893b47ec05b9a82c20af4fc8b111e'),
    (1603907790, 'e2f1e3b4db703679ae8e7861e8f8c8700bb0accf6fb5667713771a252d50e941'),
]


class JoinedParts(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        filled = 0
        while filled < len(buffer):
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = Path(path).open('rb')
            data = self.current.read(len(buffer) - filled)
            if not data:
                self.current.close()
                self.current = None
                continue
            buffer[filled:filled + len(data)] = data
            filled += len(data)
        return filled

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def extract_allowed(stream, allowed, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            if member.name not in allowed:
                continue
            if not member.isfile() or member.name in seen or not 0 < member.size <= 4 * 1024 * 1024:
                raise ValueError('Invalid/duplicate annotated archive member')
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Unsafe output path')
            data = archive.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError('Incomplete image')
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.mode != 'RGB' or image.size != (512, 512):
                    raise ValueError('Expected 512x512 RGB: ' + member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as destination:
                destination.write(data)
            seen.add(member.name)
    if seen != set(allowed):
        raise ValueError(f'Archive lacks {len(set(allowed) - seen)} annotated images; not ready for training')
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory, separate from pilot images')
    if sha256_file(args.annotations) != ANNOTATIONS_SHA256:
        parser.error('Unexpected annotation revision')
    paths = [args.parts_dir / f'EBD.tar.gz-part-{i}' for i in range(5)]
    for path, (size, digest) in zip(paths, PARTS):
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            parser.error('Missing/incomplete/wrong archive part: ' + str(path))
        print('Verified:', path.name, flush=True)
    records = [r for r in read_annotations(args.annotations) if r['source'] == 'EBD']
    allowed = {r[k] for r in records for k in ('before', 'after')}
    with io.BufferedReader(JoinedParts(paths)) as stream:
        count = extract_allowed(stream, allowed, args.output)
    print('Extracted annotated PNGs:', count, '; pairs:', len(records))


if __name__ == '__main__':
    main()
