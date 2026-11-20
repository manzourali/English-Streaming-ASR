"""Private, checksummed pilot restart bundle. Never a public review artifact."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import tarfile
import time

from pipeline import ROOT, sha

BASE = Path('/kaggle/working/surt-pilot-2/runs')
MAX_BYTES = 512 * 1024 * 1024


def canonical(path):
    if path.parent != BASE or not re.fullmatch(r'[A-Za-z0-9_-]+', path.name):
        raise ValueError('Restart must retain the original canonical pilot run path')
    return path


def pack(run):
    canonical(run)
    files = sorted(p for p in run.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in run.rglob('*')):
        raise ValueError('Symlinks are not restart artifacts')
    if sum(p.stat().st_size for p in files) > MAX_BYTES:
        raise ValueError('Private restart cap exceeded; preserve files for review')
    metadata = dict(run=str(run), project_commit=(run / 'project-commit.txt').read_text().strip(),
                    hashes={str(p.relative_to(run)): sha(p) for p in files},
                    privacy='PRIVATE: contains audio, text, ledgers and logs; never Git/public Kaggle',
                    external_inputs='Retain official bounded source clips, full accepted mini and pinned wheelhouse separately')
    target = run.parent / f'{run.name}-PRIVATE-restart-{time.time_ns()}.tar.gz'
    with target.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as bundle:
        raw = json.dumps(metadata, indent=2).encode()
        info = tarfile.TarInfo('restart-index.json')
        info.size = len(raw)
        bundle.addfile(info, io.BytesIO(raw))
        for path in files:
            bundle.add(path, arcname='run/' + str(path.relative_to(run)), recursive=False)
    print(f'بستهٔ restart خصوصی با hash تمام فایل‌ها ساخته شد: {target}؛ برای انتشار عمومی نیست.')


def restore(source):
    with tarfile.open(source, 'r:gz') as bundle:
        members = bundle.getmembers()
        if (sum(m.size for m in members) > MAX_BYTES or
                any(not m.isfile() for m in members) or
                len({m.name for m in members}) != len(members)):
            raise ValueError('Unsafe/oversized restart bundle')
        index_member = bundle.getmember('restart-index.json')
        if index_member.size > 8 * 1024 * 1024:
            raise ValueError('Oversized restart index')
        index = json.load(bundle.extractfile(index_member))
        run = canonical(Path(index['run']))
        actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != index['project_commit']:
            raise ValueError('Checkout the exact original pushed SHA before restoring this run')
        expected_names = {'restart-index.json'} | {'run/' + name for name in index['hashes']}
        if {m.name for m in members} != expected_names:
            raise ValueError('Restart index/member mismatch')
        # Verify every member before creating destination; no extraction helpers or overwrites.
        for name, expected in index['hashes'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
                raise ValueError('Unsafe member path')
            stream = bundle.extractfile('run/' + name)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError('Corrupt restart member: ' + name)
        if run.exists() or run.is_symlink() or any(p.is_symlink() for p in run.parents):
            raise ValueError('Destination exists/is unsafe; do not merge, delete or overwrite it')
        run.mkdir(parents=True, exist_ok=False)
        for name in index['hashes']:
            target = run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output, bundle.extractfile('run/' + name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    output.write(block)
        print(f'کل restart پس از بررسی hash در مسیر canonical بازیابی شد: {run}؛ runهای دیگر دست‌نخورده‌اند.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('pack', 'restore'))
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or not Path('/kaggle/working').is_dir():
        raise RuntimeError('Private restart handling is Kaggle-only')
    globals()[args.command](args.path)


if __name__ == '__main__':
    main()
