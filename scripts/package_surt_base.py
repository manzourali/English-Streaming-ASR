"""Build the small, uncommitted Kaggle tool bundle without model/audio downloads."""
import io
import json
import subprocess
import tarfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/surt_base'))
from common import sha256


def main():
    paths = []
    for directory in ('scripts/surt_base', 'configs/surt_base', 'docs/patches'):
        paths.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    paths += [ROOT / 'docs/KAGGLE_SURT_BASE_ENGLISH_RUNBOOK_FA.md',
              ROOT / 'scripts/package_surt_base.py']
    paths = sorted(paths)
    metadata = dict(project_head=subprocess.check_output(['git', '-C', str(ROOT),
                    'rev-parse', 'HEAD'], text=True).strip(),
                    git_status=subprocess.check_output(['git', '-C', str(ROOT),
                    'status', '--short'], text=True),
                    files={str(p.relative_to(ROOT)): sha256(p) for p in paths},
                    note='Workspace bundle; not a new commit; individual bytes are hashed.')
    dest = ROOT / 'checkpoints/surt-base-1a-tools.tar.gz'
    with tarfile.open(dest, 'w:gz') as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        data = (json.dumps(metadata, indent=2) + '\n').encode()
        item = tarfile.TarInfo('bundle-provenance.json')
        item.size = len(data)
        archive.addfile(item, io.BytesIO(data))
    print(f'{dest}\nsha256={sha256(dest)}\nbytes={dest.stat().st_size}')


if __name__ == '__main__':
    main()
