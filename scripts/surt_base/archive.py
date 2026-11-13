"""Small review package, including failed runs; no weights, audio, or environments."""
import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from common import dump, sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=Path)
    parser.add_argument('icefall', type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    review = base / 'review'
    review.mkdir(exist_ok=True)
    for name in ('logs', 'results'):
        if (base / name).is_dir():
            shutil.copytree(base / name, review / name, dirs_exist_ok=True)
    assets = base / 'assets'
    if assets.exists():
        for path in assets.glob('*.json'):
            shutil.copyfile(path, review / path.name)
    for source in (ROOT / 'configs/surt_base', ROOT / 'scripts/surt_base'):
        shutil.copytree(source, review / source.relative_to(ROOT), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'docs/patches', review / 'patches', dirs_exist_ok=True)
    shutil.copyfile(ROOT / 'docs/KAGGLE_SURT_BASE_ENGLISH_RUNBOOK_FA.md', review / 'runbook.md')
    if (ROOT / 'bundle-provenance.json').exists():
        shutil.copyfile(ROOT / 'bundle-provenance.json', review / 'bundle-provenance.json')
    provenance = {}
    for name, repo in [('project', ROOT), ('icefall', args.icefall)]:
        provenance[name] = {}
        for key, command in [('revision', ['rev-parse', 'HEAD']),
                             ('status', ['status', '--short']), ('diff', ['diff', '--binary'])]:
            proc = subprocess.run(['git', '-C', str(repo), *command], capture_output=True, text=True)
            provenance[name][key] = dict(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    dump(review / 'git.json', provenance)
    inventory = {}
    for root in (assets, ROOT / 'configs/surt_base', ROOT / 'scripts/surt_base', ROOT / 'docs/patches',
                 args.icefall / 'egs/libricss/SURT/dprnn_zipformer'):
        if root.is_dir():
            for path in sorted(root.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    inventory[str(path)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    dump(review / 'actual-files.json', inventory)
    result = base / 'results/result.json'
    dump(review / 'review-status.json', dict(
         status='decode evidence available; human review required' if result.exists() else 'incomplete/failed',
         gate_1a='pending human inspection of transcripts, source/audio and contracts'))
    # Also hash archived files, excluding the hash manifest itself.
    dump(review / 'review-sha256.json', {str(p.relative_to(review)): sha256(p)
         for p in sorted(review.rglob('*')) if p.is_file() and p.name != 'review-sha256.json'})
    target = base.parent / 'surt-base-1a-review.tar.gz'
    with tarfile.open(target, 'w:gz') as archive:
        archive.add(review, arcname='surt-base-1a-review')
    print(f'Review archive: {target} ({target.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
