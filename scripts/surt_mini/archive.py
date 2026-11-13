"""Append-only small evidence archive; audio, weights, caches and secrets excluded."""
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

from contracts import ROOT, dump, file_hash


def archive_run(run, icefall):
    target = run.parent / (run.name + '-review.tar.gz')
    serial = 1
    while target.exists():
        target = run.parent / f'{run.name}-review-{serial}.tar.gz'
        serial += 1
    inventory = {}
    for path in sorted(run.rglob('*')):
        if path.is_file() and not path.is_symlink():
            inventory[str(path.relative_to(run))] = dict(bytes=path.stat().st_size, sha256=file_hash(path))
    git = {}
    for name, directory in [('project', ROOT), ('icefall', icefall)]:
        git[name] = {}
        for key, argv in [('revision', ['rev-parse', 'HEAD']), ('status', ['status', '--short']),
                          ('diff', ['diff'])]:
            proc = subprocess.run(['git', '-C', str(directory), *argv], capture_output=True, text=True)
            git[name][key] = dict(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    files = []
    allowed = {
        'dataset-metadata.json', 'selection.json', 'selection-partial.json', 'fetch-budget.json',
        'runtime.json', 'tokenizer-report.json', 'feature-contract.json', 'mixtures.json',
        'update-started.json', 'model-arguments.json', 'transfer-report.json', 'update-input.json',
        'update-report.json', 'exposure-ledger.json', 'inspection-ledger.json', 'result.json', 'transcripts.txt',
    }
    for path in sorted(run.glob('*')):
        if path.name in allowed and path.is_file() and not path.is_symlink():
            files.append((path, 'run/' + path.name))
    for path in sorted((run / 'logs').rglob('*')):
        if path.suffix in ('.log', '.txt', '.csv', '.patch') and path.is_file() and not path.is_symlink():
            files.append((path, 'run/' + str(path.relative_to(run))))
    for path in sorted((run / 'tokenizer').glob('*')):
        if path.name in ('bpe.model', 'bpe.vocab', 'tokens.txt', 'train.txt') and not path.is_symlink():
            files.append((path, 'run/tokenizer/' + path.name))
    for directory in ('scripts/surt_mini', 'scripts/surt_base', 'configs/surt_mini', 'configs/surt_base', 'docs/patches'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and not path.is_symlink():
                files.append((path, 'project/' + str(path.relative_to(ROOT))))
    for name in ('docs/KAGGLE_SURT_MINI_PERSIAN_RUNBOOK_FA.md',
                 'docs/experiments/surt_mini_1b_preparation_2026-11-13.md'):
        path = ROOT / name
        if path.exists():
            files.append((path, 'project/' + name))
    if sum(path.stat().st_size for path, name in files) > 20 * 1024 * 1024:
        raise RuntimeError('Evidence exceeds 20MiB; preserve full logs and review before packaging')
    hashes = {name: file_hash(path) for path, name in files}
    with target.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
        metadata = {'actual-files.json': inventory, 'git.json': git, 'review-sha256.json': hashes,
                    'review-status.json': dict(gate_1b='pending Kaggle evidence and human review',
                                               result_exists=(run / 'result.json').exists())}
        for name, value in metadata.items():
            raw = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('icefall', type=Path)
    args = parser.parse_args()
    print(archive_run(args.run, args.icefall))


if __name__ == '__main__':
    main()
