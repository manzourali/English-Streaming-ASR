"""Restart and evidence utilities; no model, audio processing or network imports."""
import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = json.loads((ROOT / 'configs/surt_stream/pins.json').read_text())
MINI_PINS = json.loads((ROOT / 'configs/surt_mini/pins.json').read_text())
BASE_PINS = json.loads((ROOT / 'configs/surt_base/pins.json').read_text())
CANONICAL = Path('/kaggle/working/surt-mini-1b/runs') / PINS['accepted_mini_run']


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    # Every stage is append-only; a failed attempt is archived, not overwritten.
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def kaggle_only():
    if not Path('/kaggle/working').is_dir() or sys.platform != 'linux':
        raise RuntimeError('Experimental execution and restoration are Kaggle-only')


def verify_mini(directory):
    result = read(directory / 'result.json')
    expected_contract = dict(optimizer_steps=1, strict_reload=True,
                             checkpoint_sha256=PINS['checkpoint_sha256'],
                             tokenizer_sha256=PINS['tokenizer_sha256'])
    mismatches = []
    for key, expected in expected_contract.items():
        actual = result.get(key)
        matches = actual is True if key == 'strict_reload' else actual == expected
        if not matches:
            observed = repr(actual) if key in result else '<missing>'
            mismatches.append(f'{key}: expected={expected!r}; actual={observed}')
    if mismatches:
        raise RuntimeError(
            f'Not the reviewed 1B checkpoint/tokenizer contract: {directory / "result.json"}\n'
            + '\n'.join(mismatches)
            + '\nExpected reviewed run: ' + PINS['accepted_mini_run']
            + '\nCompare this result.json with the reviewed saved 1B bundle. '
              'Do not change pins or overwrite existing runs to bypass verification.')
    expected_files = dict(PINS['reviewed_metadata_sha256'], **{
        'mini-one-update.pt': PINS['checkpoint_sha256'],
        'tokenizer/bpe.model': PINS['tokenizer_sha256']})
    for relative, expected in expected_files.items():
        actual = digest(directory / relative)
        if actual != expected:
            raise RuntimeError(f'1B hash mismatch: {directory / relative}; '
                               f'expected={expected}; actual={actual}')
    selected, mixtures = read(directory / 'selection.json'), read(directory / 'mixtures.json')
    if len(selected) != 8 or len(mixtures) != 4:
        raise RuntimeError('Incomplete mini selection/mixtures')
    for key in ('speaker_hash', 'source_id', 'original_sha256', 'waveform_sha256'):
        if len({row[key] for row in selected}) != 8:
            raise RuntimeError(f'Mini collision: {key}')
    if {split: sum(row['split'] == split for row in selected)
            for split in ('train', 'validation', 'test')} != {'train': 4, 'validation': 2, 'test': 2}:
        raise RuntimeError('Mini split contract changed')
    for row in selected + mixtures:
        relative = Path(row['audio']).relative_to(CANONICAL)
        if '..' in relative.parts or digest(directory / relative) != row.get('file_sha256', row.get('sha256')):
            raise RuntimeError(f'Mini audio changed: {relative}')
    for name in ('model-arguments.json', 'feature-contract.json', 'tokenizer-report.json',
                 'update-report.json', 'transfer-report.json', 'runtime.json',
                 'inspection-ledger.json', 'exposure-ledger.json', 'tokenizer/tokens.txt',
                 'features/train.jsonl.gz', 'features/validation.jsonl.gz', 'features/test.jsonl.gz'):
        if not (directory / name).is_file():
            raise RuntimeError(f'Missing restart input: {name}')
    if len(list((directory / 'features').rglob('*.npy'))) < 8:
        raise RuntimeError('Mini feature arrays missing; review archives are not restart bundles')
    if directory.is_symlink() or any(path.is_symlink() for path in directory.rglob('*')):
        raise RuntimeError('Restore input must contain regular files/directories, not symlinks')
    return dict(run=directory.name, checkpoint_sha256=PINS['checkpoint_sha256'],
                tokenizer_sha256=PINS['tokenizer_sha256'],
                auditory_confirmation=PINS['auditory_confirmation'],
                files={str(path.relative_to(directory)): digest(path)
                       for path in sorted(directory.rglob('*'))
                       if path.is_file() and path.name != '.DS_Store'})


def restore(source):
    kaggle_only()
    if CANONICAL.exists():
        return verify_mini(CANONICAL)
    if source is None or not source.is_dir():
        raise RuntimeError('Attach the complete private mini-20261119-d folder and set MINI_INPUT; no redownload')
    verify_mini(source)
    CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    # No merges, overwrites or deletion if a copy fails midway.
    shutil.copytree(source, CANONICAL, ignore=shutil.ignore_patterns('.DS_Store'))
    return verify_mini(CANONICAL)


def archive(run):
    kaggle_only()
    files = [path for path in run.rglob('*') if path.is_file() and not path.is_symlink()
             and path.suffix in ('.json', '.txt', '.log', '.csv', '.py', '.patch')]
    source_files = [path for folder in ('scripts/surt_stream', 'configs/surt_stream',
                                      'scripts/surt_base', 'configs/surt_base', 'configs/surt_mini')
                    for path in (ROOT / folder).rglob('*')
                    if path.is_file() and '__pycache__' not in path.parts]
    source_files += [ROOT / name for name in ('master_prompt.md',
                     'docs/KAGGLE_SURT_STREAMING_RUNBOOK_FA.md',
                     'docs/KAGGLE_SURT_STREAMING_1CB_RUNBOOK_FA.md',
                     'docs/SURT_1CB_STATE_CONTRACT.md',
                     'tests/test_surt_stream_native.py',
                     'tests/test_surt_stream_kaldifeat.py',
                     'docs/experiments/surt_mini_1b_review_2026-11-19.md')]
    if sum(path.stat().st_size for path in files + source_files) > PINS['max_evidence_bytes']:
        raise RuntimeError('Evidence exceeds 20MiB; preserve all files and review the size')
    target = run.parent / (run.name + '-review.tar.gz')
    serial = 1
    while target.exists():
        target = run.parent / f'{run.name}-review-{serial}.tar.gz'
        serial += 1
    items = [(path, 'run/' + str(path.relative_to(run))) for path in sorted(files)]
    items += [(path, 'project/' + str(path.relative_to(ROOT))) for path in sorted(source_files)]
    metadata = dict(hashes={name: digest(path) for path, name in items},
                    native_streaming_gate=(read(run / 'result.json').get('gate_1c', 'OPEN')
                                           if (run / 'result.json').exists() else 'OPEN'),
                    result_exists=(run / 'result.json').exists())
    with target.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as bundle:
        for path, name in items:
            bundle.add(path, arcname=name, recursive=False)
        raw = (json.dumps(metadata, indent=2) + '\n').encode()
        info = tarfile.TarInfo('review.json')
        info.size = len(raw)
        bundle.addfile(info, io.BytesIO(raw))
    return target


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    restore_parser = commands.add_parser('restore')
    restore_parser.add_argument('--input', type=Path)
    restore_parser.add_argument('--run', required=True, type=Path)
    archive_parser = commands.add_parser('archive')
    archive_parser.add_argument('run', type=Path)
    args = parser.parse_args()
    if args.command == 'restore':
        report = restore(args.input)
        dump(args.run / 'restored-inputs.json', report)
        print('Verified existing/restored 1B inputs; no model or dataset download')
    else:
        print(archive(args.run))


if __name__ == '__main__':
    main()
