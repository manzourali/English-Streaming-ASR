"""Check exact upstream bytes, then apply only the two recorded recipe diffs."""
import argparse
import json
import subprocess
from pathlib import Path

from common import sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('icefall', type=Path)
    parser.add_argument('logs', type=Path)
    args = parser.parse_args()
    pins = json.loads((ROOT / 'configs/surt_base/pins.json').read_text())
    revision = subprocess.check_output(['git', '-C', str(args.icefall), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == pins['icefall_revision'], revision
    recipe = args.icefall / 'egs/libricss/SURT/dprnn_zipformer'
    # Check original blobs, independent of whether the known patch was already applied.
    for name, digest in pins['source_sha256'].items():
        rel = 'egs/libricss/SURT/dprnn_zipformer/' + name
        blob = subprocess.check_output(['git', '-C', str(args.icefall), 'show', f'HEAD:{rel}'])
        import hashlib
        assert hashlib.sha256(blob).hexdigest() == digest, name
    for name in ('surt-simple-sampler.patch', 'surt-base-pretrained.patch'):
        patch = ROOT / 'docs/patches' / name
        assert sha256(patch) == pins['patch_sha256'][name], name
        command = ['git', '-C', str(args.icefall), 'apply']
        check = subprocess.run([*command, '--check', str(patch)], capture_output=True, text=True)
        if check.returncode == 0:
            subprocess.run([*command, str(patch)], check=True)
        else:
            # Reruns accept only the exact reverse-applicable patch.
            subprocess.run([*command, '--reverse', '--check', str(patch)], check=True)
    for name, digest in pins['patched_source_sha256'].items():
        assert sha256(recipe / name) == digest, f'Unexpected worktree source: {name}'
    diff = subprocess.check_output(['git', '-C', str(args.icefall), 'diff'], text=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    (args.logs / 'icefall-applied.patch').write_text(diff)
    (args.logs / 'source-hashes.txt').write_text('\n'.join(
        f'{sha256(p)}  {p}' for p in [*sorted((ROOT / 'docs/patches').glob('surt-*.patch')),
                                    recipe / 'pretrained.py', recipe / 'train.py',
                                    recipe / 'asr_datamodule.py']) + '\n')


if __name__ == '__main__':
    main()
