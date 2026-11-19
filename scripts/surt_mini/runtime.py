"""Read-only verification of the inherited environment, assets and recipe."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from contracts import PINS, ROOT, dump, file_hash
from dependencies import audit, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    if not Path('/kaggle/working').is_dir() or sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError('Use the successful isolated Kaggle Python 3.12 environment')
    baseline = Path('/kaggle/working/surt-base-1a')
    icefall = Path(os.environ['ICEFALL'])
    pins = json.loads((ROOT / 'configs/surt_base/pins.json').read_text())
    if subprocess.check_output(['git', '-C', str(icefall), 'rev-parse', 'HEAD'], text=True).strip() != pins['icefall_revision']:
        raise RuntimeError('Icefall revision mismatch')
    changed = set(subprocess.check_output(['git', '-C', str(icefall), 'diff', 'HEAD', '--name-only'], text=True).splitlines())
    expected_changes = {'egs/libricss/SURT/dprnn_zipformer/' + name
                        for name in ('asr_datamodule.py', 'pretrained.py')}
    if changed != expected_changes:
        raise RuntimeError(f'Unexpected Icefall modifications: {changed}')
    actual = {}
    for name, expected in PINS['source_sha256'].items():
        actual[name] = file_hash(icefall / name)
        if actual[name] != expected:
            raise RuntimeError(f'Recipe source changed: {name}')
    recipe = icefall / 'egs/libricss/SURT/dprnn_zipformer'
    for name, expected in pins['patched_source_sha256'].items():
        if file_hash(recipe / name) != expected:
            raise RuntimeError(f'Inherited patched source changed: {name}')
    for name, expected in pins['assets'].items():
        if file_hash(baseline / 'assets/model' / name) != expected:
            raise RuntimeError(f'Official baseline asset changed: {name}')
    result = json.loads((baseline / 'results/result.json').read_text())
    contract = json.loads((baseline / 'results/checkpoint-contract.json').read_text())
    if (result['status'] != 'real_decode_completed' or len(result['utterances']) != 3
            or contract['missing'] or contract['unexpected'] or contract['shape_mismatches']
            or contract['checkpoint_sha256'] != pins['assets']['exp/surt_base/pretrained.pt']):
        raise RuntimeError('Baseline technical gate not satisfied')
    dependency_report = audit()
    require(dependency_report)
    observed = {name: row['actual'] for name, row in dependency_report['packages'].items()}
    import numpy as np
    import soundfile as sf
    import torch
    import io
    import sentencepiece as spm

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert sf.__version__ == '0.14.0' and 'MP3' in sf.available_formats()
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype='float32'), 16000, format='WAV')
    buffer.seek(0)
    wave, rate = sf.read(buffer, dtype='float32')
    assert rate == 16000 and torch.from_numpy(wave).numpy().shape == (1600,)
    old = spm.SentencePieceProcessor(model_file=str(baseline / 'assets/model/data/lang_bpe_500/bpe.model'))
    table = dict(line.rsplit(' ', 1) for line in (baseline / 'assets/model/data/lang_bpe_500/tokens.txt').read_text().splitlines())
    assert old.get_piece_size() == 500 and int(table['<blk>']) == 0 and old.unk_id() == int(table['<unk>'])
    assert all(int(table[old.id_to_piece(index)]) == index for index in range(old.get_piece_size()))
    import inspect
    import lhotse.dataset.surt
    import lhotse.features.kaldifeat

    args.run.mkdir(parents=True, exist_ok=True)
    dump(args.run / 'runtime.json', dict(packages=observed, dependencies=dependency_report, source_hashes=actual,
         icefall_revision=pins['icefall_revision'], model_revision=pins['model_revision'],
         python=sys.version, executable=sys.executable, gpu=torch.cuda.get_device_name(0),
         soundfile_formats=sf.available_formats(), extra_installs=[],
         lhotse_source_hashes={module.__name__: file_hash(inspect.getfile(module))
                               for module in (lhotse.dataset.surt, lhotse.features.kaldifeat)},
         baseline_result_sha256=file_hash(baseline / 'results/result.json'),
         baseline_contract=contract, auditory_confirmation='user confirmed 2026-11-13'))


if __name__ == '__main__':
    main()
