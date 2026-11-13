"""Instrumentation imported by the minimally patched official pretrained entry point."""
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

from common import dump, score, sha256

ROOT = Path(__file__).resolve().parents[2]
PINS = json.loads((ROOT / 'configs/surt_base/pins.json').read_text())
METRICS = {}


def preflight():
    import ctypes
    ctypes.CDLL('libnvrtc.so.12')
    for name in ('torch', 'torchaudio', 'k2', 'kaldifeat', 'lhotse', 'sentencepiece',
                 'meeteval.wer.wer', 'soundfile', 'einops', 'train', 'beam_search'):
        importlib.import_module(name)
    import numpy as np
    import torch
    from meeteval.wer import wer
    assert callable(wer.orc_word_error_rate)
    assert sys.version_info[:2] == (3, 12)
    expected = {'torch': '2.3.1+cu121', 'torchaudio': '2.3.1+cu121',
                'k2': '1.24.4.dev20241030+cuda12.1.torch2.3.1',
                'kaldifeat': '1.25.4.dev20240606+cuda12.1.torch2.3.1',
                'lhotse': '1.33.0', 'sentencepiece': '0.2.3', 'meeteval': '0.4.3'}
    actual = {name: importlib.metadata.version(name) for name in expected}
    assert actual == expected, (actual, expected)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.version.cuda == '12.1'
    # Unlike the tone fixture, pretrained.py actually uses torch.from_numpy.
    assert torch.from_numpy(np.zeros(4, dtype=np.float32)).numpy().shape == (4,)
    test = torch.ones(8, device='cuda')
    assert (test + test).sum().item() == 16
    torch.cuda.synchronize()
    METRICS.update(environment=dict(python=sys.version, executable=sys.executable,
                                  platform=platform.platform(), packages=actual,
                                  numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
                                  vram_bytes=torch.cuda.get_device_properties(0).total_memory))
    print(json.dumps(METRICS['environment'], indent=2))


def check_tokens(params, token_table):
    import sentencepiece as spm
    tokens = Path(params.tokens)
    bpe = tokens.with_name('bpe.model')
    for rel in ('data/lang_bpe_500/tokens.txt', 'data/lang_bpe_500/bpe.model'):
        path = tokens if rel.endswith('tokens.txt') else bpe
        assert sha256(path) == PINS['assets'][rel], path
    sp = spm.SentencePieceProcessor(model_file=str(bpe))
    assert params.blank_id == 0 and params.unk_id == 2
    assert params.vocab_size == sp.get_piece_size() == 500
    for i in range(sp.get_piece_size()):
        assert token_table[i] == sp.id_to_piece(i), (i, token_table[i], sp.id_to_piece(i))
    METRICS['tokens'] = dict(blank=0, unk=2, vocabulary=500,
                             disambiguation_excluded=['#0', '#1'])


def start():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def elapsed(begin):
    import torch
    torch.cuda.synchronize()
    return time.perf_counter() - begin


def check_checkpoint(checkpoint, model, path):
    import torch
    assert sha256(path) == PINS['assets']['exp/surt_base/pretrained.pt']
    expected, actual = model.state_dict(), checkpoint['model']
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    shapes = {k: [list(expected[k].shape), list(actual[k].shape)]
              for k in expected.keys() & actual.keys() if expected[k].shape != actual[k].shape}
    report = dict(missing=missing, unexpected=extra, shape_mismatches=shapes,
                  parameters=sum(p.numel() for p in model.parameters()),
                  state_tensors=len(actual), checkpoint_sha256=sha256(path))
    dump(Path(os.environ['SURT_BASE_RESULT']).with_name('checkpoint-contract.json'), report)
    assert not missing and not extra and not shapes, report
    assert all(not v.is_floating_point() or torch.isfinite(v).all().item()
               for v in actual.values())
    METRICS['checkpoint'] = report


def feature_contract(opts, features, lengths):
    import torch
    assert features.ndim == 3 and features.shape[0] == 3 and features.shape[2] == 80
    assert torch.isfinite(features).all().item()
    assert opts.frame_opts.frame_length_ms == 25 and opts.frame_opts.frame_shift_ms == 10
    METRICS['features'] = dict(shape=list(features.shape), lengths=lengths.tolist(),
                               options=str(opts), waveform_scaling='soundfile float32; no x32768',
                               padding='log(1e-10)')


def finish(params, hyps):
    import torch
    items = json.loads(Path(os.environ['SURT_BASE_INPUTS']).read_text())
    assert [i['audio'] for i in items] == params.sound_files
    assert len(items) == len(hyps) == 3
    rows = []
    for item, hypotheses in zip(items, hyps):
        assert len(hypotheses) == 2
        assert sha256(item['audio']) == item['sha256']
        rows.append(dict(input=item, hypotheses=hypotheses,
                         score=score(item['references'], hypotheses)))
    seconds = sum(i['duration'] for i in items)
    METRICS.update(audio_seconds=seconds,
                   model_load_count=1, warmup_passes=1, measured_passes=1,
                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   decode_rtf=METRICS['decode_seconds'] / seconds,
                   feature_plus_decode_rtf=(METRICS['feature_seconds'] + METRICS['decode_seconds']) / seconds)
    dump(os.environ['SURT_BASE_RESULT'], dict(status='real_decode_completed',
         protocol='official pretrained.py, patched diagnostics; whole-batch greedy_search',
         model_revision=PINS['model_revision'], icefall_revision=PINS['icefall_revision'],
         model_arguments=dict(params), metrics=METRICS, utterances=rows))
    lines = ['Bounded real English sanity check; no LibriCSS benchmark or Persian claim.']
    for row in rows:
        lines.append('\n' + row['input']['id'])
        refs = row['input']['references'] + [''] * (2 - len(row['input']['references']))
        for i, ref in enumerate(refs):
            lines.append(f'REF slot {i} -> branch {row["score"]["assignment"][i]}: {ref or "<empty>"}')
        for i, hyp in enumerate(row['hypotheses']):
            lines.append(f'HYP branch {i}: {hyp}')
        lines.append(json.dumps(row['score']))
    Path(os.environ['SURT_BASE_RESULT']).with_name('transcripts.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    preflight()
