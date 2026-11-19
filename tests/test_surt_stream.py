"""Offline mock tests for the 1C harness. No model/CUDA/audio experiments.

Written for later explicit authorization; they are not a Kaggle success claim.
"""
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('stream_' + name, ROOT / 'scripts/surt_stream' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockFrameExtractor:
    """Index-sensitive stand-in, not a Kaldi feature or recognition oracle."""
    def extract(self, wave, rate):
        count = (len(wave) + 80) // 160
        if not count:
            return np.empty((0, 80), dtype=np.float32)
        end = (count - 1) * 160 + 280
        padded = np.pad(wave, (120, max(0, end - len(wave))), mode='symmetric')
        return np.array([np.repeat(padded[i * 160:i * 160 + 400].sum(), 80)
                         for i in range(count)], dtype=np.float32)


@pytest.mark.parametrize('samples', [0, 1, 79, 80, 159, 160, 399, 400, 5119, 5120, 5361, 16001])
def test_buffered_feature_partition_reset_eof(samples):
    frontend = load('frontend')
    extractor = MockFrameExtractor()
    wave = np.arange(samples, dtype=np.float32) / 16000
    reference = extractor.extract(wave, 16000)
    stream = frontend.BufferedFbank(extractor)
    for packet in (1, 173, 1600, 3200):
        stream.reset()
        parts = [stream.push(wave[i:i + packet]) for i in range(0, len(wave), packet)]
        parts.append(stream.push([], final=True))
        assert np.array_equal(np.concatenate(parts), reference)
        assert stream.push([], final=True).shape == (0, 80)
        assert len(stream.buffer) == 0
        assert stream.peak_samples <= 3840
        with pytest.raises(RuntimeError, match='finalized'):
            stream.push([0.0])


def test_feature_buffer_stays_bounded_without_output_history():
    stream = load('frontend').BufferedFbank(MockFrameExtractor())
    for unused in range(2000):
        stream.push(np.zeros(1600, dtype=np.float32))
        assert len(stream.buffer) <= 640
    assert stream.peak_samples <= 3840
    stream.push([], final=True)
    assert stream.emitted == (stream.total + 80) // 160


def test_packet_budget_and_nonfinite_are_rejected():
    stream = load('frontend').BufferedFbank(MockFrameExtractor())
    for packet in (np.zeros(3201), np.array([np.nan]), np.zeros((2, 2))):
        with pytest.raises(ValueError, match='packet budget'):
            stream.push(packet)


def test_restore_reuses_verified_canonical_without_merging(monkeypatch, tmp_path):
    support = load('support')
    canonical = tmp_path / 'canonical'
    canonical.mkdir()
    marker = canonical / 'previous'
    marker.write_text('preserve')
    monkeypatch.setattr(support, 'CANONICAL', canonical)
    monkeypatch.setattr(support, 'kaggle_only', lambda: None)
    calls = []
    monkeypatch.setattr(support, 'verify_mini', lambda path: calls.append(path) or {'verified': True})
    assert support.restore(tmp_path / 'missing-input') == {'verified': True}
    assert calls == [canonical] and marker.read_text() == 'preserve'


def test_corrupt_existing_restore_is_preserved(monkeypatch, tmp_path):
    support = load('support')
    canonical = tmp_path / 'canonical'
    canonical.mkdir()
    marker = canonical / 'previous'
    marker.write_text('preserve')
    monkeypatch.setattr(support, 'CANONICAL', canonical)
    monkeypatch.setattr(support, 'kaggle_only', lambda: None)

    def reject(path):
        raise RuntimeError('1B hash mismatch')

    monkeypatch.setattr(support, 'verify_mini', reject)
    with pytest.raises(RuntimeError, match='hash mismatch'):
        support.restore(tmp_path / 'alternative')
    assert marker.read_text() == 'preserve'


@pytest.mark.parametrize('field,actual', [
    ('optimizer_steps', 2), ('strict_reload', False), ('strict_reload', 1),
    ('checkpoint_sha256', 'different-checkpoint'),
    ('tokenizer_sha256', 'different-tokenizer'), ('tokenizer_sha256', None),
])
def test_mini_contract_reports_expected_actual_and_path(monkeypatch, tmp_path, field, actual):
    support = load('support')
    contract = dict(optimizer_steps=1, strict_reload=True,
                    checkpoint_sha256=support.PINS['checkpoint_sha256'],
                    tokenizer_sha256=support.PINS['tokenizer_sha256'])
    expected = contract[field]
    contract[field] = actual
    (tmp_path / 'result.json').write_text(json.dumps(contract))

    def unexpected_digest(path):
        pytest.fail('Invalid metadata must fail before reading checkpoint/audio files')

    monkeypatch.setattr(support, 'digest', unexpected_digest)
    with pytest.raises(RuntimeError) as error:
        support.verify_mini(tmp_path)
    message = str(error.value)
    assert str(tmp_path / 'result.json') in message
    assert f'{field}: expected={expected!r}; actual={actual!r}' in message
    assert 'Do not change pins' in message
    assert json.loads((tmp_path / 'result.json').read_text()) == contract


def test_mini_contract_reports_all_missing_fields(tmp_path):
    support = load('support')
    (tmp_path / 'result.json').write_text('{}')
    with pytest.raises(RuntimeError) as error:
        support.verify_mini(tmp_path)
    for field in ('optimizer_steps', 'strict_reload', 'checkpoint_sha256', 'tokenizer_sha256'):
        assert f'{field}: expected=' in str(error.value)
    assert str(error.value).count('actual=<missing>') == 4


def test_matching_mini_contract_still_checks_file_hashes(monkeypatch, tmp_path):
    support = load('support')
    contract = dict(optimizer_steps=1, strict_reload=True,
                    checkpoint_sha256=support.PINS['checkpoint_sha256'],
                    tokenizer_sha256=support.PINS['tokenizer_sha256'])
    (tmp_path / 'result.json').write_text(json.dumps(contract))
    monkeypatch.setattr(support, 'digest', lambda path: 'changed-file')
    with pytest.raises(RuntimeError, match='1B hash mismatch:') as error:
        support.verify_mini(tmp_path)
    assert 'expected=' in str(error.value) and 'actual=changed-file' in str(error.value)


def test_archive_append_only_and_excludes_weights_audio_features(monkeypatch, tmp_path):
    support = load('support')
    monkeypatch.setattr(support, 'kaggle_only', lambda: None)
    run = tmp_path / 'run'
    run.mkdir()
    for name in ('model.pt', 'clip.wav', 'features.npy', 'result.json'):
        (run / name).write_text('{}')
    first = support.archive(run)
    original = first.read_bytes()
    second = support.archive(run)
    assert first != second and first.read_bytes() == original
    with tarfile.open(second) as archive:
        names = archive.getnames()
        assert 'run/result.json' in names
        assert not any(name.endswith(('.pt', '.wav', '.npy')) for name in names)


def test_runner_rejects_local_execution_before_gpu_imports(monkeypatch, tmp_path):
    support, frontend = load('support'), load('frontend')
    monkeypatch.setitem(sys.modules, 'support', support)
    monkeypatch.setitem(sys.modules, 'frontend', frontend)
    probes = load('probes')
    monkeypatch.setitem(sys.modules, 'probes', probes)
    runner = load('run')

    def refuse():
        raise RuntimeError('Kaggle-only')

    monkeypatch.setattr(runner, 'kaggle_only', refuse)
    monkeypatch.setattr(sys, 'argv', ['run.py', str(tmp_path)])
    with pytest.raises(RuntimeError, match='Kaggle-only'):
        runner.main()
    assert not list(tmp_path.iterdir())
