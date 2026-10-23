import numpy as np
import pytest
from streaming_asr.audio.features import StreamingFbank


def test_filterbank_shape_floor_and_frequency_response():
    extractor = StreamingFbank()
    silence = extractor.accept(np.zeros(400, np.float32))
    assert silence.shape == (1, 80)
    np.testing.assert_allclose(silence, np.log(np.finfo(np.float32).eps))
    low = StreamingFbank().accept(np.sin(2 * np.pi * 300 * np.arange(400) / 16000).astype(np.float32))
    high = StreamingFbank().accept(np.sin(2 * np.pi * 3000 * np.arange(400) / 16000).astype(np.float32))
    assert low.argmax() < high.argmax()


def test_optional_kaldi_numerical_parity_for_complete_frames():
    torch = pytest.importorskip("torch")
    kaldi = pytest.importorskip("torchaudio.compliance.kaldi")
    samples = np.random.default_rng(23).normal(0, .1, 2000).astype(np.float32)
    actual = StreamingFbank().accept(samples)
    expected = kaldi.fbank(torch.from_numpy(samples * 32768).unsqueeze(0),
        sample_frequency=16000, frame_length=25, frame_shift=10, num_mel_bins=80,
        low_freq=20, high_freq=7600, dither=0, snip_edges=True,
        window_type="povey", use_energy=False, energy_floor=0).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
