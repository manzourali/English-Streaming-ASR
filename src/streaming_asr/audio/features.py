"""Bounded Kaldi-style 80-bin log-Mel features for normalized 16 kHz PCM.

25 ms Povey windows, 10 ms shift, 512-point power FFT, 20--7600 Hz,
DC removal, 0.97 preemphasis, no dither, no energy coefficient. Interior
frames use snip-edges semantics. EOF pads only windows whose start is in
real audio, so even a stream shorter than one window reaches the model.
"""
from __future__ import annotations

import numpy as np


class StreamingFbank:
    sample_rate = 16000
    frame_length = 400
    frame_shift = 160
    feature_dim = 80

    def __init__(self):
        mel = lambda x: 1127.0 * np.log1p(x / 700.0)
        edges = np.linspace(mel(20), mel(7600), 82)
        frequencies = mel(np.arange(256) * 16000 / 512)
        self.filters = np.maximum(0, np.minimum(
            (frequencies[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None],
            (edges[2:, None] - frequencies[None, :]) / np.diff(edges)[1:, None],
        )).astype(np.float32)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85
        self.reset()

    def reset(self):
        self.buffer = np.empty(0, np.float32)
        self.finalized = False

    def accept(self, samples: np.ndarray, *, final: bool = False) -> np.ndarray:
        if self.finalized:
            raise RuntimeError("Feature stream is finalized")
        frames = []
        self.buffer = np.concatenate((self.buffer, samples))
        while len(self.buffer) >= self.frame_length or (final and len(self.buffer)):
            frame = np.pad(self.buffer[:400], (0, max(0, 400 - len(self.buffer)))) * 32768.0
            frame = frame - frame.mean()
            frame = frame - 0.97 * np.concatenate((frame[:1], frame[:-1]))
            power = np.abs(np.fft.rfft(frame * self.window, n=512))[:256] ** 2
            frames.append(np.log(np.maximum(self.filters @ power, np.finfo(np.float32).eps)))
            self.buffer = self.buffer[160:].copy()
        self.finalized = final
        return np.asarray(frames, dtype=np.float32).reshape(-1, 80)
