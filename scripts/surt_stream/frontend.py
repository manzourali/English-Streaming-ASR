"""Bounded buffered Kaldi fbank prototype, preserving snip_edges=False framing.

This is an incremental FEATURE component, not a stateful SURT model. The
extractor recomputes only a bounded packet/overlap buffer, not the recording.
"""


class BufferedFbank:
    shift = 160
    window = 400
    # Kaldi frame i starts at i*160 + 80 - 200 = i*160 - 120.
    first_end = 280

    def __init__(self, extractor, max_packet_samples=3200):
        import numpy as np
        self.np = np
        self.extractor = extractor
        self.max_packet_samples = max_packet_samples
        self.reset()

    def reset(self):
        self.buffer = self.np.empty(0, dtype=self.np.float32)
        self.base = self.total = self.emitted = self.peak_samples = 0
        self.finished = False

    def push(self, wave, final=False):
        np = self.np
        if self.finished:
            if final and len(wave) == 0:
                return np.empty((0, 80), dtype=np.float32)
            raise RuntimeError('Feature stream already finalized; reset before another recording')
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or len(wave) > self.max_packet_samples or not np.isfinite(wave).all():
            raise ValueError('Expected a finite mono packet within the packet budget')
        self.buffer = np.concatenate((self.buffer, wave))
        self.total += len(wave)
        self.peak_samples = max(self.peak_samples, len(self.buffer))
        available = ((self.total + self.shift // 2) // self.shift if final else
                     max(0, (self.total - self.first_end) // self.shift + 1))
        if available > self.emitted:
            computed = self.extractor.extract(self.buffer, 16000)
            start = self.emitted - self.base // self.shift
            stop = available - self.base // self.shift
            if len(computed) < stop:
                raise RuntimeError('Kaldi framing contract mismatch; do not pad or drop frames')
            emitted = computed[start:stop].copy()
        else:
            emitted = np.empty((0, 80), dtype=np.float32)
        self.emitted = available
        # Two historical shifts preserve the complete next 400-sample window.
        keep_from = max(0, (self.emitted - 2) * self.shift)
        self.buffer = self.buffer[keep_from - self.base:].copy()
        self.base = keep_from
        if final:
            self.finished = True
            self.buffer = np.empty(0, dtype=np.float32)
        if self.peak_samples > self.max_packet_samples + 4 * self.shift:
            raise RuntimeError('Feature buffer exceeded its constant memory bound')
        return emitted
