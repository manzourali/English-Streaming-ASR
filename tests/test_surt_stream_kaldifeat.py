"""CPU backend regression checks, executed only in the pinned Kaggle runtime.

Run with stdlib unittest; pytest and dependency installation are not required.
Numerical fixtures test framing, not speech recognition quality.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(Path('/kaggle/working').is_dir(), 'Real kaldifeat checks are Kaggle-only')
class KaldifeatBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from lhotse.features.kaldifeat import KaldifeatFbank, KaldifeatFbankConfig

        cls.np, cls.torch = np, torch
        cls.extractor = KaldifeatFbank(KaldifeatFbankConfig(device='cpu'))
        frame = cls.extractor.config.to_dict()['frame_opts']
        if (frame['samp_freq'] != 16000 or frame['frame_shift_ms'] != 10
                or frame['frame_length_ms'] != 25 or frame['snip_edges']
                or frame['dither'] != 0):
            raise RuntimeError('Regression checks require the reviewed 1B frame contract')
        spec = importlib.util.spec_from_file_location(
            'boundary_frontend', ROOT / 'scripts/surt_stream/frontend.py')
        cls.frontend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.frontend)

    def wave(self, length):
        indices = self.np.arange(length, dtype=self.np.float32)
        return (self.np.sin(indices * 0.037) + 0.1 * self.np.cos(indices * 0.113)).astype('float32')

    def reference(self, wave):
        # Independent Kaldi ExtractWindow reflection rule. Compute real fbank
        # directly on complete windows, bypassing only the defective strider.
        windows = []
        for frame in range((len(wave) + 80) // 160):
            values = []
            for index in range(frame * 160 - 120, frame * 160 + 280):
                while index < 0 or index >= len(wave):
                    index = -index - 1 if index < 0 else 2 * len(wave) - 1 - index
                values.append(wave[index])
            windows.append(values)
        if not windows:
            return self.np.empty((0, 80), dtype=self.np.float32)
        frames = self.torch.from_numpy(self.np.asarray(windows, dtype=self.np.float32))
        with self.torch.inference_mode():
            return self.extractor.extractor.compute(frames, vtln_warp=1.0).numpy()

    def assert_features(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(self.np.isfinite(actual).all())
        self.np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=1e-4)

    def test_short_and_minimum_window(self):
        for length in (0, 1, 79, 80, 81, 119, 120, 139, 140, 159, 160,
                       239, 240, 279, 280, 399, 400, 401, 960):
            with self.subTest(samples=length):
                wave = self.wave(length)
                actual = self.frontend.extract_with_kaldi_edges(self.extractor, wave)
                self.assert_features(actual, self.reference(wave))

    def test_partition_reset_and_final_partial(self):
        for length in (80, 279, 280, 399, 400, 401, 1601, 5361):
            wave = self.wave(length)
            reference = self.reference(wave)
            stream = self.frontend.BufferedFbank(self.extractor)
            for partition in ((1,), (80,), (1600,), (173, 997, 3200, 1, 511)):
                with self.subTest(samples=length, partition=partition):
                    stream.reset()
                    outputs, cursor, index = [], 0, 0
                    while cursor < length:
                        size = partition[index % len(partition)]
                        outputs.append(stream.push(wave[cursor:cursor + size]))
                        cursor += size
                        index += 1
                    outputs.append(stream.push([], final=True))
                    self.assert_features(self.np.concatenate(outputs), reference)
                    self.assertEqual(len(stream.push([], final=True)), 0)
                    self.assertLessEqual(stream.peak_samples, 3840)

    def test_normal_offline_path_is_unchanged(self):
        for length in (400, 960, 5361):
            with self.subTest(samples=length):
                wave = self.wave(length)
                direct = self.extractor.extract(wave, 16000)
                self.assert_features(direct, self.reference(wave))
                self.np.testing.assert_array_equal(
                    self.frontend.extract_with_kaldi_edges(self.extractor, wave), direct)


if __name__ == '__main__':
    unittest.main(verbosity=2)
