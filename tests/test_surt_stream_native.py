"""Actual pinned GPU model checks. Run only via the authorized 1C-B Kaggle runner.

The runner injects the restored fixture; no implicit weight/audio download.
"""
import unittest


class NativeAcceptance(unittest.TestCase):
    fixture = None
    reports = []

    @classmethod
    def setUpClass(cls):
        if cls.fixture is None:
            raise unittest.SkipTest('Use scripts/surt_stream/run_b.py on Kaggle')

    def check(self, wave, label):
        from native_checks import check_case
        model, extractor, _ = self.fixture
        self.reports.append(check_case(model, extractor, wave, label))

    def test_full_partition_and_live_reset(self):
        self.check(self.fixture[2], 'real-train-mixture')

    def test_short_eof_and_last_frame(self):
        wave = self.fixture[2]
        # fbank, 9-frame subsampling, 16-frame mask stride, 39-frame Zip chunk edges
        for samples in (0, 1, 79, 80, 279, 280, 399, 400, 1279, 1280, 1359,
                        1360, 1519, 1520, 5119, 5120, 6159, 6160, 6239, 6240, 6401):
            with self.subTest(samples=samples):
                self.check(wave[:samples], f'prefix-{samples}')

    def test_pre_eof_and_bounded_long_input(self):
        import torch
        import numpy as np
        from native import NativeSURT
        model, extractor, wave = self.fixture
        stream = NativeSURT(model, extractor)
        checkpoints = []
        with torch.inference_mode():
            # Generate one packet at a time; never build a long recording/history.
            for start in range(0, 120 * 16000, 1600):
                packet = wave[np.arange(start, start+1600) % len(wave)]
                y, events = stream.push(packet)
                self.assertTrue(torch.isfinite(y).all().item())
                if start % 128000 == 0:
                    checkpoints.append(dict(samples=start+1600, bytes=stream.acoustic_bytes(),
                                            emitted_frames=stream.recognition.emitted))
                self.assertLess(stream.acoustic_bytes(), 16 * 1024 * 1024)
            stream.push([], final=True)
        self.assertTrue(any(c['emitted_frames'] > 0 for c in checkpoints[:3]))
        # Eight-second checkpoints have the same mask/chunk phase.
        self.assertEqual(len({c['bytes'] for c in checkpoints[1:]}), 1)
        self.assertEqual(stream.recognition.emitted, (max(0, (12000-7)//2)+1)//2)
        # Scalar counters may grow; tensor storage capacity may not.
        self.assertLessEqual(max(c['bytes'] for c in checkpoints), 16 * 1024 * 1024)
        self.reports.append(dict(label='120s-repeated-train-state-stress',
                                 purpose='bounded-state stress; not long-session recognition',
                                 checkpoints=checkpoints, peak_bytes=stream.peak_state_bytes))
