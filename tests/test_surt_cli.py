"""Exercise the public file inference command with a labeled test backend."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from streaming_asr.audio.loader import write_audio

ROOT = Path(__file__).resolve().parents[1]


def test_file_inference_collects_all_deltas_and_eof_output(tmp_path):
    audio = tmp_path / "short.wav"
    write_audio(audio, np.ones(100, np.float32) * .1, 16000)
    (tmp_path / "test_backend.py").write_text('''
from streaming_asr.models.multitalker import BackendOutput
class Backend:
    num_channels = 2
    chunk_frames = 32
    right_context_frames = 0
    frame_shift_samples = 160
    max_state_bytes = 1024
    def reset(self): pass
    def step(self, features, *, valid_frames, final):
        assert valid_frames == 1 and final
        return BackendOutput(("short", "second"), (0, 0))
    def finish(self): return BackendOutput(("tail", ""), (0, 0))
def create(config): return Backend()
''')
    override = tmp_path / "override.yaml"
    override.write_text("model:\n  backend_factory: test_backend:create\n")
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, "scripts/run_streaming.py", "--audio", str(audio),
                             "--override", str(override)], cwd=ROOT, env=environment,
                            text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    assert output["hypothesis_streams"] == ["short tail", "second"]
    assert output["audio_seconds"] == 100 / 16000
    assert output["first_output_available_sample"] == 100
