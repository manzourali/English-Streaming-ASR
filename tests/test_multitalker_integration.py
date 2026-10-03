"""Optional real-model test; excluded unless an external SURT override is supplied."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_configured_surt_decoder_processes_one_manifest_sample():
    override = os.environ.get("STREAMING_ASR_SURT_OVERRIDE")
    if not os.environ.get("STREAMING_ASR_RUN_SURT_INTEGRATION") or not override:
        pytest.skip("set STREAMING_ASR_RUN_SURT_INTEGRATION=1 and STREAMING_ASR_SURT_OVERRIDE=<yaml>")
    from streaming_asr.audio.loader import load_audio
    from streaming_asr.audio.stream import AudioStream
    from streaming_asr.datasets.manifests import read_overlap_manifest
    from streaming_asr.models.multitalker import SURT2WindowedASR
    from streaming_asr.utils.config import Config

    config = Config.from_yaml(ROOT / "configs/multitalker.yaml", override).to_dict()
    manifest = Path(config["data"]["manifest"])
    if not manifest.is_absolute(): manifest = ROOT / manifest
    record = read_overlap_manifest(manifest)[0]
    samples, metadata = load_audio(record.audio_path)
    model = SURT2WindowedASR.from_config(config["model"])
    model.start()
    chunk_samples = metadata.sample_rate * int(config["model"]["streaming"]["chunk_duration_ms"]) // 1000
    updates = [model.process(chunk) for chunk in AudioStream.from_array(samples, metadata.sample_rate, chunk_samples)]
    final = model.finalize()
    assert final is not None
    assert all(update.streaming_mode == "low_latency_windowed" for update in updates)
