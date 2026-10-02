import numpy as np

from streaming_asr.audio.loader import load_audio
from streaming_asr.datasets.manifests import read_overlap_manifest
from streaming_asr.datasets.overlap_generator import OverlapGenerator, SourceUtterance, calculate_overlap
from streaming_asr.datasets.validators import overlap_statistics, validate_overlap_records


def _source(source_id, speaker, length=100):
    return SourceUtterance(source_id, speaker, np.ones(length, dtype=np.float32) * 0.1, 100, source_id, f"{source_id}.wav")


def test_overlap_calculation_cases():
    assert calculate_overlap(0, 1, 2, 3)[2] == 0
    assert calculate_overlap(0, 2, 1, 3)[2] == 1
    assert calculate_overlap(0, 3, 1, 2)[2] == 1


def test_generator_metadata_audio_and_determinism(tmp_path):
    generator = OverlapGenerator(tmp_path, sample_rate=100, seed=42)
    record = generator.generate_pair(_source("a", "A"), _source("b", "B"), split="test", regime="medium", target_ratio=0.5)
    assert record.overlap_ratio == 0.5
    assert not validate_overlap_records([record], check_source_files=False)
    samples, metadata = load_audio(record.audio_path)
    assert metadata.sample_rate == 100 and np.isfinite(samples).all()
    assert overlap_statistics([record])["num_mixtures"] == 1


def test_same_speaker_rejected(tmp_path):
    generator = OverlapGenerator(tmp_path, sample_rate=100)
    try:
        generator.generate_pair(_source("a", "A"), _source("b", "A"), split="test", regime="x", target_ratio=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("same-speaker pair should be rejected")
