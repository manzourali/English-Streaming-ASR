import gzip
import json
from dataclasses import replace

import numpy as np
import pytest

from streaming_asr.datasets.icefall_adapter import convert
from streaming_asr.datasets.manifests import write_overlap_manifest
from streaming_asr.datasets.overlap_generator import OverlapGenerator, SourceUtterance


def _records(tmp_path):
    generator = OverlapGenerator(tmp_path, sample_rate=16000)
    first = SourceUtterance("a", "A", np.ones(1600), 16000, "سلام", "a.wav")
    second = SourceUtterance("b", "B", np.ones(1600), 16000, "درود", "b.wav")
    return [generator.generate_pair(first, second, split="test", regime="test", target_ratio=ratio, mixture_id=f"mix{index}") for index, ratio in enumerate((0, 0.2, 0.4))]


def _load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_three_mixtures_convert(tmp_path):
    records = _records(tmp_path)
    manifest = tmp_path / "manifests" / "test.jsonl"
    write_overlap_manifest(records, manifest)
    paths = convert(manifest, tmp_path / "lhotse", "test")
    cuts, supervisions = _load(paths["cuts"]), _load(paths["supervisions"])
    assert len(cuts) == 3 and len(supervisions) == 6
    assert [item["text"] for item in cuts[0]["supervisions"]] == ["سلام", "درود"]
    assert {item["speaker"] for item in cuts[0]["supervisions"]} == {"A", "B"}
    assert cuts[0]["recording"]["sampling_rate"] == 16000


def test_invalid_transcript_is_rejected(tmp_path):
    record = _records(tmp_path)[0]
    record = replace(record, sources=[replace(record.sources[0], transcript=" "), record.sources[1]])
    manifest = tmp_path / "manifests" / "test.jsonl"
    write_overlap_manifest([record], manifest)
    with pytest.raises(ValueError, match="transcript"):
        convert(manifest, tmp_path / "lhotse", "test")


def test_invalid_length_and_overlap_warning(tmp_path):
    record = _records(tmp_path)[0]
    manifest = tmp_path / "manifests" / "test.jsonl"
    write_overlap_manifest([replace(record, duration=record.duration + 1)], manifest)
    with pytest.raises(ValueError, match="length"):
        convert(manifest, tmp_path / "lhotse", "test")
    write_overlap_manifest([replace(record, overlap_ratio=0.8)], manifest)
    with pytest.warns(UserWarning, match="overlap"):
        convert(manifest, tmp_path / "lhotse", "test")


def test_lhotse_can_load_cuts_if_installed(tmp_path):
    lhotse = pytest.importorskip("lhotse")
    manifest = tmp_path / "manifests" / "test.jsonl"
    write_overlap_manifest(_records(tmp_path), manifest)
    cuts = lhotse.CutSet.from_file(convert(manifest, tmp_path / "lhotse", "test")["cuts"])
    assert len(cuts) == 3
    assert all(len(cut.supervisions) == 2 for cut in cuts)
