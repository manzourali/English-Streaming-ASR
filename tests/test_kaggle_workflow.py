"""Data handoff and evaluation tests without network, a GPU, or ASR weights."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from streaming_asr.audio.loader import write_audio
from streaming_asr.datasets.librispeech import prepare_librispeech
from streaming_asr.datasets.loaders import iter_librispeech
from streaming_asr.datasets.manifests import read_overlap_manifest, read_manifest
from streaming_asr.datasets.overlap_generator import OverlapGenerator, SourceUtterance
from streaming_asr.datasets.validators import validate_overlap_records
from streaming_asr.evaluation.records import read_evaluation_records

ROOT = Path(__file__).resolve().parents[1]


def _librispeech_fixture(root, speakers=6):
    for number in range(speakers):
        identifier = f"{number}-1-0001"
        folder = root / "dev-clean" / str(number) / "1"
        write_audio(folder / f"{identifier}.wav", np.ones(160, np.float32) * .1, 16000)
        (folder / f"{number}-1.trans.txt").write_text(f"{identifier} HELLO WORLD\n")


def test_pilot_manifests_preserve_speakers_and_use_balanced_local_loading(tmp_path):
    sources = tmp_path / "LibriSpeech"
    _librispeech_fixture(sources)
    manifests = tmp_path / "clean_manifests"
    result = prepare_librispeech(sources, manifests, max_per_split=4)
    assert result["purpose"] == "development_only"
    groups = []
    for split in ("train", "validation", "test"):
        records = read_manifest(manifests / f"{split}.jsonl")
        assert len({r.metadata["speaker_id"] for r in records}) >= 2
        groups.append({r.metadata["speaker_id"] for r in records})
        examples = list(iter_librispeech({"backend": "local", "path": str(manifests / f"{split}.jsonl"), "max_samples": 1}))
        assert len(examples) == 1 and examples[0].speaker_id is not None
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_overlap_audio_paths_survive_kaggle_output_relocation(tmp_path):
    original = tmp_path / "working" / "bundle"
    sources = [SourceUtterance(str(i), str(i), np.ones(160, np.float32) * .1, 16000, "hello", f"source-{i}") for i in range(2)]
    OverlapGenerator(original).generate_dataset({"test": sources}, num_samples=2, regimes={"medium": .5})
    raw = json.loads((original / "manifests/test.jsonl").read_text().splitlines()[0])
    assert raw["audio_path"].startswith("audio/test/")
    attached = tmp_path / "input" / "new-slug"
    attached.parent.mkdir()
    shutil.move(str(original), attached)
    records = read_overlap_manifest(attached / "manifests/test.jsonl")
    assert not validate_overlap_records(records, check_source_files=False)


def test_native_bundle_inspection_does_not_label_raw_weights_ready(tmp_path):
    spec = importlib.util.spec_from_file_location("check_surt_assets", ROOT / "scripts/check_surt_assets.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path / "pretrained.pt").write_bytes(b"not-an-export")
    result = module.inspect_bundle(tmp_path)
    assert not result["ready"] and "mask.pt" in result["missing"]
    assert not module.inspect_bundle(None)["ready"]


def test_runner_records_are_directly_accepted_by_evaluator(tmp_path):
    import yaml
    sources = [SourceUtterance(str(i), str(i), np.ones(160, np.float32) * .1, 16000, "hello", f"source-{i}") for i in range(2)]
    dataset = tmp_path / "dataset"
    OverlapGenerator(dataset).generate_dataset({"test": sources}, num_samples=2, regimes={"medium": .5})
    (tmp_path / "contract_backend.py").write_text('''
from streaming_asr.models.multitalker import BackendOutput
class Backend:
    num_channels = 2
    chunk_frames = 32
    right_context_frames = 0
    frame_shift_samples = 160
    max_state_bytes = 1024
    def reset(self): pass
    def step(self, features, *, valid_frames, final): return BackendOutput(("hello", "hello"), (0, 0))
    def finish(self): return BackendOutput(("", ""), (0, 0))
def create(config): return Backend()
''')
    override = tmp_path / "inference.yaml"
    override.write_text(yaml.safe_dump({"runtime": {"environment": "local"},
        "experiment": {"system": "test_only_backend"},
        "data": {"manifest": str(dataset / "manifests/test.jsonl"), "dataset_id": "contract_fixture"},
        "model": {"backend_factory": "contract_backend:create"},
        "paths": {"outputs": str(tmp_path / "results"), "checkpoints": str(tmp_path / "checkpoints")}}))
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    subprocess.run([sys.executable, "scripts/run_multitalker.py", "--override", str(override)],
                   cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
    record_file = next((tmp_path / "results/predictions").glob("*_evaluation_records.jsonl"))
    records = read_evaluation_records(record_file)
    assert len(records) == 2 and all(r.error_count == 0 and r.reference_units == 2 for r in records)
    assert all(len(r.metadata["manifest_sha256"]) == 64 for r in records)
    evaluation = subprocess.run([sys.executable, "scripts/evaluate.py", "--override", str(override),
                                 "--records", str(record_file)], cwd=ROOT, env=environment,
                                check=True, capture_output=True, text=True)
    report = json.loads(evaluation.stdout)
    assert report["records"] == 2 and report["summaries"][0]["value"] == 0


def test_active_notebooks_compile_and_exclude_retired_router_commands():
    paths = [ROOT / "english-streaming-asr.ipynb", *sorted((ROOT / "notebooks").glob("*.ipynb"))]
    assert len(paths) == 5
    assert not (ROOT / "notebooks/12_adaptive_e2e_surt.ipynb").exists()
    for path in paths:
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            assert "--modes" not in source and "decoder_factory" not in source
            if cell["cell_type"] == "code":
                compile(source, str(path), "exec")
                assert cell["outputs"] == [] and cell["execution_count"] is None


def test_prepare_generate_and_training_validation_commands_handoff(tmp_path):
    import yaml
    sources = tmp_path / "LibriSpeech"
    _librispeech_fixture(sources)
    source_manifests = tmp_path / "source_manifests"
    subprocess.run([sys.executable, "scripts/prepare_dataset.py", "--source-root", str(sources),
                    "--output-root", str(source_manifests), "--max-per-split", "4"],
                   cwd=ROOT, check=True, capture_output=True)
    data = tmp_path / "data"
    override = tmp_path / "generation.yaml"
    override.write_text(yaml.safe_dump({"runtime": {"environment": "local"},
        "data": {"backend": "local", "manifest_root": str(source_manifests)},
        "generation": {"num_samples": 4, "max_source_samples": 4},
        "output": {"root": str(data)},
        "paths": {"outputs": str(tmp_path / "outputs"), "checkpoints": str(tmp_path / "checkpoints")}}))
    subprocess.run([sys.executable, "scripts/generate_overlap.py", "--override", str(override)],
                   cwd=ROOT, check=True, capture_output=True)
    metadata = json.loads((data / "dataset_info.json").read_text())
    assert metadata["data_kind"] == "speech_mixtures"
    assert metadata["source_preparation"]["purpose"] == "development_only"
    training = tmp_path / "training.yaml"
    training.write_text(yaml.safe_dump({"dataset": {f"{name}_manifest": str(data / "manifests" / f"{name}.jsonl")
                                                   for name in ("train", "validation", "test")}}))
    result = subprocess.run([sys.executable, "scripts/train_asr.py", "--override", str(training), "--validate-only"],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["examples"] == {"train": 4, "validation": 4, "test": 4}
