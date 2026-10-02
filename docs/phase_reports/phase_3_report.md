# Phase 3 Report — Synthetic Overlap Dataset Generation

## Implemented

Deterministic two-speaker mixture generation with source timing, transcripts, speaker IDs, controlled low/medium/high/control overlap regimes, relative gain, RMS normalization, peak-safe output, JSONL manifests, validation, statistics, and speaker-leakage checks.

## Files changed

- `src/streaming_asr/datasets/overlap_generator.py`
- `src/streaming_asr/datasets/manifests.py`
- `src/streaming_asr/datasets/validators.py`
- `src/streaming_asr/audio/loader.py`
- `scripts/generate_overlap.py`, `scripts/smoke_test.py`
- `configs/overlap_dataset.yaml`
- `notebooks/04_overlap_generation.ipynb`
- Phase 3 tests and documentation

## Dataset

Configured source: LibriSpeech via the existing Hugging Face/local/Kaggle backend. The executed run used tiny deterministic local fixtures and downloaded no LibriSpeech data.

## Generation configuration

- Seed: 42
- Two speakers
- 16 kHz mono WAV
- Four mixtures per split
- Regimes: low 0.25, medium 0.50, high 0.80, control 0.0
- Relative gains: 0 dB and -3 dB
- Ratio definition: overlap duration divided by shorter source duration

## Results

Executed demo: 12 mixtures total, 9 overlapping and 3 controls. All three splits contained four mixtures and all four configured regimes. Validation errors: 0. Total generated duration: 13.925 seconds.

## Validation

Passed audio readability, sample-rate, finite waveform, timing, overlap arithmetic, metadata, manifest, and speaker-leakage checks on the generated fixtures.

## Tests

```text
pytest -q: PASS (17 tests)
Phase 0 smoke test: PASS
Phase 3 overlap smoke test: PASS
Manifest validation: PASS (12 records)
Notebook JSON validation: PASS
```

## Issues

Real LibriSpeech generation was not executed because the external dataset backend and data were not available locally.

## Limitations

These are controlled synthetic mixtures, not real conversational overlap. No OSD, overlap recognition, separation, diarization, or ASR evaluation was added.

## Research implications

Later phases can consume exact source timing and transcript ground truth for WhisperRT-under-overlap and future OSD experiments.

## Next phase

Proceed to **Phase 4 — WhisperRT Under Overlap Baseline** only after selecting and running a real or attached LibriSpeech source subset:

```bash
python3 scripts/generate_overlap.py --config configs/overlap_dataset.yaml
```

