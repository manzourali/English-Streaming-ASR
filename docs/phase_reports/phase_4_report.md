# Phase 4 Report — WhisperRT Under Synthetic Overlap Baseline

## Objective

Evaluate the unchanged single-stream WhisperRT streaming path on Phase 3 mixtures without adding overlap-aware processing.

## Experimental Setup

Configured model: `MLSpeech/WhisperRT-Streaming`. Configured audio: 16 kHz mono, 300 ms chunks, batch size one, VAD disabled. Input: Phase 3 JSONL manifests. The executed validation used the local Phase 3 demo data; the real model run was not available because the WhisperRT stack/model was unavailable.

## Results

Real WhisperRT ASR results: **NOT MEASURED**.

Input validation: 12 mixture records validated successfully. Overlap WER is intentionally not measured for single-stream overlapping hypotheses. Control WER is implemented but was not executed without the model.

## Baseline Comparison

Clean vs non-overlap vs overlap ASR comparison: **NOT MEASURED**.

## Failure Analysis

No model hypotheses were produced in this environment, so no failure category is claimed.

## Streaming Analysis

The implementation records incremental hypotheses, first-output latency, end-of-utterance latency, average chunk latency, output-update count, and RTF when WhisperRT is available. These values are **NOT MEASURED** in this run.

## Limitations

Phase 4 evaluates a single-stream recognizer on two-speaker mixtures; it does not represent explicit multi-talker recognition. No overlap WER is assigned without speaker-separated hypotheses or a justified mapping. No OSD, separation, diarization, or optimization was implemented.

## Research Implications

The result schema is ready to quantify degradation by overlap regime, overlap ratio, duration, and relative gain once the real model run is executed.

## Next Phase

Phase 5 — Overlap Speech Detection / OSD.

1. Files Created/Modified
Implemented:

- scripts/run_overlap_baseline.py
- configs/overlap_baseline.yaml
- src/streaming_asr/metrics/overlap_baseline.py
- notebooks/05_overlap_baseline.ipynb
- docs/phase_reports/phase_4_report.md
- Extended Phase 4 documentation, smoke tests, and unit tests

2. Baseline Architecture
Phase 3 mixture WAV
        ↓
Streaming chunks
        ↓
WhisperRT Streaming ASR
        ↓
Incremental hypotheses
        ↓
Per-example and condition-level metrics
VAD is disabled in the primary baseline.
2. Model Used
Configured model:
MLSpeech/WhisperRT-Streaming
Real model execution was not possible because PyTorch, the WhisperRT package, and model weights are unavailable in the current environment.
3. Dataset Used
Phase 3 synthetic-overlap manifests:

- 12 total mixtures
- Train, validation, and test splits
- Low, medium, high, and control conditions
- Source timing, transcripts, speaker IDs, overlap ratios, and relative gains preserved

5. Experiment Conditions

- 16 kHz mono audio
- 300 ms streaming chunks
- Batch size 1
- VAD disabled
- Single-stream WhisperRT
- Control WER supported using time-ordered concatenated references
- Overlap WER explicitly marked NOT_MEASURED

6. Commands Executed
python3 -m pytest -q
python3 scripts/generate_overlap.py --config configs/overlap_dataset.yaml --demo
python3 scripts/run_overlap_baseline.py --config configs/overlap_baseline.yaml --validate-only
python3 scripts/smoke_test.py --phase 4 --validate-overlap --config configs/overlap_baseline.yaml
2. Measured Results
Manifest validation:
12 records validated
0 validation errors
Real WhisperRT WER, RTF, latency, and hypotheses:
NOT MEASURED
The real benchmark fails clearly without the required model stack:
Phase 4 cannot start: WhisperRT requires torch, huggingface_hub, and the official WhisperRT package
3. Metric Definitions

- WER is computed only for non-overlap controls.
- Overlap mixtures retain both source references but receive no ordinary single-reference WER.
- RTF, first-output latency, finalization latency, average chunk latency, and incremental hypotheses are implemented for real model runs.

9. Test Results
pytest: PASS — 19 tests
Phase 1/2 regression: PASS
Phase 3 manifest validation: PASS
Phase 4 validation-only run: PASS
Phase 4 smoke test: PASS
Notebook validation: PASS
Real WhisperRT integration: NOT RUN
2. Failure Examples
No model hypotheses were produced, so no failure category is claimed.
3. Limitations

- No real WhisperRT benchmark executed
- No WER/RTF/latency measurements
- Single-stream recognition does not represent explicit multi-talker ASR
- No OSD, separation, diarization, or overlap optimization
    - **Unresolved Research Questions**:
The effect of overlap ratio, relative gain, and streaming chunk size remains experimentally unmeasured until WhisperRT and LibriSpeech/model artifacts are available.
