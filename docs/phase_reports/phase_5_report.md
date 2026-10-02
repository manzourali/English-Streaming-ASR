# Phase 5 Report — Streaming Overlap Speech Detection

## Objective

Add an independent streaming overlap-speech detector with explicit `NO_SPEECH`, `SINGLE_SPEAKER`, and `OVERLAP` states. Phase 5 does not change WhisperRT, VAD, routing, diarization, or source separation.

## Candidate selection

The selected candidate is `SpectralHeuristicOSD`, exposed as the `heuristic` backend. It is a causal, dependency-free algorithmic baseline: fixed frames are buffered incrementally, speech energy gates the decision, and two or more dominant separated spectral peaks produce an overlap label. It uses no future frames, source timing, transcripts, oracle labels, or pretrained weights. A neural pretrained OSD backend was not claimed because no verified streaming OSD model and dependency stack were available in this environment. VAD alone was not selected because it cannot distinguish one active speaker from two.

## Dataset and protocol

The Phase 3 deterministic demo was regenerated locally: 12 mixtures total, with 4 test mixtures (3 overlapping and 1 control), 4.57 seconds of test audio, and 30 ms OSD frames. Ground truth was independently derived from each manifest's source start/end intervals: zero, one, or at least two active sources map to the three OSD states. Predictions were aligned to reference frames by reference-frame midpoint.

## Measured results

Command:

```bash
python3 scripts/generate_overlap.py --config configs/overlap_dataset.yaml --demo
python3 scripts/evaluate_osd.py --config configs/overlap_detection.yaml
```

| Metric | Result |
|---|---:|
| Evaluated frames | 154 |
| Overlap precision | 1.000 |
| Overlap recall | 0.636 |
| Overlap F1 | 0.778 |
| Mean event detection delay | 15 ms |
| Processing time | 0.0125 s |
| Audio duration | 4.57 s |
| OSD RTF | 0.00274 |

Confusion matrix:

```text
                 predicted
actual          no_speech  single_speaker  overlap
no_speech             0             0         0
single_speaker        0           121         0
overlap               0            12        21
```

The test split contains no silence-only frames, so the `NO_SPEECH` row is structurally zero rather than evidence of no false alarms in general. The 12 overlap misses are classified as `SINGLE_SPEAKER`; the detector produced no overlap false positives on this small synthetic split.

## Verification

- `python3 -m pytest -q`: **23 passed**
- Phase 3 demo generation: **PASS**, 0 manifest validation errors
- `python3 scripts/smoke_test.py --phase 5 --validate-osd --config configs/overlap_detection.yaml`: **PASS**
- Notebook JSON validation: **PASS**

## Limitations and readiness

This is a transparent baseline on synthetic mixtures, not a production-quality or pretrained OSD result. Spectral peaks are sensitive to speech harmonics, noise, reverberation, and relative gain; the demo does not establish real conversational-overlap performance. The detector interface and timing metrics are ready to feed a future routing experiment, but adaptive routing is intentionally not implemented in Phase 5.

## Files

- `src/streaming_asr/models/overlap_detector.py`
- `src/streaming_asr/metrics/osd.py`
- `scripts/evaluate_osd.py`
- `configs/overlap_detection.yaml`
- `notebooks/06_overlap_detection.ipynb`
- `tests/test_osd.py`

Implemented Phase 5 — Streaming Overlap Speech Detection.
Highlights:

- Added causal SpectralHeuristicOSD backend with:
  - NO_SPEECH
  - SINGLE_SPEAKER
  - OVERLAP
- Added independent timing-based ground-truth alignment and OSD metrics.
- Added evaluator, configuration, smoke validation, notebook, tests, and documentation.
- No WhisperRT, VAD, routing, diarization, or separation changes.
Measured demo results:
- Precision: 1.000
- Recall: 0.636
- F1: 0.778
- Mean detection delay: 15 ms
- OSD RTF: 0.00274
- Tests: 23 passed
Key files:
- [overlap_detector.py](../../src/streaming_asr/models/overlap_detector.py)
- [osd.py](../../src/streaming_asr/metrics/osd.py)
- [evaluate_osd.py](../../scripts/evaluate_osd.py)
- [overlap_detection.yaml](../../configs/overlap_detection.yaml)
- [Phase 5 report](../../docs/phase_reports/phase_5_report.md)
- [OSD notebook](../../notebooks/06_overlap_detection.ipynb)
Validation commands completed successfully:
python3 -m pytest -q
python3 scripts/check_environment.py
python3 scripts/generate_overlap.py --config configs/overlap_dataset.yaml --demo
python3 scripts/evaluate_osd.py --config configs/overlap_detection.yaml
python3 scripts/smoke_test.py --phase 5 --validate-osd --config configs/overlap_detection.yaml
