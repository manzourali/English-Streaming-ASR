# Execution matrix

Use this as the operational checklist. “Can skip?” means skip for a narrower experiment, not that the corresponding science is complete.

| Step | Phase | Notebook / script | Dataset | Model | GPU? | Input | Output | Can skip? |
| ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | `scripts/check_environment.py` | None | None | No | Installed package | Environment report | No |
| 2 | 0 | `scripts/smoke_test.py --phase 0` / notebook 00 | None | Dummy components | No | `configs/base.yaml` | Small run/validation record | No |
| 3 | 3 | `scripts/generate_overlap.py --demo` / notebook 04 | Built-in fixtures | None | No | `configs/overlap_dataset.yaml` | `outputs/synthetic_overlap/` | No for overlap phases |
| 4 | 2 | `scripts/run_vad.py` / notebook 03 | Synthetic demo or WAV | WebRTC VAD | No | VAD config | Metrics/run record | Yes for clean-only baseline |
| 5 | 5 | `scripts/evaluate_osd.py` / notebook 06 | Phase 3 manifest | Heuristic OSD | No | `outputs/synthetic_overlap/` | OSD JSONL/metrics/report | Yes for Phase 4 only |
| 6 | 1 | `scripts/run_streaming.py --max-samples 1` / notebook 02 | LibriSpeech | WhisperRT | Recommended | Hub/cache plus config | Predictions/metrics/report | No for real ASR path |
| 7 | 4 | `scripts/run_overlap_baseline.py --max-samples 1` / notebook 05 | Phase 3 manifest | WhisperRT | Recommended | Manifest + WhisperRT | Phase 4 metrics/report | Yes for clean baseline only |
| 8 | 6 | `scripts/run_adaptive.py --max-samples 1` / notebook 07 | Phase 3 manifest | WhisperRT, optional VAD/OSD | Recommended | Manifest + valid ASR runtime | Routing traces/metrics/report | Yes unless studying routing |
| 9 | 7 | `scripts/run_multitalker.py` / notebook 08 | Phase 3 manifest | External SURT | Likely | Private override/factory/checkpoint | Multi-stream records/PI-WER | Yes until SURT is available |
| 10 | 8 | `scripts/train_asr.py --validate-only` / notebook 09 | Phase 3 split manifests | External trainable SURT | Likely | Private backend override | Validation or external checkpoints | Yes; required only for fine-tuning |
| 11 | 9 | `scripts/evaluate.py` + `scripts/benchmark.py` / notebook 10 | Saved prediction records | None | No | Frozen config/JSONL records | Final aggregate reports | No for final claims |

## Minimum commands for the first session

```bash
!python scripts/check_environment.py
!python scripts/smoke_test.py --phase 0 --config configs/base.yaml
!python scripts/generate_overlap.py --config configs/overlap_dataset.yaml --demo
!python scripts/evaluate_osd.py --config configs/overlap_detection.yaml --max-samples 4
```

Only after those succeed should you configure the external WhisperRT stack and begin Step 6.
