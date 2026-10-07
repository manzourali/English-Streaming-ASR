# Kaggle notebook guide

Use these notebooks as explanatory launchers, not as a mandatory linear course. Run commands from a Kaggle notebook whose working directory is `/kaggle/working/English-Streaming-ASR`; the checked-in notebooks mostly contain configuration inspection and commented command examples.

| Notebook | Phase | Run? | Required inputs / purpose | Produces / resource | Independent? | Needed by |
| --- | ---: | --- | --- | --- | --- | --- |
| `00_environment.ipynb` | 0 | YES | Core install; model-free infrastructure check. | Prints pipeline state; CPU. | Yes. | Nothing. |
| `01_dataset_validation.ipynb` | — | NO | Placeholder with no validator code. Use scripts instead. | None; CPU. | Yes, but no value. | Nothing. |
| `02_whisperrt_baseline.ipynb` | 1 | AFTER SETUP | LibriSpeech, official runtime, Hub token/access, checkpoint. | Predictions/metrics/report; GPU recommended. | No: external asset setup. | Phase 4/6 comparison baseline. |
| `03_streaming_vad.ipynb` | 2 | YES, OPTIONAL DATA | WebRTC VAD; a real WAV is optional. | Metrics/run record; CPU. | Yes. | Phase 6 VAD gate. |
| `04_overlap_generation.ipynb` | 3 | YES | Demo fixtures or LibriSpeech; create controlled two-source mixtures. | `outputs/synthetic_overlap/`; CPU, potentially storage-heavy. | Yes for demo. | Phase 4–8 data. |
| `05_overlap_baseline.ipynb` | 4 | AFTER 3 + 1 | Valid Phase 3 manifest and WhisperRT. | Phase 4 summaries; GPU recommended. | No. | Phase 6/7/9 comparison. |
| `06_overlap_detection.ipynb` | 5 | AFTER 3 | Phase 3 timing labels; heuristic OSD. | JSONL/metrics/report; CPU. | No. | Predicted routing in Phase 6. |
| `07_adaptive_pipeline.ipynb` | 6 | AFTER 3 + 1 + 5 | Manifest, WhisperRT, optional VAD/OSD; compare three routing modes. | Routing traces/metrics/report; GPU for real ASR. | No. | Phase 7 integration/Phase 9 records. |
| `08_multitalker.ipynb` | 7 | BLOCKED UNTIL EXTERNAL SETUP | Phase 3 manifest plus private pinned SURT factory/checkpoint. | PI-WER records; GPU likely. | No. | True multi-talker branch / Phase 9. |
| `09_training.ipynb` | 8 | VALIDATION ONLY | Split manifests; real training additionally needs exact external Icefall backend. | Validation or external checkpoint; GPU likely for training. | No. | Fine-tuned Phase 7 variant. |
| `10_evaluation.ipynb` | 9 | LAST | Saved per-example records and frozen config; it does not create model results. | Final reports/benchmark; CPU. | Yes after records exist. | Final analysis. |

## Recommended order

```text
0 environment ──→ 3 overlap generation ──→ 5 OSD
       │                    │
       ├──→ 2 VAD           └──→ 4 overlap baseline ←── 1 WhisperRT baseline
       │                                      │
       └──────────────────────────────────────→ 6 adaptive routing
                                                       │
                                             external SURT ready?
                                              ├─ no: stop/document
                                              └─ yes → 7 multi-talker → 8 training (external) → 9 evaluation
```

The numerical notebook names are phase labels, not proof of a required sequential execution order. In particular, Phase 2 can run before or after Phase 3; Phase 5 needs only the generated overlap manifest; and Phase 9 consumes records, not notebook state.

## Persistent artifacts and re-run policy

- Persist `outputs/synthetic_overlap/` after a real Phase 3 generation. It is input to later phases and may be expensive to recreate.
- Persist `outputs/predictions/`, `outputs/metrics/`, `outputs/reports/`, and the exact override used after any real ASR run.
- Do not repeatedly re-run Phase 1/4/6 with the same model/dataset just to open a notebook; use existing records for Phase 9.
- Do not run Phase 7 or Phase 8 until their external interfaces are configured. Their validation commands are safe and intentionally fail rather than invent a model.
