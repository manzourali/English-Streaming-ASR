# Current execution matrix

| Entry point | Real requirement | Output | Status |
| --- | --- | --- | --- |
| `download_librispeech.py` | Internet and disk, or use attached data instead | Verified OpenSLR speech files | Implemented; network execution not tested locally |
| `prepare_dataset.py` | Extracted local LibriSpeech transcripts/audio | Speaker-disjoint source manifests | Fixture-tested |
| `generate_overlap.py` | Real source manifests or explicit tone demo | Portable two-source mixtures, labels, provenance | Tested |
| `check_surt_assets.py` | Optional native export directory | Missing/incompatible/present/smoke status | Structure checks tested; real GPU probe pending |
| `run_multitalker.py` / `run_streaming.py` | Trained native exports and matching features/tokenizer | Transcripts, raw metrics, EvaluationRecord JSONL | Scheduler/schema tested; real model execution pending |
| `train_asr.py --validate-only` | Split manifests | HEAT target/leakage validation | Tested |
| `train_asr.py` | Actual ExternalSurtTrainingBackend | Trained checkpoints and trainer metadata | Integration missing |
| Native exporter | Pinned recipe, checkpoint and exact streaming state implementation | Streaming export bundle | Not implemented |
| `evaluate.py` / `benchmark.py` | Actual EvaluationRecord JSONL | Aggregation and compute report | Handoff tested |
| `run_baseline.py` | Optional compatible WhisperRT runtime, weights and clean audio | Independent baseline result | Actual model execution pending |

Full training needs a scalable recipe/DataLoader; the local trainer's eager
batch materialization and epoch-level resume are not a full-scale training
solution. No fixed Kaggle GPU/quota or k2 wheel combination is assumed.
