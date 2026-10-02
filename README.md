# English Streaming ASR with Overlapped Speech

Current phase: **Phase 0 — Infrastructure**  
Status: **Infrastructure implementation**

This repository is a reproducible, phase-by-phase foundation for research on English automatic speech recognition over continuous audio streams with overlapped voices. The long-term system will combine streaming VAD, overlap speech detection, WhisperRT streaming ASR, adaptive routing, and later overlap-aware/multi-talker recognition. Phase 0 intentionally does not implement real inference, training, dataset downloads, or benchmark results.

## Architecture

```text
Audio Stream → Streaming Buffer → Streaming VAD → Overlap Detection
                                      ↓                ↓
                                  Normal ASR      Overlap-aware ASR
                                      └──────→ Incremental Transcript → Evaluation
```

In Phase 0, audio chunks, state transitions, configuration, backend interfaces, and deterministic dummy VAD/OSD components are implemented. WhisperRT, real VAD/OSD, adaptive routing, diarization, and multi-talker ASR are future work. Speaker diarization, identification, enrollment, video, microphone arrays, and beamforming are out of scope.

## Repository

- `src/streaming_asr/`: importable library using a `src/` layout.
- `configs/`: YAML configurations; `base.yaml` and `kaggle.yaml` are usable in Phase 0.
- `scripts/`: environment check and smoke test plus future-phase entry points.
- `tests/`: lightweight lifecycle and metric tests.
- `notebooks/`: phase notebooks; `00_environment.ipynb` covers Phase 0.
- `docs/`: architecture, dataset, experiment, and thesis mapping notes.
- `data/`, `checkpoints/`, `outputs/`: runtime locations; generated data and weights are not committed.

## Installation

Python 3.9+ is supported; Python 3.10 is the documented Conda target. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The Phase 0 stack is deliberately lightweight. PyTorch, Transformers, datasets, and WhisperRT dependencies are deferred to later phases.

## Configuration and Kaggle

All runs use YAML. Nested values are available through dotted keys and a second YAML file can be merged with `--override`. Local outputs use `data/`, `outputs/`, and `checkpoints/`. In Kaggle, `/kaggle/input` is treated as read-only and generated artifacts go to `/kaggle/working/`.

Kaggle Internet ON may be used by later phases for Hugging Face access. Internet OFF is supported when models/datasets are attached as Kaggle inputs. Phase 0 needs neither internet nor attached data.

## Validation

```bash
python scripts/check_environment.py
python scripts/smoke_test.py --config configs/base.yaml
python scripts/smoke_test.py --config configs/kaggle.yaml
pytest -q
```

The smoke test creates deterministic synthetic audio, chunks it, runs the streaming engine and dummy components, finalizes state, and writes machine-readable run metadata. It downloads no models or datasets and reports no fabricated metrics.

## Roadmap

Phase 0 — Infrastructure; Phase 1 — WhisperRT streaming baseline; Phase 2 — Streaming VAD; Phase 3 — Synthetic overlap generation; Phase 4 — Overlap benchmark; Phase 5 — OSD; Phase 6 — Adaptive routing; Phase 7 — Multi-talker/overlap-aware ASR; Phase 8 — Training/PEFT; Phase 9 — Evaluation/ablation; Phase 10 — Reproducibility/thesis experiments.

## Research integrity

Placeholders are explicitly labeled. Unmeasured values remain unmeasured, no benchmark result is fabricated, and research decisions are deferred until the relevant phase and evidence exist.

