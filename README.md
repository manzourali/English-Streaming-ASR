# English Streaming ASR with Overlapped Speech

Current phase: **Phase 9 — Evaluation, Ablation, and Robustness Protocol**

Status: **Phases 0–9 implemented; final measured systems and datasets remain to be executed**

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

## Current status by phase

| Phase | What is implemented | What is actually measured |
| --- | --- | --- |
| 0 | Project structure, configs, streaming contracts, smoke tests | Infrastructure smoke tests pass |
| 1 | WhisperRT causal streaming adapter and clean-speech runner | Real WER/RTF/latency: **NOT MEASURED**; required model stack was unavailable |
| 2 | Streaming WebRTC VAD with fixed-frame buffering | Synthetic integration timing only; VAD accuracy: **NOT MEASURED** |
| 3 | Deterministic two-speaker overlap generator, manifests, validation | 12 local demo mixtures generated and validated; real LibriSpeech generation not run |
| 4 | Single-stream WhisperRT overlap baseline and honest WER policy | Manifest validation passed; real WhisperRT overlap results: **NOT MEASURED** |
| 5 | Independent causal heuristic OSD, timing alignment, metrics, evaluator | Demo: precision 1.000, recall 0.636, F1 0.778, 15 ms mean delay, 0.00274 RTF |
| 6 | Incremental VAD/OSD controller, configurable routing state machine, branch contract, oracle/predicted controls | **NOT MEASURED**: Phase 3 manifest plus WhisperRT runtime/model are required |
| 7 | SURT 2.0-compatible structured overlap branch, low-latency window adapter, PI-WER metric, adaptive integration | **NOT MEASURED**: verified external SURT decoder/checkpoint, Phase 3 manifest, and compute are required |
| 8 | Training-manifest/HEAT target preparation, leakage checks, collator, checkpoint lifecycle, external-training contract | **TRAINING NOT TECHNICALLY SUPPORTED** locally: no pinned Icefall recipe/trainable checkpoint/tokenizer/backend factory |
| 9 | Frozen evaluation protocol, per-example result schema, aggregate/stratified analysis, uncertainty, benchmark/report generation | **NOT MEASURED**: no verified final checkpoints, test manifest, or execution hardware available |

The main research gap is therefore the real WhisperRT/LibriSpeech execution. Phases 1 and 4 are implemented, but their model/data benchmarks still need to run in an environment with PyTorch, the verified WhisperRT package, model weights, and dataset access. Phase 5 is a transparent spectral baseline, not a pretrained neural OSD result. See [`docs/phase_reports/`](docs/phase_reports/) for the evidence and limitations of every phase.

## Installation

Python 3.9+ is supported; Python 3.10 is the documented Conda target. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The base installation is deliberately lightweight. PyTorch, Hugging Face `datasets`, and the official WhisperRT dependency stack are optional because the repository can run its infrastructure, VAD, synthetic-data, and heuristic OSD checks without them.

## Phase 8 — Training feasibility

The selected SURT 2.0 family has official external Icefall recipes, but the Phase 7 project adapter exposes inference only. Phase 8 therefore implements a strict feasibility bridge: Phase 3 manifest-to-HEAT-channel targets, split-leakage validation, variable-length collation, checkpoint/resume lifecycle, finite-loss detection, and a version-pinned `ExternalSurtTrainingBackend` contract. It does **not** invent a Transformers/PEFT training procedure.

```bash
python3 scripts/smoke_test.py --phase 8 --config configs/training.yaml
python3 scripts/train_asr.py --config configs/training.yaml --validate-only
```

The smoke path is model-free. A real training command additionally requires a valid Phase 3 train/validation/test manifest and an override whose `model.backend_factory` wraps one exact, verified Icefall/SURT recipe. Until then, the correct status is **TRAINING NOT TECHNICALLY SUPPORTED**, not a reported fine-tuning result. See [model compatibility](docs/training_model_compatibility.md), [target objective](docs/training_objective.md), and [the Phase 8 report](docs/phase_reports/phase_8_report.md).

## Phase 9 — Evaluation protocol

Phase 9 freezes the experimental protocol without changing the system architecture. [`configs/evaluation.yaml`](configs/evaluation.yaml) records the intended checkpoints/configurations/manifests, primary and oracle systems, test-set protection, and six-row ablation plan. [`scripts/evaluate.py`](scripts/evaluate.py) ingests real per-example JSONL records and generates aggregate results, overlap/temporal stratifications, deterministic bootstrap intervals, error taxonomy, computational tables, and a Markdown report. It refuses invalid paired improvement claims when sample sets or metric directions differ.

```bash
python3 scripts/evaluate.py --config configs/evaluation.yaml --validate-only
python3 scripts/smoke_test.py --phase 9 --config configs/evaluation.yaml
# After real runs:
python3 scripts/evaluate.py --config configs/evaluation.yaml --records outputs/predictions/final_records.jsonl
python3 scripts/benchmark.py --records outputs/predictions/final_records.jsonl
```

The current configuration is a protocol freeze, not a measured final-system freeze: model checkpoints remain unset and all Phase 9 scientific outcomes are **NOT MEASURED**. See [the Phase 9 report](docs/phase_reports/phase_9_report.md).

## Configuration and Kaggle

All runs use YAML. Nested values are available through dotted keys and a second YAML file can be merged with `--override`. Local outputs use `data/`, `outputs/`, and `checkpoints/`. In Kaggle, `/kaggle/input` is treated as read-only and generated artifacts go to `/kaggle/working/`.

Kaggle Internet ON may be used by later phases for Hugging Face access. Internet OFF is supported when prepared data/model artifacts are attached as Kaggle inputs. The generic [`configs/kaggle.yaml`](configs/kaggle.yaml) is intended for infrastructure/path checks; for phase experiments use [`configs/kaggle_paths.yaml`](configs/kaggle_paths.yaml), which preserves Hugging Face loading while moving generated artifacts and manifests to `/kaggle/working/`.

### Kaggle setup

1. Create a Kaggle Notebook with Internet enabled if you need Hugging Face or WhisperRT downloads, select a GPU accelerator for WhisperRT, and add this repository as a Kaggle Dataset or clone it into `/kaggle/working/English-Streaming-ASR`.
2. In the first cell, install the base and optional dependencies:

   ```bash
   %cd /kaggle/working/English-Streaming-ASR
   !pip install -q -r requirements.txt webrtcvad-wheels datasets huggingface_hub
   !python scripts/check_environment.py
   ```

3. Run the model-free regression checks first:

   ```bash
   !python -m pytest -q
   !python scripts/smoke_test.py --config configs/kaggle.yaml
   ```

4. Run Phase 2 VAD on the synthetic demo:

   ```bash
   !python scripts/run_vad.py --config configs/streaming_vad.yaml --override configs/kaggle_paths.yaml
   !python scripts/smoke_test.py --phase 2 --real-vad --config configs/streaming_vad.yaml
   ```

5. Generate the deterministic Phase 3 development set and evaluate Phase 5 OSD:

   ```bash
   !python scripts/generate_overlap.py --config configs/overlap_dataset.yaml --override configs/kaggle_paths.yaml --demo
   !python scripts/evaluate_osd.py --config configs/overlap_detection.yaml --override configs/kaggle_paths.yaml
   !python scripts/smoke_test.py --phase 5 --validate-osd --config configs/overlap_detection.yaml --override configs/kaggle_paths.yaml
   ```

   The generated WAV files, manifests, metrics, and reports will be under `/kaggle/working/outputs/` and can be saved as Kaggle Notebook outputs or published as a new Dataset version.

6. Run Phase 1 clean-speech WhisperRT with a small sample count before attempting a full benchmark. The exact upstream WhisperRT package and model-loading dependencies must be installed according to the verified WhisperRT project used by this adapter:

   ```bash
   !python scripts/run_streaming.py --config configs/whisperrt_baseline.yaml --override configs/kaggle_paths.yaml --max-samples 1
   ```

   If the upstream package/model is not installed or cannot be downloaded, the command should stop with a dependency error; do not substitute offline Whisper and do not report fabricated WER.

7. After Phase 3 data and a successful Phase 1 model smoke run, validate Phase 4 without inference, then run the real baseline:

   ```bash
   !python scripts/run_overlap_baseline.py --config configs/overlap_baseline.yaml --override configs/kaggle_paths.yaml --validate-only
   !python scripts/run_overlap_baseline.py --config configs/overlap_baseline.yaml --override configs/kaggle_paths.yaml --max-samples 1
   ```

   For a full experiment, remove `--max-samples` only after the one-sample run succeeds. Phase 4 intentionally marks ordinary WER for overlapping mixtures as `NOT_MEASURED` because one transcript stream has no justified speaker assignment.

8. Run Phase 6 routing controls only after the synthetic manifest and WhisperRT model configuration are available:

   ```bash
   !python scripts/run_adaptive.py --config configs/adaptive.yaml --override configs/kaggle_paths.yaml --validate-only
   !python scripts/run_adaptive.py --config configs/adaptive.yaml --override configs/kaggle_paths.yaml --max-samples 1
   ```

   The runner writes separate always-normal, **ORACLE ROUTING — NOT A DEPLOYABLE SYSTEM**, and predicted-OSD artifacts. The default Phase 6 overlap route deliberately shares WhisperRT with the normal route, so it measures routing correctness, transition behavior, and overhead—not overlap-recognition improvement.

#### Kaggle data modes

- **Internet ON / Hugging Face mode:** use the commands above. Keep the phase config's `data.backend: huggingface`; `kaggle_paths.yaml` only changes paths.
- **Attached-input mode:** attach prepared WAV/manifests/model files under `/kaggle/input` and create a project-specific override with the exact input paths. The current `KaggleDatasetBackend` expects a prepared local manifest layout; it does not automatically discover arbitrary Kaggle Dataset folder names.
- **Model artifacts:** model files and the official WhisperRT package are not bundled in this repository. Attach or download them according to the upstream project, verify the model filename/configuration, and record the resolved config and hardware in the generated report.

All generated outputs should remain in `/kaggle/working`; `/kaggle/input` is read-only. Before claiming a benchmark, save the JSON metrics, JSONL predictions/manifests, resolved configuration, logs, and Markdown report from `/kaggle/working/outputs/`.

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

## Phase 1 — WhisperRT Streaming Baseline

Phase 1 adds the verified `whisper_rt` causal streaming adapter for `MLSpeech/WhisperRT-Streaming`. The adapter uses the upstream `StreamingWhisper.reset`, `SpectrogramStream`, and repeated `decode` calls, preserving model state across chunks. It does not use offline Whisper and has no fallback model. The clean-speech benchmark is configured for LibriSpeech `test-clean` through the Hugging Face `openslr/librispeech_asr` dataset (`clean`/`test` internally), with a configurable 300 ms chunk and batch size one.

Install the optional Phase 1 stack according to the upstream WhisperRT project (PyTorch, `huggingface_hub`, the official `WhisperRT-Streaming` repository/package, and `datasets`), then run:

```bash
python3 scripts/run_streaming.py --config configs/whisperrt_baseline.yaml --max-samples 1
```

The run saves JSONL predictions, JSON metrics, a Markdown report, and the resolved configuration under `outputs/`. It requires model/data access and is intentionally separate from the lightweight Phase 0 smoke test. Phase 1 measures clean single-speaker speech only; it does not establish overlap robustness and does not implement VAD, OSD, routing, or multi-talker recognition.

## Phase 2 — Streaming VAD

Phase 2 adds a stateful WebRTC VAD backend. WebRTC VAD consumes 16 kHz mono PCM in fixed 10/20/30 ms frames; the project buffers those frames internally while accepting configurable ASR chunks. Its binary decision is exposed as speech/non-speech with a 0/1 decision score, not a calibrated probability. Install `webrtcvad-wheels` and run the explicit demo with:

```bash
python3 scripts/run_vad.py --config configs/streaming_vad.yaml
python3 scripts/smoke_test.py --phase 2 --real-vad --config configs/streaming_vad.yaml
```

The selected integration design observes audio with VAD while forwarding the continuous audio stream to WhisperRT, preserving ASR context. This makes VAD overhead and decisions measurable without silently changing the Phase 1 input. Phase 2 does not perform overlap detection.

## Phase 3 — Synthetic Overlap Dataset Generation

Phase 3 generates deterministic two-speaker mixtures from clean source utterances. Each JSONL record stores source IDs, speaker IDs, transcripts, source timing, overlap interval, overlap duration/ratio, relative gain, split, and mixture audio path. The overlap ratio is `overlap_duration / shorter_source_duration`; source RMS normalization is applied before relative gain, followed by common peak headroom scaling.

Run the small no-download development dataset with:

```bash
python3 scripts/generate_overlap.py --config configs/overlap_dataset.yaml --demo
```

Configured LibriSpeech/Hugging Face or local/Kaggle generation uses the same dataset backend and must be run explicitly. Outputs are written under `outputs/synthetic_overlap/` and ignored by Git. Synthetic overlap provides controlled ground truth and is not a substitute for real conversational overlap datasets.

## Phase 4 — WhisperRT Under Synthetic Overlap Baseline

Phase 4 evaluates the existing single-stream WhisperRT pipeline directly on Phase 3 mixtures. It does not add OSD, separation, multi-talker decoding, overlap prompts, or optimization. For non-overlap controls, WER uses time-ordered concatenated source transcripts. For overlapping mixtures, source references and metadata are retained but ordinary WER is `NOT_MEASURED` because a single transcript stream has no justified speaker assignment.

```bash
python3 scripts/run_overlap_baseline.py --config configs/overlap_baseline.yaml --validate-only
```

The real benchmark requires the Phase 1 WhisperRT dependencies and model. Phase 4 artifacts are stored under `outputs/phase4/`.

## Research integrity

Placeholders are explicitly labeled. Unmeasured values remain unmeasured, no benchmark result is fabricated, and research decisions are deferred until the relevant phase and evidence exist.

## Phase 5 — Streaming Overlap Speech Detection

Phase 5 adds an independent streaming OSD interface with `NO_SPEECH`, `SINGLE_SPEAKER`, and `OVERLAP` states. The selected candidate is a causal, dependency-free spectral-peak heuristic (`heuristic` backend); it is an algorithmic baseline, not a pretrained neural model. It runs incrementally on fixed audio frames and does not modify WhisperRT, VAD, routing, diarization, or source separation.

Run the reproducible development evaluation after generating the Phase 3 demo data:

```bash
python3 scripts/generate_overlap.py --config configs/overlap_dataset.yaml --demo
python3 scripts/evaluate_osd.py --config configs/overlap_detection.yaml
python3 scripts/smoke_test.py --phase 5 --validate-osd --config configs/overlap_detection.yaml
```

Metrics are scored against source-timing ground truth, with explicit midpoint alignment for frame-size differences. Reports include overlap precision/recall/F1, a three-class confusion matrix, event detection delay when measurable, and OSD RTF. Synthetic mixtures are controlled development data and do not establish real conversational-overlap performance.

## Phase 6 — Adaptive Streaming ASR Routing

`AdaptiveStreamingASRPipeline` orchestrates timestamped chunks as `VAD → OSD → RoutingPolicy → ASRBranch`. Its explicit state machine is `NO_SPEECH`, `SINGLE_SPEAKER`, and `OVERLAP`; the logical routes are `idle`, `normal`, and `overlap`. `RoutingPolicy` exposes the OSD threshold, enter/exit persistence frames, optional minimum overlap duration, VAD gating, idle behavior, history window, and deterministic normal-route fallback in [`configs/adaptive.yaml`](configs/adaptive.yaml).

The mandatory controls are `always_normal`, `oracle`, and `predicted`. Oracle routing uses Phase 3 source timing and is marked non-deployable. Predicted routing uses the configured streaming OSD. The default overlap branch is the *same* WhisperRT instance as the normal branch: every live chunk is processed once, logical route changes do not reset WhisperRT, and rolling hypotheses remain owned by the single shared stream. This is a deliberate Phase 6 control, not a multi-talker recognizer. A future separate branch may receive configurable preceding context; replay output is suppressed to avoid duplicate user-visible text.

Run after generating/pointing the config at a valid Phase 3 manifest and configuring model weights:

```bash
python3 scripts/run_adaptive.py --config configs/adaptive.yaml --validate-only
python3 scripts/run_adaptive.py --config configs/adaptive.yaml --max-samples 1
```

The runner saves routing JSONL traces, transition events, per-mode metrics/reports, route duration, routing precision/recall/F1, detection-to-routing delay, and end-to-end RTF. No Phase 6 benchmark values are claimed in this repository because the required manifest and WhisperRT runtime/model were not present during implementation.

## Phase 7 — Overlap-aware / multi-talker ASR

Phase 7 selects the SURT 2.0 model family after documenting alternatives in [the candidate review](docs/multitalker_candidates.md). SURT 2.0 is designed for continuous multi-talker ASR, but this repository deliberately does not guess or vendor its external runtime API. [`SURT2WindowedASR`](src/streaming_asr/models/multitalker.py) accepts a versioned external `decoder_factory` and emits structured, unordered recognition channels—not speaker identities—with timestamps, optional confidence, rolling text, newly emitted text, and partial/final state.

The published SURT family is a true-streaming architecture. Until a selected external decoder exposes and verifies native causal state through this adapter, the project execution mode is explicitly **LOW-LATENCY WINDOWED INFERENCE**, not true streaming. It never falls back to ordinary WhisperRT when SURT is unavailable.

Configure a Phase 3 manifest plus a verified factory/checkpoint in an override, then run:

```bash
python3 scripts/run_multitalker.py --config configs/multitalker.yaml --validate-only
python3 scripts/run_multitalker.py --config configs/multitalker.yaml --max-samples 1
```

Set `branches.overlap: surt2` in an adaptive override to route Phase 6 overlap events to the structured SURT branch; `overlap_pre_roll_ms` is represented by the existing adaptive history setting. Multi-talker output is evaluated with permutation-invariant WER across output channels, never with speaker-attributed WER. No Phase 7 ASR, latency, memory, or routing result has been measured in this checkout.
