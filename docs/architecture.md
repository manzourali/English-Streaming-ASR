# Active architecture

```text
16 kHz mono PCM → sample-indexed chunks → 80-bin log-Mel
  → dual-path channel masks → shared streaming Zipformer
  → optional channel-tying layer → stateless predictor + RNN-T joiner
  → committed text deltas and endpoint events
```

`SURT2StreamingASR` schedules 32-frame advances and retains bounded acoustic
history. `TorchSURTBackend` composes supplied trained native modules. The
actual official-checkpoint connection/native exporter and feature parity are
still missing; this is not a bundled trained recognizer. See the
[stage2 report](stages/stage2_report.md) for contracts and latency accounting.

`StreamingASRPipeline` forwards every chunk to one recognizer. VAD/OSD can be
observers in independent experiments; they do not route, drop or replay audio.
WhisperRT is an optional separate comparison.

The data path is local LibriSpeech → speaker-disjoint source manifests →
controlled two-source mixtures → portable audio/manifest bundle. Native
inference writes EvaluationRecord JSONL directly for the evaluator. Actual
training needs an ExternalSurtTrainingBackend; training checkpoints require a
validated native export before inference. See the
[dependency map](phase_dependency_map.md) and [Persian workflow](KAGGLE_WORKFLOW_FA.md).
