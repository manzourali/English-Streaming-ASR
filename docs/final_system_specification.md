# Final system specification

## Status

This is the final **implemented-system specification**, not a claim that a primary final experiment was executed. The repository has no verified final WhisperRT, SURT, or fine-tuned SURT checkpoint, and no final per-example test record set. Consequently, a deployable “primary proposed system” is **NOT EVALUATED**.

## Implemented candidate architecture

```text
16 kHz mono audio stream
  → fixed timestamped chunks
  → streaming VAD
  → causal heuristic OSD
  → stabilized adaptive routing policy
  ├─ normal/no-speech path: WhisperRT streaming adapter
  └─ overlap path: external SURT 2.0-compatible adapter
  → incremental transcript updates
  → Phase 9 per-example evaluation records
```

The overlap branch is activated only when `branches.overlap: surt2` is supplied with a verified external decoder factory. Otherwise, the Phase 6 shared-WhisperRT control remains the runnable default. Oracle routing is an experiment-only, **NOT DEPLOYABLE** condition.

## Component freeze

| Item | Final artifact status |
| --- | --- |
| ASR model | `MLSpeech/WhisperRT-Streaming` adapter implemented; exact final checkpoint **NOT PROVIDED** |
| ASR checkpoint | **NOT REPRODUCED** |
| VAD | WebRTC VAD backend or deterministic dummy backend, selected by configuration |
| VAD configuration | [`configs/streaming_vad.yaml`](../configs/streaming_vad.yaml) |
| OSD | Causal spectral heuristic OSD |
| OSD checkpoint | Not applicable; no learned OSD checkpoint |
| OSD threshold | Configurable; final value **NOT EVALUATED** |
| Routing policy | Threshold + enter/exit persistence + optional VAD gate |
| Multi-talker model | External SURT 2.0-compatible decoder interface |
| Multi-talker checkpoint | **NOT PROVIDED / NOT REPRODUCED** |
| Fine-tuning strategy | No project-local fine-tuning; external recipe required |
| Chunk duration | 300 ms WhisperRT/default adaptive config; configurable |
| SURT context / lookahead | 1280 ms / 0 ms defaults; configurable low-latency window wrapper |
| Overlap pre-roll | 640 ms when SURT branch is enabled; configurable |
| Sample rate / format | 16 kHz mono floating-point samples internally |
| Device / dtype | Configurable (`auto` by default); **NOT MEASURED** for final system |

Exact configuration, Git state, hashes, and worktree cleanliness are recorded only when `scripts/evaluate.py` or `scripts/audit_artifact.py` is run. A final result must reference those generated manifests.
