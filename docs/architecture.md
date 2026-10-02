# Architecture

Phase 0 establishes boundaries between audio representation, dataset/model backends, streaming state, pipeline components, metrics, and utilities. `AudioChunk` carries samples, sample rate, timestamps, and an index. `StreamingEngine` enforces ordered incremental processing and lifecycle transitions. `StreamingASRPipeline` composes VAD, overlap detection, and an optional ASR component without requiring model weights.

Dataset and model factories reserve `local`, `kaggle`, and `huggingface` backends. Hugging Face loading deliberately raises a clear deferred-backend error in Phase 0. Paths prevent Kaggle outputs from being directed to `/kaggle/input`.

Phase 1 adds the separate clean-speech path `Audio → Chunker → WhisperRT Streaming Adapter → Transcript`. The adapter follows the upstream causal API: model reset starts an utterance, each chunk updates `SpectrogramStream` and calls `decode`, and finalization marks the latest rolling hypothesis final. VAD/OSD/routing remain out of this path.

Limitations: Phase 0 dummy components are integration scaffolding, and a real Phase 1 run requires the optional upstream WhisperRT stack and external model/data access.
