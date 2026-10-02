# Architecture

Phase 0 establishes boundaries between audio representation, dataset/model backends, streaming state, pipeline components, metrics, and utilities. `AudioChunk` carries samples, sample rate, timestamps, and an index. `StreamingEngine` enforces ordered incremental processing and lifecycle transitions. `StreamingASRPipeline` composes VAD, overlap detection, and an optional ASR component without requiring model weights.

Dataset and model factories reserve `local`, `kaggle`, and `huggingface` backends. Hugging Face loading deliberately raises a clear deferred-backend error in Phase 0. Paths prevent Kaggle outputs from being directed to `/kaggle/input`.

Limitations: dummy components are integration scaffolding, not accuracy claims; no real ASR/VAD/OSD or routing is implemented.

