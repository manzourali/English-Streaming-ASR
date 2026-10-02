# Architecture

Phase 0 establishes boundaries between audio representation, dataset/model backends, streaming state, pipeline components, metrics, and utilities. `AudioChunk` carries samples, sample rate, timestamps, and an index. `StreamingEngine` enforces ordered incremental processing and lifecycle transitions. `StreamingASRPipeline` composes VAD, overlap detection, and an optional ASR component without requiring model weights.

Dataset and model factories reserve `local`, `kaggle`, and `huggingface` backends. Hugging Face loading deliberately raises a clear deferred-backend error in Phase 0. Paths prevent Kaggle outputs from being directed to `/kaggle/input`.

Phase 1 adds the separate clean-speech path `Audio → Chunker → WhisperRT Streaming Adapter → Transcript`. The adapter follows the upstream causal API: model reset starts an utterance, each chunk updates `SpectrogramStream` and calls `decode`, and finalization marks the latest rolling hypothesis final. VAD/OSD/routing remain out of this path.

Phase 2 adds `Audio → Streaming VAD → WhisperRT Streaming ASR → Transcript`. The selected WebRTC backend buffers fixed frames internally and emits speech/non-speech decisions. In the primary integration design, VAD observes the stream while continuous audio still reaches WhisperRT, preserving model context; VAD is not OSD and produces no speaker/overlap labels.

Phase 3 adds a separate data path: `LibriSpeech sources → deterministic selector → temporal scheduler → amplitude-safe mixer → JSONL manifest + WAV mixture`. Ground-truth overlap is calculated from source timing, not inferred from the mixed waveform. Generated audio can subsequently be passed to the existing streaming/VAD infrastructure.

Phase 4 reuses the Phase 1 path unchanged: `mixture WAV → streaming chunks → WhisperRT → incremental transcript`. VAD is disabled in the primary overlap baseline; VAD-enabled runs must be separately labeled.

Limitations: Phase 0 dummy components are integration scaffolding, and a real Phase 1 run requires the optional upstream WhisperRT stack and external model/data access.
