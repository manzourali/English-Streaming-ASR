# Architecture

Phase 0 establishes boundaries between audio representation, dataset/model backends, streaming state, pipeline components, metrics, and utilities. `AudioChunk` carries samples, sample rate, timestamps, and an index. `StreamingEngine` enforces ordered incremental processing and lifecycle transitions. `StreamingASRPipeline` composes VAD, overlap detection, and an optional ASR component without requiring model weights.

Dataset and model factories reserve `local`, `kaggle`, and `huggingface` backends. Hugging Face loading deliberately raises a clear deferred-backend error in Phase 0. Paths prevent Kaggle outputs from being directed to `/kaggle/input`.

Phase 1 adds the separate clean-speech path `Audio → Chunker → WhisperRT Streaming Adapter → Transcript`. The adapter follows the upstream causal API: model reset starts an utterance, each chunk updates `SpectrogramStream` and calls `decode`, and finalization marks the latest rolling hypothesis final. VAD/OSD/routing remain out of this path.

Phase 2 adds `Audio → Streaming VAD → WhisperRT Streaming ASR → Transcript`. The selected WebRTC backend buffers fixed frames internally and emits speech/non-speech decisions. In the primary integration design, VAD observes the stream while continuous audio still reaches WhisperRT, preserving model context; VAD is not OSD and produces no speaker/overlap labels.

Phase 3 adds a separate data path: `LibriSpeech sources → deterministic selector → temporal scheduler → amplitude-safe mixer → JSONL manifest + WAV mixture`. Ground-truth overlap is calculated from source timing, not inferred from the mixed waveform. Generated audio can subsequently be passed to the existing streaming/VAD infrastructure.

Phase 4 reuses the Phase 1 path unchanged: `mixture WAV → streaming chunks → WhisperRT → incremental transcript`. VAD is disabled in the primary overlap baseline; VAD-enabled runs must be separately labeled.

Limitations: Phase 0 dummy components are integration scaffolding, and a real Phase 1 run requires the optional upstream WhisperRT stack and external model/data access.

Phase 5 adds a parallel observation path: `Audio → Streaming OSD → overlap state`. OSD consumes the same timestamped chunks independently of WhisperRT and VAD and emits `NO_SPEECH`, `SINGLE_SPEAKER`, or `OVERLAP`. The selected baseline is causal spectral complexity; no OSD output is routed into ASR in Phase 5, so adaptive routing remains a Phase 6 concern.

Phase 6 turns those observations into `AudioChunk → VAD → OSD (or oracle timing) → RoutingPolicy → ASRBranch`. The policy owns the three-state transition logic and configurable threshold/persistence; the controller owns component lifecycle, history, transition events, fallback, and transcript ownership. Its default `forward_normal` idle behavior keeps all live chunks flowing through the shared normal WhisperRT adapter, avoiding decoder resets and retaining causal context across silence and short overlap events. The default overlap route is therefore a logical branch of the same adapter, not a second recognizer; each live chunk has one branch owner and one user-visible output. For a future distinct Phase 7 branch, a bounded history can be replayed solely as context, with replay output suppressed. Oracle timing is an explicitly non-deployable experimental source and is never mixed into predicted OSD results.

Phase 7 supplies that distinct overlap branch as `MultiTalkerASRBranch`. The selected SURT 2.0 family produces unordered output channels rather than speaker identities. `SURT2WindowedASR` retains configurable rolling context and a per-channel rolling hypothesis manager; it emits only the novel suffix separately from the full current hypothesis, preventing repeated partial text from becoming duplicate transcript content. The adapter requires an explicit external decoder factory and is classified as **LOW-LATENCY WINDOWED** until the configured decoder's native SURT state API is verified. No missing decoder is replaced with WhisperRT.
