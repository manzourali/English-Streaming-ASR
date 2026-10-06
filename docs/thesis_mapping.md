# Thesis mapping

Phase 0 maps to environment and software infrastructure. Phase 1 establishes a clean, single-speaker streaming ASR baseline against which later overlap-aware methods can be evaluated. Phases 2–5 add VAD, overlap data, overlap baselines, and OSD; Phases 6–8 investigate routing and overlap-aware recognition/training; Phases 9–10 provide evaluation, ablations, and reproducibility packaging. Scientific contribution claims remain pending experimental evidence.
Phase 2 establishes a streaming speech-activity layer that enables later temporal routing and overlap detection; it does not solve overlapping speech.

Phase 6 maps to the adaptive-control contribution: a stateful VAD/OSD router with oracle and predicted experimental conditions, transition tracing, and real-time overhead accounting. It establishes the interface and evidence path for Phase 7, but does not implement speaker identities, separation, serialized output, or a multi-talker recognizer.

Phase 7 maps to overlap-aware recognition: an external SURT 2.0-compatible branch with structured unordered output channels, window/state management, permutation-invariant evaluation, and adaptive-branch integration. It does not perform diarization or identify people; concrete checkpoint reproduction and any training remain later evidence-gathering work.

Phase 8 maps to adaptation feasibility: a reproducible bridge from Phase 3 manifests to SURT-style channel targets, batch collation, leakage control, checkpoint metadata, and strict external recipe integration. It does not claim local model adaptation, PEFT support, or a performance improvement.
