# Phase 7 — Streaming overlap-aware / multi-talker ASR

    Phase 7 should introduce the first real overlap-aware / multi-talker ASR branch, while preserving everything built in Phases 1–6. The key is to avoid prematurely locking the implementation to a particular model before the agent verifies streaming compatibility, output format, and available checkpoints.

## 1. Objective

Add the first actual overlap-branch contract and select a defensible multi-talker ASR family without changing the existing WhisperRT normal branch.

## 2–3. Candidates and selection

The review is in [multitalker_candidates.md](../multitalker_candidates.md). **Selected:** SURT 2.0 through its public release/icefall ecosystem. The research architecture is true streaming; this project's generic adapter is **LOW-LATENCY WINDOWED** until an installed, verified external decoder exposes native streaming state.

## 4–5. Architecture and streaming behavior

`Audio → VAD → OSD → Adaptive Router → WhisperRT | MultiTalkerASRBranch → incremental output`.

`SURT2WindowedASR` maintains an audio context buffer, hop schedule, rolling stream hypotheses, and finalization state. It never resets on each chunk. It accepts a configured external callable decoder and emits structured channels with timestamps, optional confidence, partial/final state, full rolling text, and newly committed text. Channel IDs are not person identities. A Phase 6 history window supplies overlap pre-roll when the branch is activated; replay output is suppressed by the adaptive controller.

## 6–13. Dataset, experiments, metrics, results, latency, routing, failures, resources

**IMPLEMENTED:** `scripts/run_multitalker.py` evaluates a configured decoder against existing Phase 3 manifests using permutation-invariant WER, structured JSONL output, and RTF. The Phase 6 adaptive controller can accept `MultiTalkerASRBranch` as its overlap branch.

**NOT MEASURED:** no SURT checkpoint/runtime, Phase 3 manifest, GPU, or external decoder factory was available in this checkout. Therefore no ASR metric, latency, memory, routing result, oracle-vs-predicted comparison, or qualitative failure example is reported.

**Failure handling:** missing external runtime/factory raises a clear `BackendNotAvailableError`; unsupported sample rate, malformed decoder output, and invalid lifecycle calls fail explicitly. There is no Whisper fallback.

## 14. Tests

Unit tests use a callable mock decoder to verify structured multiple-stream output, rolling-hypothesis duplicate suppression, hop buffering, malformed-output errors, finalization, adaptive-branch wrapping, and permutation-invariant WER. They download no model.

## 15. Limitations

The adapter is not proof that the selected SURT checkpoint has been reproduced. Its public recipe/runtime must be installed through a versioned external decoder factory, and that configuration plus exact hardware/software data must be recorded before reporting results. The low-latency window wrapper must not be called true streaming unless the decoder's native causal state is verified.

## 16. Phase 8 requirements

Verify one exact SURT 2.0 recipe/checkpoint and its license, add a version-pinned decoder factory, execute the Phase 3 matrix, measure cp/pi-WER, latency/RTF/memory, and compare always-WhisperRT, always-SURT, adaptive predicted OSD, and oracle routing. Only then consider justified training/fine-tuning work.

---

Implemented Phase 7 scaffolding and integration.
- Selected SURT 2.0 as the primary multi-talker model family; its published architecture targets continuous streaming multi-talker ASR, with public release/recipe material. The project adapter is honestly labelled LOW-LATENCY WINDOWED, pending verification of a concrete external decoder’s native state API.
- Added structured SURT2WindowedASR output: unordered stream IDs, rolling text, non-duplicated new text, timestamps, confidence, partial/final flags.
Added MultiTalkerASRBranch and adaptive-router activation via branches.overlap: surt2.
Added permutation-invariant WER for unordered output streams.
- Added scripts/run_multitalker.py, configs/multitalker.yaml, optional real-model integration test, Kaggle notebook, candidate review, Phase 7 report, and documentation updates.
    Important limitation: no exact SURT checkpoint/runtime, Phase 3 manifest, or GPU was available locally. Therefore ASR metrics, PI-WER, latency/RTF, memory, adaptive routing results, and oracle-vs-predicted comparisons are NOT MEASURED. The code raises a clear error if no external decoder_factory is configured; it never falls back to WhisperRT.
