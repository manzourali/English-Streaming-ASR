# Phase 6 — Adaptive Streaming ASR Routing

## Objective

Implement a streaming controller that consumes VAD and OSD observations per audio chunk, makes stabilized route decisions, and preserves a future overlap-ASR branch boundary. It does not implement a multi-talker recognizer.

## Architecture and routing policy

`AdaptiveStreamingASRPipeline` owns lifecycle, timestamped routing events, bounded context history, branch fallback, and transcript ownership. `RoutingPolicy` maps VAD plus OSD confidence to `NO_SPEECH`, `SINGLE_SPEAKER`, or `OVERLAP`, with configurable overlap threshold, enter/exit frame persistence, and minimum overlap duration. Logical routes are `idle`, `normal`, and `overlap`.

The default `forward_normal` idle policy forwards silence to normal WhisperRT rather than resetting it. The default overlap branch is shared with normal WhisperRT; every live chunk is processed exactly once and emits at most one visible update. Consequently, a route transition preserves the existing WhisperRT decoder/spectrogram state. A distinct future branch can receive bounded replay history on entry; replay updates are context-only and suppressed.

## Experimental conditions

- **Always normal:** controller observes state but always routes to normal.
- **Oracle routing — NOT A DEPLOYABLE SYSTEM:** source timings from Phase 3 provide the routing signal.
- **Predicted OSD routing:** configured streaming VAD plus configured causal OSD provides the routing signal.

The runner records routing confusion, precision/recall/F1, transitions, detection-to-routing delay, state duration, errors, and RTF. It labels the shared-branch control as controller/overhead evaluation rather than recognition improvement.

## Results

No Phase 6 end-to-end result was measured in this checkout. The configured default manifest (`data/manifests/test.jsonl`) and the external WhisperRT runtime/model artifacts are absent. This is intentional: no dataset or model was downloaded and no numbers were fabricated.

## Tests and limitations

Lightweight tests cover oracle `single → overlap → single → silence`, stabilization entry/exit, shared-branch no-duplicate delivery/finalization, and idle skipping. Existing Phase 1–5 unit tests pass. The real integration path is `scripts/run_adaptive.py` with a valid Phase 3 manifest and WhisperRT installation.

The spectral OSD remains a causal heuristic baseline. The Phase 6 shared overlap route is not overlap-aware ASR and cannot support a claim of improved WER. Phase 7 must supply a genuine causal overlap-aware branch with explicit input-context requirements, incremental outputs, finalization behavior, latency/cost reporting, and a transcript-merging policy compatible with this controller.

---

Implemented Phase 6 adaptive streaming routing.

Key additions:

AdaptiveStreamingASRPipeline with VAD + OSD orchestration, route state machine, hysteresis, transition events, error fallback, and bounded context replay.
Generic ASRBranch interface plus shared-WhisperRT Phase 6 control branch.
Always-normal, oracle-routing, and predicted-OSD experiment modes in
run_adaptive.py
.
Routing metrics, trace/event output, optional routing visualizations, config, notebook, tests, documentation, and
MD
Phase 6 report.
Updated adaptive config and adaptive pipeline.

Validation passed:

python3 -m pytest -q → 28 passed
python3 scripts/smoke_test.py --config configs/base.yaml
 → passed
Python compilation, notebook JSON validation, and whitespace/diff checks passed.
No Phase 6 benchmark was run: the configured Phase 3 manifest and external WhisperRT model/runtime are not present locally. This is documented as NOT MEASURED; no results were fabricated.

The default overlap route intentionally shares the normal WhisperRT instance, preserving decoder context and ensuring each live chunk is processed once. It measures routing, transitions, and overhead—not improved overlap ASR. Phase 7 now needs to provide a real causal overlap-aware branch with incremental outputs, explicit context needs, latency/cost reporting, and compatible transcript merging.
