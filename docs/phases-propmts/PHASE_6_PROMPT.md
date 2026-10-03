# PHASE 6 — Adaptive Streaming ASR Pipeline

You are implementing **Phase 6** of the English Streaming ASR with Overlapped Voices research project.

The project operates under the previously defined **Master Prompt / project contract**. Follow all architecture, reproducibility, testing, Kaggle/Hugging Face, documentation, and research-integrity requirements from that contract.

Do not implement Phase 7 functionality prematurely.

---

# 1. Current Project State

The project has progressed through:

### Phase 0

Project infrastructure and repository skeleton.

### Phase 1

`MLSpeech/WhisperRT-Streaming` streaming ASR baseline.

### Phase 2

Stateful streaming VAD.

### Phase 3

Synthetic two-speaker overlap dataset generation with source timing and ground truth.

### Phase 4

WhisperRT evaluation under synthetic overlap.

### Phase 5

Streaming Overlap Speech Detection (OSD), including:

* OSD abstraction
* streaming inference
* overlap ground truth
* OSD evaluation
* detection latency
* computational cost

Phase 6 now combines these components into an:

> **Adaptive Streaming ASR Pipeline**

The goal is to determine whether detecting overlap and changing the processing path can improve the behavior of the streaming ASR system.

---

# 2. Main Objective

Build a streaming pipeline that dynamically responds to the detected speech condition.

The conceptual architecture is:

```text id="2x8j6u"
Audio Stream
      ↓
Streaming Audio Buffer
      ↓
Streaming VAD
      ↓
Streaming OSD
      ↓
┌──────────────┴──────────────┐
│                             │
Single-speaker             Overlap
│                             │
WhisperRT                 Overlap-aware /
Streaming                 multi-talker path
│                             │
└──────────────┬──────────────┘
               ↓
       Incremental Output
               ↓
           Evaluation
```

However, the exact overlap branch must remain configurable.

**Do not assume that Phase 7's final multi-talker recognizer already exists.**

---

# 3. Critical Phase 6 Principle

Phase 6 is about **adaptive routing**, not yet about building the final overlap-aware ASR model.

The key research question is:

> Can a streaming VAD + OSD controller selectively route audio based on overlap state while preserving real-time behavior and improving or maintaining recognition performance?

Therefore, the pipeline must support an overlap branch even if the branch initially uses a baseline/reference implementation.

Possible Phase 6 progression:

```text id="e4t5pc"
Phase 6A
OSD → routing controller → baseline branches

Phase 6B
OSD → normal WhisperRT for single speech
     → reference overlap handler for overlap

Phase 7
Replace/improve overlap branch with actual multi-talker /
overlap-aware ASR
```

Do not build Phase 7 inside Phase 6.

---

# 4. Scope

Implement:

* adaptive streaming controller
* VAD + OSD integration
* state machine
* routing decisions
* branch management
* transition handling
* confidence/threshold handling
* overlap-state persistence
* latency accounting
* routing metrics
* baseline adaptive experiments
* oracle-routing experiments
* predicted-OSD routing experiments
* tests
* configuration
* Kaggle notebook
* documentation
* phase report

Do not implement:

* diarization
* speaker identification
* speaker enrollment
* source separation
* new multi-talker ASR architecture
* large-scale ASR training
* PEFT
* RL
* final overlap-aware recognizer

---

# 5. Preserve Existing Components

Do not rewrite the previous components unless an actual interface incompatibility requires it.

Reuse:

```text id="n1x2d4"
Phase 1 → WhisperRT
Phase 2 → Streaming VAD
Phase 3 → synthetic overlap data + metadata
Phase 4 → ASR evaluation
Phase 5 → Streaming OSD
```

The adaptive controller should orchestrate them.

It should not duplicate their logic.

---

# 6. Adaptive Pipeline Interface

Create an abstraction similar to:

```python id="f0r6n1"
class AdaptiveStreamingASRPipeline:
    def process(self, audio_chunk):
        ...

    def finalize(self):
        ...
```

The output should contain enough information to understand:

* VAD state
* OSD state
* selected route
* route confidence
* timestamps
* ASR output
* transition events

For example:

```python id="j95x8b"
{
    "timestamp": ...,
    "vad_state": ...,
    "osd_state": ...,
    "route": "normal",
    "confidence": ...,
    "transcript_update": ...
}
```

The exact schema can differ.

Keep the interface implementation-independent.

---

# 7. Streaming-First Requirement

The entire adaptive controller must operate incrementally.

Do not:

```text id="v7t1z8"
record complete audio
→ classify overlap
→ choose one model
→ run offline ASR
```

Instead:

```text id="l8g1y2"
chunk
 ↓
VAD
 ↓
OSD
 ↓
routing decision
 ↓
appropriate processing
 ↓
incremental output
```

The controller must maintain state across chunks.

---

# 8. State Machine

Implement an explicit routing state machine.

At minimum support:

```text id="6n9x5c"
NO_SPEECH
SINGLE_SPEAKER
OVERLAP
```

Potential transitions:

```text id="q2p8ys"
NO_SPEECH
   ↓
SINGLE_SPEAKER
   ↓
OVERLAP
   ↓
SINGLE_SPEAKER
   ↓
NO_SPEECH
```

The state machine should not switch routes on every noisy frame.

Design transition behavior explicitly.

---

# 9. Hysteresis

OSD predictions may fluctuate around a decision threshold.

Do not route directly from every individual prediction unless that behavior is explicitly being tested.

Support configurable temporal stabilization.

Possible controls:

```yaml id="k5f3t9"
routing:
  overlap_threshold: ...
  enter_overlap_frames: ...
  exit_overlap_frames: ...
  min_overlap_duration_ms: ...
```

The exact mechanism is a research/design decision.

Do not hard-code arbitrary values.

Compare at least:

1. raw routing
2. stabilized routing

where practical.

---

# 10. Detection Confidence

If the OSD produces probabilities, preserve them.

The controller should be able to make decisions based on:

```text id="o8r2z4"
overlap_probability
```

rather than requiring a hard class.

The configuration should expose thresholds.

Do not assume `0.5` is automatically correct.

If thresholds are tuned, use validation/development data rather than the final test set.

---

# 11. Routing Policy

Create an explicit routing policy abstraction.

For example:

```python id="4m1g8j"
class RoutingPolicy:
    def decide(self, vad_result, osd_result, state):
        ...
```

This keeps the adaptive controller separate from the policy itself.

Possible initial policy:

```text id="q4m1v7"
NO_SPEECH
    → no ASR processing / appropriate idle behavior

SINGLE_SPEAKER
    → WhisperRT

OVERLAP
    → overlap branch
```

The exact treatment of `NO_SPEECH` must respect WhisperRT's streaming state semantics.

Do not reset WhisperRT state merely because VAD reports silence unless the existing architecture explicitly supports and justifies that behavior.

---

# 12. Critical WhisperRT State Issue

Switching between branches can affect ASR context.

You must explicitly investigate:

* whether WhisperRT maintains decoder state
* whether state can be paused
* whether state can be resumed
* whether resetting state causes transcript degradation
* whether context should be preserved across short overlap events
* how partial hypotheses should be handled during a route transition

Do not assume that a model can be stopped and restarted without consequences.

Document the actual behavior.

---

# 13. Branch Interface

The adaptive controller should not directly depend on WhisperRT internals.

Create a generic branch interface such as:

```python id="xq0m4k"
class ASRBranch:
    def process(self, audio_chunk):
        ...

    def finalize(self):
        ...
```

Then:

```text id="4q2v5f"
NormalASRBranch
OverlapASRBranch
```

can be implemented independently.

---

# 14. Phase 6 Overlap Branch

At this point, the final multi-talker ASR system from Phase 7 may not exist.

Therefore create a **baseline overlap branch** that allows routing to be evaluated without pretending that the overlap problem has already been solved.

Possible implementations include:

### Option A — Same WhisperRT

```text
single → WhisperRT
overlap → WhisperRT
```

This provides an important control showing routing overhead without changing recognition.

### Option B — Oracle/reference branch

If a suitable overlap-processing reference exists, expose it as an optional branch.

### Option C — Placeholder adapter

Create the interface for the future multi-talker branch without implementing the model.

The default implementation must remain executable.

Do not invent an overlap-aware model.

---

# 15. Oracle OSD Experiment

Implement an optional **oracle routing mode**.

In oracle mode, routing decisions are derived from Phase 3 ground-truth overlap metadata rather than the predicted OSD.

This answers:

> What could adaptive routing achieve if overlap detection were perfect?

Conceptually:

```text id="tq3y9c"
Ground-truth overlap
       ↓
routing controller
       ↓
branch selection
```

This is an experimental upper/reference condition.

It must be explicitly labeled:

```text id="jv9y3a"
ORACLE ROUTING
NOT A DEPLOYABLE SYSTEM
```

Do not report it as normal model performance.

---

# 16. Predicted OSD Experiment

Then evaluate:

```text id="z1c4m7"
Streaming VAD
      ↓
Streaming OSD
      ↓
Routing
      ↓
ASR branch
```

This is the actual adaptive pipeline.

Compare it against the oracle condition.

This comparison is important because OSD errors can directly affect routing.

---

# 17. Mandatory Baselines

At minimum compare:

### Baseline 1 — Always Normal

```text id="9i3b5v"
Audio → WhisperRT
```

### Baseline 2 — Adaptive + Oracle OSD

```text id="2y8j6x"
Audio
 ↓
Ground-truth overlap
 ↓
Router
 ↓
branches
```

### Baseline 3 — Adaptive + Predicted OSD

```text id="7p6w2m"
Audio
 ↓
VAD
 ↓
OSD
 ↓
Router
 ↓
branches
```

If the overlap branch is initially identical to WhisperRT, explicitly state that the experiment measures routing/controller behavior rather than recognition improvement.

---

# 18. Optional VAD Ablation

Where practical compare:

```text id="5kq8v3"
OSD only
```

vs

```text id="r9c2a6"
VAD + OSD
```

The purpose is to determine whether VAD contributes useful gating/stability.

Do not assume it does.

---

# 19. Route Transition Handling

When the state changes:

```text id="f5z0r7"
SINGLE_SPEAKER → OVERLAP
```

the controller must define what happens to the current audio chunk.

Do not silently discard audio at transition boundaries.

The implementation should account for:

* buffered audio
* model context
* overlap onset
* overlap offset
* partial hypotheses
* duplicated audio
* missing audio

This is a critical streaming issue.

---

# 20. Boundary Buffering

OSD may detect overlap only after some audio has already arrived.

Therefore consider a small configurable history buffer:

```text id="3x7n2a"
audio history
     ↓
OSD detects overlap
     ↓
overlap branch receives required context
```

The implementation must quantify any lookback or detection delay.

Do not claim zero-latency routing.

If historical audio is replayed into a branch, prevent duplicate transcript output.

---

# 21. Duplicate Output Prevention

Route transitions can cause the same audio to be processed by more than one branch.

The pipeline must explicitly manage:

* audio ownership
* branch input ranges
* transcript ownership
* duplicate partial hypotheses

Do not simply concatenate outputs from branches.

Create a clear policy for how incremental outputs are merged.

---

# 22. Transcript Management

The adaptive pipeline should expose:

### Internal branch output

What each ASR branch produces.

### Final pipeline output

What the user would actually receive.

Keep these separate.

This allows future Phase 7 work to improve the overlap branch without redesigning transcript management.

---

# 23. Routing Logs

Record every significant routing event.

For example:

```json id="u0r6l8"
{
  "timestamp": 12.40,
  "previous_state": "SINGLE_SPEAKER",
  "new_state": "OVERLAP",
  "osd_probability": 0.87,
  "route": "overlap",
  "reason": "threshold_crossed"
}
```

This will be essential for debugging adaptive behavior.

---

# 24. Routing Metrics

Phase 6 introduces new metrics.

At minimum measure:

### Routing correctness

Compare predicted route with ground truth.

### Transition metrics

* number of transitions
* false transitions
* missed overlap transitions
* unnecessary overlap transitions

### Detection-to-routing latency

Measure:

```text id="b4w9q2"
ground-truth overlap start
        ↓
OSD detection
        ↓
routing transition
```

### Route duration

How much time was spent in:

* normal route
* overlap route
* idle state

### Computational overhead

Measure the cost of:

```text
VAD + OSD + router
```

relative to:

```text
WhisperRT alone
```

---

# 25. End-to-End Metrics

Preserve the metrics from Phase 4:

* WER where valid
* RTF
* first-output latency
* end-of-utterance latency
* processing time
* incremental output behavior

Add adaptive-specific measurements.

Do not replace the existing metrics.

---

# 26. Recognition Evaluation

Because the Phase 6 overlap branch may initially be the same WhisperRT baseline, interpret results carefully.

Do not claim:

> adaptive routing improves ASR

unless the selected overlap branch actually provides a different recognition capability and the experiment demonstrates that improvement.

If the overlap branch is only a placeholder/control, the meaningful results are:

* routing correctness
* routing latency
* overhead
* branch-transition behavior
* infrastructure readiness

---

# 27. Synthetic Dataset Conditions

Use the Phase 3 synthetic dataset and Phase 4 experimental conditions.

Where available evaluate:

```text id="4v9s2r"
clean
no overlap
low overlap
medium overlap
high overlap
```

and relative source-level conditions.

Do not create a new dataset-generation mechanism.

---

# 28. Real-Time Constraint

The adaptive pipeline must remain compatible with real-time operation.

Measure:

```text id="x6q3u1"
audio duration
processing duration
RTF
routing latency
OSD latency
branch-switch latency
```

The objective is not merely accurate routing.

The pipeline must also be computationally feasible.

---

# 29. Experiment Matrix

Create a manageable default matrix.

At minimum:

```text id="j0v7px"
Always WhisperRT
Adaptive + Oracle OSD
Adaptive + Predicted OSD
```

Optionally:

```text id="n5f4ce"
Adaptive + OSD without VAD
Adaptive + OSD with VAD
Raw OSD threshold
Smoothed OSD threshold
```

Do not create an enormous sweep by default.

Allow larger experiments through configuration.

---

# 30. Configuration

Create:

```text id="7g3r9k"
configs/adaptive.yaml
```

It should expose:

```yaml id="c8v2rm"
experiment:
  name: adaptive_streaming_asr
  seed: 42

data:
  manifest: ...

model:
  normal_asr:
    backend: huggingface
    name: MLSpeech/WhisperRT-Streaming

vad:
  enabled: true

osd:
  enabled: true
  threshold: ...

routing:
  policy: ...
  smoothing:
    enabled: ...
  history_ms: ...

branches:
  normal: ...
  overlap: ...

evaluation:
  metrics:
    - wer
    - rtf
    - routing_f1
    - detection_latency
```

Use the existing config system.

---

# 31. Adaptive Pipeline Script

Create:

```text id="q7n2b5"
scripts/run_adaptive.py
```

or an equivalent project-consistent name.

It should:

1. load configuration
2. initialize VAD
3. initialize OSD
4. initialize routing controller
5. initialize ASR branches
6. process streaming audio
7. record routing events
8. collect transcript output
9. calculate metrics
10. save per-example results
11. save routing traces
12. generate reports

---

# 32. Notebook

Create:

```text id="1a4v6z"
notebooks/07_adaptive_pipeline.ipynb
```

Suggested structure:

```text id="f0h8q1"
1. Environment check
2. Load configuration
3. Inspect Phase 3 data
4. Inspect OSD output
5. Run always-normal baseline
6. Run oracle routing
7. Run predicted-OSD routing
8. Visualize routing timeline
9. Inspect transition examples
10. Compare latency/RTF
11. Compare recognition metrics where valid
12. Analyze failures
13. Save results
```

The notebook must use the project code rather than duplicate the implementation.

---

# 33. Visualization

Create useful adaptive-pipeline visualizations.

At minimum:

### Routing timeline

Show:

```text id="v6m2jx"
Ground truth overlap
OSD prediction
selected route
WhisperRT output activity
```

### Route state distribution

Show proportion of time spent in:

* no speech
* single speaker
* overlap

### Latency comparison

Compare:

```text
WhisperRT
vs
VAD + OSD + router
```

### Routing confusion

Where appropriate:

```text
ground-truth state
vs
selected route
```

---

# 34. Tests

Extend:

```text id="q1x9a4"
tests/test_pipeline.py
tests/test_streaming.py
tests/test_overlap.py
```

Test:

### State machine

* no speech → single
* single → overlap
* overlap → single
* single → no speech
* overlap → no speech

### Hysteresis

* threshold fluctuations
* enter threshold
* exit threshold
* minimum duration

### Routing

* correct branch selection
* oracle routing
* predicted OSD routing

### Buffering

* overlap onset
* overlap offset
* boundary conditions
* no audio loss
* no duplicate audio

### Transcript management

* duplicate partial output
* route transition
* finalization

### Failure handling

* missing OSD result
* branch error
* empty chunk
* end-of-stream

Tests should not require downloading large models.

---

# 35. Oracle Tests

Create unit tests using oracle OSD decisions.

This allows the routing logic to be tested independently from model quality.

For example:

```text id="b3v7n9"
given:
SINGLE → OVERLAP → SINGLE

expect:
normal → overlap → normal
```

This should be deterministic.

---

# 36. Integration Tests

Create a separate integration path for:

```text id="3x8m5n"
Audio
→ VAD
→ OSD
→ router
→ WhisperRT
→ final output
```

It may require:

* model downloads
* GPU
* Kaggle/HF
* real audio data

Keep it separate from lightweight unit tests.

---

# 37. Kaggle Compatibility

The notebook must support Kaggle.

It should work with:

* attached synthetic overlap data
* attached model artifacts or Hugging Face download
* configurable paths
* GPU when available

Do not require code modifications to switch between local and Kaggle environments.

---

# 38. Hugging Face Compatibility

Continue using the project's model backend.

Do not bypass the existing abstraction for WhisperRT, OSD, or future branches.

If a model is loaded from Hugging Face, record:

* repository ID
* revision/version where available
* configuration
* device/dtype

---

# 39. Failure Recovery

Streaming systems must handle branch errors gracefully.

Define what happens if:

* OSD fails
* OSD returns no result
* overlap branch is unavailable
* a chunk is malformed
* stream ends during overlap
* finalization occurs during a route transition

The default fallback must be deterministic.

For example, a safe fallback could be:

```text id="q5s2m7"
route to normal WhisperRT
```

but do not assume this is always appropriate. Make the policy explicit and configurable.

---

# 40. No Speaker Identity

The router must never attempt:

```text id="6s2k4w"
speaker A
speaker B
```

identification.

It only needs:

```text id="n7c4p2"
no speech
single speaker
overlap
```

This preserves the thesis scope.

---

# 41. No Final Multi-Talker ASR

Do not implement:

* serialized output training
* t-SOT
* source-specific decoding
* multi-talker Whisper modification
* speaker-attributed transcripts
* separation networks

These belong to Phase 7.

Phase 6 should make the branch interface ready for them.

---

# 42. Research Questions

Phase 6 should provide evidence for:

### RQ1

Can streaming OSD be used to control an adaptive ASR pipeline without breaking streaming behavior?

### RQ2

What routing errors occur because of OSD errors?

### RQ3

What latency does adaptive routing introduce?

### RQ4

How much computational overhead does VAD + OSD + routing add?

### RQ5

What is the difference between oracle routing and predicted-OSD routing?

### RQ6

Does the adaptive architecture provide a meaningful infrastructure for a future overlap-aware ASR branch?

Do not answer these questions before executing experiments.

---

# 43. Research Integrity

Do not report:

```text
adaptive system improves WER
```

unless an actual alternative overlap branch produces a measurable improvement.

If both branches use the same WhisperRT model, explicitly report that the adaptive experiment evaluates:

* routing behavior
* overhead
* state transitions
* infrastructure
* oracle-vs-predicted routing

rather than recognition improvement.

This distinction is important for the thesis.

---

# 44. Documentation

Update:

```text id="r5k3q8"
README.md
docs/architecture.md
docs/experiments.md
docs/thesis_mapping.md
```

Document:

* adaptive architecture
* state machine
* routing policy
* hysteresis
* buffering
* branch abstraction
* oracle routing
* predicted routing
* latency
* computational overhead
* limitations
* relationship to Phase 7

---

# 45. Phase Report

Create:

```text id="p9x4w2"
docs/phase_reports/phase_6_report.md
```

Include:

## Objective

What adaptive behavior was tested.

## Architecture

Actual implemented pipeline.

## Routing Policy

Actual state machine and threshold logic.

## Experimental Conditions

* always-normal
* oracle routing
* predicted OSD routing
* optional VAD ablations

## Results

Actual measured:

* WER where valid
* RTF
* latency
* routing precision/recall/F1
* transition latency
* overhead
* route distribution

## Error Analysis

Actual routing failures.

## Oracle vs Predicted

Quantify the effect of imperfect OSD.

## Limitations

Especially the absence of a real overlap-aware ASR branch if applicable.

## Phase 7 Requirements

Identify what the future overlap-aware/multi-talker ASR branch must provide.

---

# 46. Completion Criteria

Phase 6 is complete only when:

* [ ] Adaptive streaming pipeline exists.
* [ ] VAD and OSD are integrated without tightly coupling their implementations.
* [ ] Explicit routing state machine exists.
* [ ] Routing policy is configurable.
* [ ] Hysteresis/stabilization is configurable.
* [ ] Route transitions are logged.
* [ ] Boundary buffering is handled.
* [ ] Duplicate audio/output is prevented.
* [ ] WhisperRT state behavior across transitions is understood and documented.
* [ ] Normal ASR branch exists.
* [ ] Overlap branch interface exists.
* [ ] Default overlap branch is executable without pretending to be a multi-talker solution.
* [ ] Oracle OSD/routing mode exists.
* [ ] Predicted OSD/routing mode exists.
* [ ] Always-normal baseline exists.
* [ ] Routing metrics are measured.
* [ ] Detection-to-routing latency is measured.
* [ ] Computational overhead is measured.
* [ ] Existing ASR/streaming metrics remain available.
* [ ] Tests cover state transitions and routing.
* [ ] Integration test path exists.
* [ ] Kaggle notebook works or its execution limitations are documented.
* [ ] `notebooks/07_adaptive_pipeline.ipynb` exists.
* [ ] `configs/adaptive.yaml` exists.
* [ ] README/docs are updated.
* [ ] Phase 6 report exists.
* [ ] Existing Phase 1–5 functionality remains intact.
* [ ] `scripts/smoke_test.py` remains functional.
* [ ] No diarization is implemented.
* [ ] No speaker identification is implemented.
* [ ] No source separation is implemented.
* [ ] No final multi-talker ASR is implemented.
* [ ] No ASR fine-tuning/PEFT/RL is implemented.

---

# 47. Final Execution Rule

Do not implement Phase 7 during this task.

At the end of Phase 6, stop and report:

1. files created/modified
2. adaptive architecture
3. routing state machine
4. routing policy
5. overlap branch implementation
6. oracle-routing implementation
7. predicted-OSD implementation
8. experiment conditions
9. commands executed
10. actual measured results
11. routing metrics
12. latency/RTF/overhead
13. transition/buffering behavior
14. failure examples
15. tests
16. smoke-test status
17. notebook status
18. limitations
19. exact requirements for the Phase 7 overlap-aware/multi-talker ASR branch

The next phase will implement and evaluate the **actual overlap-aware / multi-talker ASR branch** that the adaptive controller can route to when overlap is detected.

Do not implement that branch during Phase 6.
