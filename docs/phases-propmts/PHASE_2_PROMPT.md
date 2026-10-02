# PHASE 2 — Streaming Voice Activity Detection

You are implementing **Phase 2** of the project:

> **English Streaming ASR with Overlapped Speech**

This phase operates under the previously defined **Master Prompt / Project Contract**, after Phase 0 infrastructure and Phase 1 WhisperRT streaming baseline have been implemented.

The objective of Phase 2 is to introduce a **real streaming Voice Activity Detection (VAD) component** into the architecture and evaluate its behavior independently and as part of the streaming ASR pipeline.

The VAD must operate incrementally on incoming audio chunks.

---

# 1. Phase 2 Objective

Implement:

```text
Streaming Audio
       ↓
Streaming VAD
       ↓
Speech / Non-Speech Decision
       ↓
WhisperRT Streaming ASR
       ↓
Incremental Transcript
```

The VAD must be:

* stateful
* incremental
* low-latency
* chunk-based
* configurable
* replaceable
* independently testable

The implementation must support later integration with:

```text
Streaming OSD
Adaptive Routing
Overlap-aware ASR
```

but none of those components should be implemented in this phase.

---

# 2. Critical Scope Boundary

Phase 2 MUST implement:

* real streaming VAD
* VAD abstraction
* VAD backend selection
* streaming VAD state
* speech/non-speech decisions
* VAD timing
* VAD evaluation
* integration with WhisperRT
* VAD-aware streaming pipeline
* VAD configuration
* VAD tests
* VAD benchmark
* Phase 2 notebook
* documentation updates

Phase 2 MUST NOT implement:

* overlap speech detection
* speaker diarization
* speaker identification
* speaker enrollment
* multi-talker ASR
* speech separation
* adaptive routing
* overlap-aware recognition
* synthetic overlap generation
* LibriCSS evaluation
* LibriSpeechMix evaluation
* ASR fine-tuning
* PEFT
* RL
* OSD-specific training

The distinction is critical:

> **VAD answers whether speech activity is present. It does not determine whether one or multiple speakers are talking.**

---

# 3. First Step — Inspect Current Repository

Before modifying anything:

1. Inspect the current repository.
2. Verify Phase 0 infrastructure.
3. Verify Phase 1 WhisperRT integration.
4. Run existing tests.
5. Run the Phase 1 development benchmark if available.
6. Inspect:

   * `models/vad.py`
   * `pipeline/streaming.py`
   * `streaming/engine.py`
   * `streaming/state.py`
   * audio chunking
   * configuration system
   * metrics
   * tests
   * Phase 1 notebook
7. Preserve all working Phase 1 behavior.

Run:

```bash
pytest -q
```

and the existing Phase 1 smoke/development command.

Do not rebuild the architecture unnecessarily.

---

# 4. VAD Architecture

The target architecture becomes:

```text
Audio Stream
      ↓
Audio Buffer / Chunker
      ↓
Streaming VAD
      ↓
Speech Activity State
      ↓
WhisperRT Streaming ASR
      ↓
Transcript
      ↓
Evaluation
```

The VAD must sit before the ASR component.

However, preserve the ability to run:

```text
Audio → WhisperRT
```

without VAD for comparison against the Phase 1 baseline.

Therefore the pipeline must support:

```text
VAD disabled:
Audio → WhisperRT

VAD enabled:
Audio → VAD → WhisperRT
```

This comparison is essential.

---

# 5. VAD Abstraction

Extend:

```text
src/streaming_asr/models/vad.py
```

with a clean streaming interface.

Conceptually:

```python
class StreamingVAD:
    def start(self):
        ...

    def process(self, audio_chunk):
        ...

    def finalize(self):
        ...
```

The exact interface may be adapted to the Phase 0/1 architecture.

The VAD must preserve state between chunks.

Do NOT recreate the VAD model for every chunk.

---

# 6. VAD Output Contract

Each VAD processing step should return a structured result.

At minimum:

```text
speech_probability
is_speech
start_time
end_time
chunk_index
```

If the selected VAD provides additional useful information, it may be preserved.

For example:

```text
speech_start
speech_end
confidence
```

but do not add unnecessary complexity.

The core semantic output is:

```text
SPEECH
NON_SPEECH
```

Do not introduce:

```text
SINGLE_SPEAKER
OVERLAP
```

in Phase 2.

Those belong to OSD.

---

# 7. VAD State

The VAD state must support streaming behavior.

Conceptually:

```text
VADState:
    started
    previous_decision
    speech_probability
    speech_duration
    silence_duration
    current_segment_start
    current_segment_end
```

The exact implementation can differ.

The important requirement is that the VAD can maintain temporal context across chunks.

---

# 8. VAD Backend Selection

Do not hard-code one VAD implementation into the pipeline.

The architecture must support interchangeable VAD backends.

Candidate backends may include:

* WebRTC VAD
* Silero VAD
* TEN VAD
* another verified streaming-capable VAD

However:

> Do not select a backend merely because it is popular.

First verify whether the implementation genuinely supports the required streaming/incremental use case.

---

# 9. Candidate Selection

Before implementing the real backend, investigate the currently available candidate implementations.

Evaluate candidates according to:

| Criterion                       | Requirement        |
| ------------------------------- | ------------------ |
| Streaming/incremental operation | Required           |
| Stateful processing             | Required           |
| English                         | Required           |
| 16 kHz audio                    | Preferred          |
| CPU support                     | Strongly preferred |
| Low latency                     | Important          |
| Python integration              | Important          |
| Offline operation               | Preferred          |
| Configurable thresholds         | Preferred          |
| Open-source accessibility       | Preferred          |

Use current official documentation/source when determining API behavior.

Do not assume that an offline VAD can simply be made streaming by slicing the waveform.

---

# 10. Recommended Design

Use the architecture:

```text
VADBackend
    │
    ├── WebRTCVADBackend
    ├── SileroVADBackend
    └── TENVADBackend
```

Only implement backends that can actually satisfy the streaming requirements.

If one backend is selected as the primary implementation, document why.

Do not implement three complete VAD systems merely for architectural symmetry.

A clean primary backend plus a backend abstraction is sufficient.

---

# 11. Primary VAD Selection

The final Phase 2 implementation must identify:

```text
Primary VAD:
<actual selected implementation>
```

and document:

* package/model
* version
* official API
* sample-rate requirements
* frame/chunk requirements
* state behavior
* threshold behavior
* latency characteristics
* CPU/GPU requirements
* known limitations

If the candidate selection remains uncertain after investigation:

```text
RESEARCH DECISION PENDING
```

must be used rather than silently making an unsupported assumption.

---

# 12. VAD Configuration

Add or update:

```text
configs/streaming_vad.yaml
```

Example structure:

```yaml
experiment:
  name: streaming_vad_librispeech
  seed: 42

runtime:
  environment: auto
  device: auto

data:
  backend: huggingface
  dataset: librispeech
  split: test-clean

audio:
  sample_rate: 16000
  channels: 1
  chunk_ms: 320

vad:
  enabled: true
  backend: <selected_backend>
  threshold: 0.5
  min_speech_ms: 0
  min_silence_ms: 0

asr:
  enabled: true
  model:
    backend: huggingface
    name: MLSpeech/WhisperRT-Streaming

evaluation:
  metrics:
    - vad_precision
    - vad_recall
    - vad_f1
    - vad_latency
    - wer
    - rtf
```

Do not copy these exact threshold values without checking whether they make sense for the selected VAD.

---

# 13. VAD Thresholds

If the selected VAD provides probabilities, expose the threshold through configuration.

For example:

```yaml
vad:
  threshold: 0.5
```

Do not claim `0.5` is optimal.

It is merely a configurable baseline.

Later experiments may investigate:

```text
0.3
0.5
0.7
```

or other values appropriate to the backend.

Do not conduct an extensive hyperparameter search in Phase 2 unless explicitly configured.

---

# 14. Frame and Chunk Compatibility

A major technical requirement is reconciling:

```text
audio streaming chunk size
```

with:

```text
VAD frame size
```

The VAD may require fixed frame sizes that differ from the ASR chunk size.

Implement a buffering layer if necessary:

```text
Incoming ASR Chunk
       ↓
VAD Frame Buffer
       ↓
VAD Frames
       ↓
VAD Decisions
```

Do not assume:

```text
ASR chunk size == VAD frame size
```

The VAD adapter should hide this implementation detail from the rest of the pipeline.

---

# 15. Streaming Requirement

The VAD must not receive the entire utterance before making decisions.

Invalid:

```text
load complete utterance
        ↓
VAD complete utterance
        ↓
generate labels
```

Valid:

```text
chunk 1 → VAD state → decision
chunk 2 → VAD state → decision
chunk 3 → VAD state → decision
...
```

The implementation must preserve temporal state.

---

# 16. Synthetic Audio Unit Tests

Create deterministic synthetic audio tests for:

1. silence
2. speech-like signal
3. silence → signal
4. signal → silence
5. silence → signal → silence
6. consecutive speech chunks
7. consecutive silence chunks

Do not interpret a sine wave as actual VAD accuracy.

Synthetic signals are only for testing:

* state transitions
* buffering
* timing
* lifecycle
* API behavior

They are not benchmark evidence.

---

# 17. Real Speech Evaluation

Use LibriSpeech clean speech to test VAD on real speech.

For Phase 2, the primary dataset can remain:

```text
LibriSpeech test-clean
```

No overlap is needed.

The evaluation should produce speech activity decisions over real English utterances.

---

# 18. Ground-Truth VAD Labels

LibriSpeech transcripts provide utterance-level speech regions but not necessarily frame-level speech/non-speech annotations in the exact form needed.

Therefore carefully define how VAD ground truth is constructed.

Possible approach:

```text
Known utterance boundaries
+
known audio duration
```

can provide coarse utterance-level speech activity.

However, do NOT automatically assume that the entire file is speech.

Investigate the actual LibriSpeech audio structure and annotations.

If precise frame-level ground truth cannot be established from the chosen data:

* document the limitation,
* do not claim frame-level VAD accuracy,
* use a suitable evaluation setup or another dataset with appropriate annotations.

This is an important research decision.

---

# 19. VAD Evaluation Options

Evaluate at the level actually supported by the available ground truth.

Potential metrics:

### Precision

```text
TP / (TP + FP)
```

### Recall

```text
TP / (TP + FN)
```

### F1

```text
2PR / (P + R)
```

Potential additional measures:

* false alarm rate
* miss rate
* segment-level accuracy
* detection latency

Do not calculate frame-level metrics without valid frame-level reference labels.

---

# 20. VAD Latency

Measure the delay between actual speech onset and VAD speech detection where the ground truth allows this.

For example:

```text
speech onset
     ↓
first VAD speech decision
```

Measure:

```text
detection_latency
```

Similarly, if meaningful:

```text
speech end
     ↓
VAD transitions to non-speech
```

Measure end detection latency.

If ground truth is insufficient:

```text
NOT MEASURED
```

---

# 21. VAD + WhisperRT Integration

Extend:

```text
src/streaming_asr/pipeline/streaming.py
```

so it can support:

### Baseline mode

```text
Audio
 ↓
WhisperRT
```

### VAD mode

```text
Audio
 ↓
VAD
 ↓
WhisperRT
```

The implementation must preserve the Phase 1 baseline.

---

# 22. Critical ASR Behavior Decision

Do not automatically discard all non-speech audio before WhisperRT without understanding the consequences.

There are several possible designs:

### Design A

```text
VAD gates ASR processing
```

### Design B

```text
VAD informs ASR state but audio remains continuous
```

### Design C

```text
VAD controls when ASR decoding is triggered
```

Determine which behavior is compatible with WhisperRT's streaming architecture.

Do not choose purely based on intuition.

Document the selected behavior and why.

---

# 23. Preserve Streaming Context

The VAD must not break WhisperRT's streaming state.

For example, do not implement:

```text
speech
 ↓
run ASR
 ↓
silence
 ↓
destroy ASR state
 ↓
speech
 ↓
create new ASR
```

unless the actual experimental design explicitly requires utterance segmentation and the consequences are documented.

The primary pipeline should remain capable of continuous streaming.

---

# 24. VAD Gating Experiment

Implement a comparison between:

### Experiment A — Phase 1 baseline

```text
Audio → WhisperRT
```

and:

### Experiment B — VAD integrated

```text
Audio → Streaming VAD → WhisperRT
```

Measure:

* WER
* RTF
* first-output latency
* end-of-utterance latency
* VAD processing overhead

The objective is not to declare one approach universally better.

The purpose is to quantify the effect of adding VAD.

---

# 25. VAD Overhead

Measure the computational cost of VAD separately where possible:

```text
VAD processing time
ASR processing time
total pipeline time
```

This allows:

```text
ASR-only
```

to be compared against:

```text
VAD + ASR
```

without hiding the VAD computational cost.

---

# 26. ASR Metric Interpretation

If VAD gating changes the audio presented to WhisperRT, document exactly how.

For example:

```text
continuous audio preserved
```

or:

```text
non-speech frames skipped
```

This is essential because WER differences can otherwise be difficult to interpret.

---

# 27. VAD Segment Representation

If the VAD produces speech regions, define a generic segment object such as:

```python
SpeechSegment(
    start_time=...,
    end_time=...,
    confidence=...,
)
```

This representation should later be usable by:

```text
OSD
Adaptive Router
```

Do not add speaker identity.

---

# 28. OSD Boundary

The output of Phase 2 is only:

```text
speech
non-speech
```

It must NOT become:

```text
single speaker
overlap
```

The future OSD module will consume audio/speech regions and make the overlap decision.

---

# 29. Tests

Extend:

```text
tests/test_vad.py
```

with tests for:

* initialization
* configuration
* backend selection
* frame buffering
* state persistence
* speech decision
* non-speech decision
* timestamps
* reset
* finalize
* threshold behavior
* invalid frame sizes
* sample-rate validation

Also add relevant tests to:

```text
tests/test_pipeline.py
tests/test_streaming.py
tests/test_metrics.py
```

Ensure all Phase 0 tests remain passing.

---

# 30. Real VAD Integration Test

Create a separate real-model integration test.

It should verify:

```text
real audio
 ↓
real VAD
 ↓
streaming decisions
```

Do not make the default unit-test suite download VAD model weights.

Use a separate integration marker or explicit test command.

---

# 31. Phase 2 Smoke Test

Update:

```text
scripts/smoke_test.py
```

so Phase 2 can optionally run a real VAD smoke test.

The default Phase 0 smoke test must remain lightweight.

Provide an explicit Phase 2 command such as:

```bash
python scripts/smoke_test.py --phase 2 --real-vad
```

or an equivalent configuration-driven approach.

The real VAD smoke test should use a tiny audio example.

It must verify:

* model/backend loads
* streaming state initializes
* several chunks are processed
* decisions are returned
* finalization works

---

# 32. Notebook

Create:

```text
notebooks/03_streaming_vad.ipynb
```

The notebook must contain:

1. Environment
2. Configuration
3. VAD backend information
4. Model initialization
5. Synthetic streaming demonstration
6. Real LibriSpeech example
7. VAD decisions
8. Speech segment visualization
9. VAD metrics where ground truth permits
10. VAD latency
11. VAD + WhisperRT integration
12. ASR comparison
13. Results
14. Limitations

Do not implement the actual VAD logic inside notebook cells.

Use project code.

---

# 33. Visualization

Where useful, create a simple visualization showing:

```text
Audio waveform
Speech / non-speech regions
```

The visualization should help verify temporal alignment.

Do not use the visualization as a substitute for quantitative evaluation.

---

# 34. Experiment Configuration

Support at least:

```text
VAD backend
VAD threshold
VAD frame size
ASR chunk size
device
dtype
dataset
dataset split
max samples
```

Do not hard-code these values inside the implementation.

---

# 35. Experiment Outputs

Store:

```text
outputs/
├── predictions/
├── metrics/
├── figures/
├── reports/
└── logs/
```

Use the existing Phase 0 run directory.

Per-sample VAD output may contain:

```json
{
  "sample_id": "...",
  "speech_segments": [
    {
      "start": 0.0,
      "end": 1.42,
      "confidence": 0.91
    }
  ]
}
```

Only include confidence if the selected backend actually provides it.

---

# 36. VAD Benchmark Report

Generate a Phase 2 report containing:

## Backend

* VAD implementation
* package/version
* model/version if applicable

## Audio

* sample rate
* frame size
* ASR chunk size

## Dataset

* dataset
* split
* number of samples
* total duration

## VAD metrics

* precision
* recall
* F1
* miss rate
* false alarm rate
* detection latency

Only include metrics that have valid ground truth and were actually measured.

## ASR impact

* baseline WER
* VAD pipeline WER
* baseline RTF
* VAD pipeline RTF
* latency comparison

Again, use actual measurements.

---

# 37. Text / ASR Evaluation

The VAD itself should not alter transcript normalization.

Use the same WER normalization used in Phase 1.

This allows an apples-to-apples comparison.

---

# 38. README Update

Add:

```text
## Phase 2 — Streaming VAD
```

Document:

* purpose
* VAD definition
* selected backend
* architecture
* installation
* configuration
* commands
* dataset
* evaluation
* metrics
* VAD + ASR integration
* limitations
* current status

Explicitly state:

> Phase 2 does not perform overlap detection.

---

# 39. Documentation Updates

Update:

```text
docs/architecture.md
docs/datasets.md
docs/experiments.md
docs/thesis_mapping.md
```

Architecture should now show:

```text
Audio
 ↓
Streaming VAD
 ↓
WhisperRT Streaming ASR
 ↓
Transcript
```

The thesis mapping should describe Phase 2 as:

> Establishing a streaming speech-activity layer that enables later temporal routing and overlap detection.

Do not claim that it solves overlapping speech.

---

# 40. Dataset Strategy

Continue using clean English speech in Phase 2.

Primary:

```text
LibriSpeech
```

Do NOT introduce:

```text
LibriSpeechMix
LibriCSS
AMI
CHiME-6
```

yet unless a specific ground-truth requirement makes another dataset necessary.

If another dataset is required specifically for VAD evaluation, document that as a research decision and keep the addition minimal.

---

# 41. Performance Benchmark

Measure VAD computational behavior independently where possible.

At minimum:

```text
VAD processing time
```

and, where meaningful:

```text
VAD RTF
```

using:

```text
VAD processing time / audio duration
```

Do not confuse:

```text
VAD RTF
```

with:

```text
full pipeline RTF
```

---

# 42. Streaming Correctness Test

Create a test proving that:

```text
processing audio as chunks
```

maintains consistent state.

Where the selected VAD guarantees equivalent behavior, compare:

```text
chunked processing
```

against an appropriate reference.

Do not assume bit-identical output if the model/backend is inherently stateful or approximate.

The goal is to verify correct streaming lifecycle, not manufacture numerical equivalence.

---

# 43. Reset Behavior

Verify:

```text
Stream A
 → process
 → finalize

Stream B
 → process
 → finalize
```

does not leak VAD state.

This must be tested.

---

# 44. Error Handling

The VAD implementation must provide clear errors for:

* unsupported sample rate
* unsupported frame size
* missing package
* missing model
* invalid threshold
* unavailable device
* malformed audio
* invalid lifecycle state

Do not silently resample or reshape data in a way that changes the experimental configuration without recording it.

---

# 45. Dependency Management

Only add dependencies actually required by the selected VAD.

Do not install:

```text
all possible VAD packages
```

unless there is a specific reason.

Keep the environment reproducible.

Document optional dependencies separately where appropriate.

---

# 46. Kaggle Compatibility

Phase 2 must remain Kaggle-compatible.

Support:

```text
/kaggle/input
/kaggle/working
```

without assuming:

* Internet
* GPU
* pre-existing model cache
* credentials

Document how the VAD model/package is supplied in:

1. Internet-enabled Kaggle
2. attached Kaggle input

Never write model/data outputs into `/kaggle/input`.

---

# 47. Research Integrity

Do not fabricate:

* VAD precision
* VAD recall
* F1
* latency
* RTF
* WER
* dataset statistics

If frame-level ground truth is unavailable:

```text
NOT MEASURED
```

If a backend is only proposed:

```text
RESEARCH DECISION PENDING
```

If a value is from official documentation:

clearly identify it as externally documented rather than experimentally measured.

---

# 48. Phase 2 Success Criteria

Phase 2 is complete only if:

* [ ] Phase 0 remains functional.
* [ ] Phase 1 WhisperRT baseline remains functional.
* [ ] A real streaming VAD backend is selected and documented.
* [ ] The VAD is genuinely stateful/incremental.
* [ ] VAD operates on incoming chunks.
* [ ] VAD frame buffering works where necessary.
* [ ] VAD output has a stable interface.
* [ ] Speech/non-speech decisions are produced.
* [ ] VAD state resets correctly.
* [ ] VAD lifecycle is tested.
* [ ] Real speech is processed.
* [ ] VAD metrics are calculated only where valid ground truth exists.
* [ ] VAD processing overhead is measured.
* [ ] VAD + WhisperRT pipeline works.
* [ ] Phase 1 baseline can still be run without VAD.
* [ ] Baseline vs VAD pipeline comparison is possible.
* [ ] WER comparison is measured if the experiment is run.
* [ ] RTF comparison is measured if the experiment is run.
* [ ] `tests/test_vad.py` is implemented.
* [ ] `03_streaming_vad.ipynb` exists.
* [ ] Phase 2 configuration exists.
* [ ] Phase 2 smoke test exists.
* [ ] README is updated.
* [ ] documentation is updated.
* [ ] no OSD is implemented.
* [ ] no overlap detection is implemented.
* [ ] no diarization is implemented.
* [ ] no multi-talker ASR is implemented.
* [ ] no ASR fine-tuning is implemented.

---

# 49. Final Phase 2 Report

After implementation, provide:

## 1. Implementation Summary

## 2. Selected VAD Backend

Include:

* name
* version
* API
* streaming behavior
* reason for selection

## 3. Architecture Changes

Explain:

```text
Audio → VAD → WhisperRT
```

and how the original:

```text
Audio → WhisperRT
```

baseline remains available.

## 4. Files Created/Modified

List important files.

## 5. Dataset

Report actual dataset/subset used.

## 6. VAD Configuration

Report:

* frame size
* threshold
* sample rate
* device

## 7. VAD Results

Report only measured metrics.

## 8. ASR Comparison

Compare Phase 1 and Phase 2 using actual measurements.

## 9. Tests

Report:

```text
pytest: PASS/FAIL
Phase 1 regression: PASS/FAIL
real VAD integration: PASS/FAIL
VAD smoke test: PASS/FAIL
LibriSpeech development run: PASS/FAIL
full benchmark: PASS/FAIL/NOT RUN
Kaggle validation: PASS/FAIL/NOT RUN
```

## 10. Limitations

Especially:

* clean speech
* no overlap
* no OSD
* no diarization
* ground-truth limitations

## 11. Research Decisions Pending

Identify unresolved choices.

## 12. Phase 3 Readiness

State whether the repository is ready for:

> **Phase 3 — Synthetic Overlap Dataset Generation**

Do not implement Phase 3.

---

# 50. Final Instruction

Implement **ONLY Phase 2**.

The central deliverable is a **real, stateful, streaming VAD integrated with the existing WhisperRT streaming architecture and experimentally evaluated on English speech**.

Do not turn VAD into OSD.

Do not detect overlapping speakers.

Do not add diarization.

Do not add multi-talker ASR.

Do not modify WhisperRT architecture.

Do not fine-tune ASR.

Do not fabricate metrics.

Preserve the Phase 1 clean-speech baseline.

After Phase 2 implementation and validation, stop and provide the Phase 2 completion report.

The next phase will be:

> **PHASE 3 — Synthetic Overlap Dataset Generation**
