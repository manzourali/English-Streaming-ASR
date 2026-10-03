# PHASE 5 — Overlap Speech Detection (OSD)

You are implementing **Phase 5** of the English Streaming ASR with Overlapped Voices research project.

The project operates under the previously defined **Master Prompt / project contract**. Follow all architecture, reproducibility, testing, Kaggle/Hugging Face, documentation, and research-integrity requirements from that contract.

Do not implement Phase 6 functionality prematurely.

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

Synthetic two-speaker overlap dataset generation with source timing and overlap ground truth.

### Phase 4

WhisperRT evaluation under synthetic overlap.

Phase 5 now introduces:

> **Streaming Overlap Speech Detection (OSD)**

The purpose is to determine, from an incoming audio stream, whether the current speech contains:

```text
NO_SPEECH
SINGLE_SPEAKER
OVERLAP
```

or an equivalent state/probability representation.

---

# 2. Main Objective

Build and evaluate a **streaming OSD component** that can identify whether speech from multiple sources overlaps.

The OSD component must:

1. operate incrementally on streaming audio
2. maintain state across chunks
3. produce decisions or probabilities for the current audio region
4. use the ground-truth timing generated in Phase 3 for evaluation
5. be independent of WhisperRT
6. be replaceable with another OSD backend
7. provide a clean interface for Phase 6 adaptive routing

The OSD system must not perform:

* diarization
* speaker identification
* speaker enrollment
* speaker verification
* transcription
* source separation

---

# 3. Critical Conceptual Distinction

Keep these concepts separate:

```text
VAD:
Is there speech?

OSD:
Is more than one speaker speaking simultaneously?

Diarization:
Who is speaking and when?

ASR:
What was said?

Source separation:
Can the individual speech signals be separated?
```

The Phase 5 OSD component answers only:

> **Is the current speech overlapping?**

It does not need to know who the speakers are.

---

# 4. Target Output

The OSD abstraction should support a representation such as:

```python
class StreamingOverlapDetector:
    def process(self, audio_chunk):
        ...
```

A result could conceptually contain:

```python
{
    "state": "NO_SPEECH",
    "overlap_probability": 0.02,
    "speech_probability": 0.98,
    "timestamp": ...,
}
```

or:

```python
{
    "state": "SINGLE_SPEAKER",
    "overlap_probability": 0.12,
    ...
}
```

The exact schema can differ if a better design is justified.

The important requirement is that the interface remains stable even if the underlying OSD model changes.

---

# 5. Required States

The initial conceptual state space is:

```text
NO_SPEECH
SINGLE_SPEAKER
OVERLAP
```

However, do not force a three-class model if the selected OSD backend naturally produces only a binary:

```text
NON_OVERLAP
OVERLAP
```

representation.

In that case, combine it with the existing Phase 2 VAD state to derive:

```text
VAD = no speech
VAD = speech + OSD = non-overlap
VAD = speech + OSD = overlap
```

The final architecture should preserve the semantic distinction.

---

# 6. Streaming-First Requirement

OSD must be genuinely streaming.

Do not implement:

```text
complete audio
→ offline OSD
→ artificially divide predictions into chunks
```

Instead:

```text
audio chunk
    ↓
OSD state
    ↓
prediction
    ↓
next audio chunk
    ↓
updated OSD state
```

The detector must maintain state when the underlying model/algorithm requires context.

Do not reset the model for every chunk.

---

# 7. Frame/Chunk Alignment

The OSD model may require a frame size different from the ASR chunk size.

Build or reuse an appropriate buffering layer.

Conceptually:

```text
Streaming Audio
      ↓
Audio Buffer
      ├── VAD frame
      ├── OSD frame
      └── ASR chunk
```

Do not force VAD, OSD, and WhisperRT to use identical frame sizes.

The interfaces should tolerate different temporal resolutions.

Document:

* input sample rate
* frame size
* hop size
* algorithmic/context window
* output update interval
* expected detection latency

---

# 8. OSD Model Selection

Do not assume a particular OSD model before investigating the available options.

Research and inspect candidate approaches that can realistically satisfy:

* English speech
* streaming/online operation
* overlap detection
* publicly available implementation or reproducible method
* reasonable GPU/CPU requirements
* compatibility with the project's 16 kHz streaming pipeline
* suitable licensing for research use where relevant

Potential candidate families may include:

* dedicated overlap speech detection models
* speech activity models extended to overlap detection
* multi-speaker detection models
* frame-level speaker-count estimation models
* models available through Hugging Face
* lightweight neural OSD models
* conventional signal/model-based approaches where appropriate

Do not select a model merely because it is popular.

Evaluate its actual streaming suitability.

---

# 9. Candidate Evaluation Before Final Selection

Before committing to one OSD backend, document the candidate-selection process.

For each serious candidate consider:

```text
Model / Method
Repository
Paper
License
Input format
Streaming capability
Required context
Frame rate
Output
Number of classes
Training data
English support
Inference requirements
CPU/GPU feasibility
Public implementation
```

Do not fabricate information.

If a candidate is not actually streaming, classify it accordingly.

Do not call an offline model "streaming" merely because it can process short windows.

---

# 10. Streaming Suitability

For every candidate, distinguish between:

### Native streaming

The model is explicitly designed for causal/online processing.

### Low-latency windowed inference

The model processes short windows but is not truly causal.

### Offline

The model requires future context or complete recordings.

The preferred candidate should be the strongest practical option that satisfies the project's streaming requirement.

If a technically useful candidate is not truly streaming, it may be evaluated as a reference baseline, but it must not silently become the project's streaming OSD.

---

# 11. Primary Dataset for OSD Evaluation

Use the synthetic overlap dataset created in Phase 3 because it provides exact source timing.

Ground-truth classes can be derived from the source timing.

For example:

```text
no active speaker
→ NO_SPEECH

exactly one active source
→ SINGLE_SPEAKER

two active sources
→ OVERLAP
```

Do not derive ground truth from the OSD model's own predictions.

---

# 12. Ground-Truth Generation

Implement a deterministic ground-truth conversion layer.

Given source intervals:

```text
A: start_A → end_A
B: start_B → end_B
```

construct frame-level or segment-level labels.

For each temporal frame:

```text
active_sources = number of source intervals covering frame
```

Then:

```text
0 → NO_SPEECH
1 → SINGLE_SPEAKER
2+ → OVERLAP
```

This should be implemented independently from the detector.

That independence is important for evaluation integrity.

---

# 13. Temporal Resolution

The ground truth and prediction must have a defined temporal resolution.

Do not compare:

```text
10 ms ground truth
vs
500 ms prediction
```

without explicitly defining the alignment procedure.

Document:

* ground-truth frame duration
* OSD output resolution
* alignment rule
* tolerance/windowing
* timestamp convention

If the OSD has inherent detection latency, measure and report it rather than silently shifting predictions until they look better.

---

# 14. Primary OSD Metrics

At minimum evaluate:

### Frame/segment-level

* Precision
* Recall
* F1

for overlap detection.

Where appropriate, also report:

* false positive rate
* false negative rate
* confusion matrix

The primary research metric should be clearly identified.

Do not collapse everything into a single score.

---

# 15. Class-Specific Metrics

If using:

```text
NO_SPEECH
SINGLE_SPEAKER
OVERLAP
```

report class-specific performance where meaningful.

Especially report:

```text
OVERLAP:
precision
recall
F1
```

because overlap detection is the central purpose of this phase.

Do not report only overall accuracy if overlap is relatively rare.

---

# 16. Segment/Event-Level Evaluation

Where practical, additionally measure overlap events.

For each true overlap region, determine whether the detector:

* detected it
* missed it
* detected it late
* stopped detecting it early

Potential measurements include:

```text
overlap event recall
mean detection delay
mean detection duration error
```

Keep event-level metrics separate from frame-level metrics.

---

# 17. Detection Latency

Streaming OSD must be evaluated for latency.

Measure:

```text
true overlap start
        ↓
first overlap prediction
        ↓
detection delay
```

For example:

```text
OSD detection latency =
first_valid_overlap_prediction_time
-
ground_truth_overlap_start
```

Use the actual timestamp conventions of the implementation.

Do not hide negative/early predictions.

Define how early predictions are handled.

---

# 18. Hysteresis / Temporal Smoothing

Do not immediately introduce aggressive smoothing merely to maximize F1.

The first implementation should establish the raw detector behavior.

If thresholding or temporal smoothing is introduced, make it configurable.

For example:

```yaml
osd:
  threshold: ...
  smoothing:
    enabled: false
```

Then later evaluate smoothing as an explicit experiment.

The goal is to preserve causal behavior and understand the raw model first.

---

# 19. Probability Output

If the selected model produces probabilities or scores, preserve them.

Do not immediately convert everything into hard labels.

Store:

```text
timestamp
overlap_probability
decision
```

where available.

This will be useful for Phase 6 adaptive routing because routing may benefit from confidence rather than a hard binary decision.

---

# 20. Threshold Analysis

If the model provides an overlap probability/score, support configurable thresholding.

Evaluate a reasonable threshold range rather than assuming:

```text
threshold = 0.5
```

is automatically optimal.

Potential analysis:

```text
threshold
precision
recall
F1
false positives
false negatives
detection latency
```

Do not select a final thesis threshold based on the test set.

Use validation/development data for threshold selection where applicable.

---

# 21. Dataset Split Integrity

Respect the Phase 3 train/validation/test separation.

If OSD model inference does not involve training, the test set remains the final evaluation set.

If a trainable OSD model is selected, do not train on the test set.

Use:

```text
train → development/training
validation → threshold/model selection
test → final evaluation
```

Document the split usage.

---

# 22. If the Candidate Requires Training

A candidate OSD model may require fine-tuning/training on the synthetic data.

If so:

1. determine whether training is genuinely necessary
2. document the training objective
3. keep training separate from the evaluation pipeline
4. use train/validation/test separation
5. do not implement large-scale training unnecessarily
6. keep the model interface identical regardless of whether weights are pretrained or trained

However:

**Do not automatically train an OSD model merely because training is possible.**

First establish whether an existing pretrained model can provide a useful baseline.

---

# 23. Synthetic Data Limitation

The synthetic overlap dataset is useful because its ground truth is precise.

However, it may not represent real conversational overlap.

Therefore Phase 5 must explicitly distinguish:

```text
Synthetic OSD evaluation
```

from:

```text
Real-world OSD evaluation
```

Real-world evaluation may be introduced later using datasets such as LibriCSS or AMI.

Do not claim general real-world OSD performance from synthetic data alone.

---

# 24. Optional Real-World Validation

If a suitable publicly accessible real-world dataset can be integrated without significantly expanding the phase scope, you may create an **optional evaluation path**.

Do not make it mandatory if the dataset's annotation format, access, or setup would destabilize the phase.

Any real-world result must clearly identify:

* dataset
* split
* annotation type
* evaluation protocol
* limitations

Do not mix synthetic and real-world metrics into one aggregate number.

---

# 25. Integration With Phase 2 VAD

OSD and VAD should remain separate modules.

Conceptually:

```text
Audio
  ↓
Streaming VAD
  ↓
speech?
  ├── no → NO_SPEECH
  │
  └── yes
       ↓
   Streaming OSD
       ↓
   SINGLE_SPEAKER / OVERLAP
```

However, the OSD implementation should not become tightly coupled to a specific VAD backend.

It must remain independently testable.

---

# 26. Integration With WhisperRT

Do not alter WhisperRT.

For Phase 5, the conceptual pipeline becomes:

```text
Audio
   ├──→ Streaming VAD
   │
   └──→ Streaming OSD
             ↓
        OSD prediction
```

WhisperRT may be used only to verify that the OSD component can coexist with the existing streaming architecture.

Do not route audio differently based on OSD yet.

That belongs to Phase 6.

---

# 27. No Adaptive Routing

This is a strict boundary.

Do **not** implement:

```text
if overlap:
    use model A
else:
    use model B
```

Do not switch WhisperRT behavior based on OSD.

Phase 5 only produces the detection signal.

Phase 6 will consume that signal.

---

# 28. OSD Backend Abstraction

Implement an interchangeable backend.

Conceptually:

```python
class StreamingOverlapDetector:
    def process(self, audio_chunk):
        ...

    def finalize(self):
        ...
```

Potential backend implementations:

```text
RuleBasedOSD
ModelBasedOSD
...
```

At minimum, preserve a simple dummy/rule-based implementation for tests.

Do not make the entire project dependent on one external OSD library.

---

# 29. Rule-Based / Oracle Detector

Create a lightweight test/reference implementation if useful.

Possible implementations:

### Oracle OSD

Uses Phase 3 ground-truth metadata.

Useful for testing Phase 6 routing without depending on model errors.

### Dummy OSD

Returns controlled outputs for unit tests.

These must be explicitly labeled:

```text
ORACLE
DUMMY
NOT A REAL OSD MODEL
```

Do not include oracle performance as actual model performance.

---

# 30. Configuration

Create:

```text
configs/overlap_detection.yaml
```

The configuration should expose at least:

```yaml
experiment:
  name: overlap_detection
  seed: 42

data:
  manifest: ...

model:
  backend: ...
  name: ...

osd:
  frame_ms: ...
  threshold: ...
  smoothing:
    enabled: false

evaluation:
  metrics:
    - precision
    - recall
    - f1
    - detection_latency
```

Use the project's existing configuration system.

---

# 31. Script

Create:

```text
scripts/evaluate_osd.py
```

or an equivalent name consistent with the repository.

The script should:

1. load configuration
2. load dataset manifest
3. load OSD backend
4. generate/obtain ground-truth labels
5. stream audio through the detector
6. collect timestamped predictions
7. align predictions with ground truth
8. calculate metrics
9. save per-example results
10. save aggregate results
11. generate plots/reports
12. record experiment metadata

---

# 32. Notebook

Create:

```text
notebooks/06_overlap_detection.ipynb
```

Suggested workflow:

```text
1. Environment check
2. Load configuration
3. Inspect Phase 3 metadata
4. Construct OSD ground truth
5. Load candidate OSD backend
6. Run streaming OSD on examples
7. Visualize predictions vs ground truth
8. Measure frame-level metrics
9. Measure event-level metrics
10. Measure detection latency
11. Inspect false positives/negatives
12. Evaluate threshold if applicable
13. Save results
```

Keep actual implementation in `src/`.

---

# 33. Visualizations

At minimum create useful diagnostics such as:

### Ground truth vs prediction timeline

```text
Ground truth:
████████        overlap

Prediction:
    ███████     overlap
```

### Confusion matrix

For multi-class OSD where applicable.

### Precision/Recall/F1 vs threshold

If probability thresholding is supported.

### Detection latency distribution

If enough events exist.

Do not generate misleading plots.

---

# 34. Results Storage

Store per-frame or per-segment results where practical.

For example:

```json
{
  "mixture_id": "...",
  "timestamp": 2.40,
  "ground_truth": "OVERLAP",
  "overlap_probability": 0.81,
  "prediction": "OVERLAP"
}
```

And per-event information such as:

```json
{
  "mixture_id": "...",
  "true_start": 2.1,
  "true_end": 4.3,
  "detected_start": 2.28,
  "detected_end": 4.1,
  "detection_delay": 0.18
}
```

Adapt the schema to the implementation.

---

# 35. Tests

Extend:

```text
tests/test_overlap.py
```

and any other relevant test modules.

Test at least:

### Ground-truth generation

* no active source
* one active source
* two active sources
* exact boundary conditions
* touching intervals
* nested intervals

### Temporal alignment

* equal frame rates
* different prediction/frame rates
* timestamp offsets

### OSD interface

* `process()`
* state persistence
* `finalize()`

### Metrics

* perfect detector
* all-negative detector
* all-positive detector
* missed overlap
* false positive overlap
* threshold behavior

Tests must not require a large pretrained model.

---

# 36. Real Model Integration Test

If a real pretrained OSD model is used, create a separate integration test.

It may require:

* model download
* GPU
* internet
* additional dependencies

Keep it separate from lightweight unit tests.

Document exactly how to run it.

---

# 37. Kaggle Compatibility

The notebook must be runnable in Kaggle.

Support:

* attached Phase 3 generated dataset
* Hugging Face model loading where supported
* local/configured model paths
* configurable output paths

Do not require manual code modification for paths.

Do not store model weights in Git.

---

# 38. Hugging Face Support

If the selected OSD model is hosted on Hugging Face:

* use the existing model backend abstraction
* document the model identifier
* document revision/version where available
* avoid hidden downloads
* make model acquisition configurable

Do not assume that every Hugging Face model is streamable.

Verify its actual inference behavior.

---

# 39. Performance Measurements

Measure the computational cost of OSD.

At minimum where practical:

* processing time per chunk/frame
* real-time factor
* average processing latency
* peak memory/GPU memory where available

The later adaptive architecture needs to know whether OSD itself introduces unacceptable overhead.

Do not optimize prematurely.

Measure first.

---

# 40. Research Questions

Phase 5 should provide evidence for:

### RQ1

Can overlap be detected reliably in a streaming setting?

### RQ2

What precision/recall/F1 does the selected OSD approach achieve?

### RQ3

What detection latency does it introduce?

### RQ4

How does performance change with overlap ratio and relative speaker level?

### RQ5

Is the computational cost compatible with a real-time adaptive ASR pipeline?

### RQ6

What failure cases remain before OSD can safely control adaptive routing?

Do not answer these questions without actual measurements.

---

# 41. Research Decision: OSD Sufficiency

At the end of Phase 5, assess whether the selected OSD approach is technically sufficient to proceed to adaptive routing.

Do not use a subjective "good/bad" label.

Instead report concrete evidence:

```text
F1
precision
recall
false-positive rate
false-negative rate
detection latency
RTF
memory
failure cases
```

The decision to proceed should be based on documented measurements and limitations.

Do not optimize the detector merely to meet a predetermined threshold.

---

# 42. Documentation

Update:

```text
README.md
docs/architecture.md
docs/experiments.md
docs/datasets.md
```

Document:

* OSD definition
* VAD vs OSD
* selected model/method
* candidate-selection rationale
* streaming characteristics
* frame/chunk configuration
* ground-truth generation
* metrics
* latency measurement
* thresholding
* limitations
* synthetic-vs-real evaluation distinction
* commands
* outputs

---

# 43. Phase Report

Create:

```text
docs/phase_reports/phase_5_report.md
```

Include:

## Objective

What was evaluated.

## Candidate Methods

What candidate OSD approaches were considered.

## Selected Method

Which implementation was actually used and why, based on technical characteristics.

## Dataset

Exact dataset/configuration.

## Ground Truth

How overlap labels were generated.

## Streaming Setup

Frame size, hop, context, state handling, and output timing.

## Results

Actual:

* precision
* recall
* F1
* confusion matrix where applicable
* detection latency
* RTF
* computational cost

## Failure Analysis

Examples of false positives/negatives and timing errors.

## Limitations

Especially synthetic-data limitations and model streaming limitations.

## Phase 6 Readiness

Explain whether the OSD output is technically usable as an input signal for adaptive routing.

Do not implement routing.

---

# 44. Completion Criteria

Phase 5 is complete only when:

* [ ] A streaming OSD abstraction exists.
* [ ] OSD is independent from WhisperRT.
* [ ] OSD is independent from a specific VAD implementation.
* [ ] Synthetic overlap ground truth is generated independently.
* [ ] Temporal alignment is explicitly defined.
* [ ] A real OSD candidate is evaluated.
* [ ] Streaming behavior is verified.
* [ ] Overlap precision/recall/F1 are measured.
* [ ] Detection latency is measured.
* [ ] Computational overhead is measured where practical.
* [ ] Thresholding is configurable if applicable.
* [ ] Per-example predictions are stored.
* [ ] False positives/negatives can be inspected.
* [ ] Tests cover ground truth, alignment, interface, and metrics.
* [ ] Real-model integration test is separated from lightweight tests.
* [ ] Kaggle workflow is documented.
* [ ] Hugging Face model loading works where applicable.
* [ ] Existing Phase 1–4 functionality remains intact.
* [ ] `scripts/smoke_test.py` remains functional.
* [ ] `notebooks/06_overlap_detection.ipynb` is created.
* [ ] README/docs are updated.
* [ ] Phase 5 report is created.
* [ ] No adaptive routing is implemented.
* [ ] No diarization is implemented.
* [ ] No speaker identification is implemented.
* [ ] No source separation is implemented.
* [ ] No multi-talker ASR is implemented.
* [ ] No overlap-specific ASR optimization is implemented.

---

# 45. Final Execution Rule

Do not implement Phase 6 during this task.

At the end of Phase 5, stop and report:

1. files created/modified
2. OSD architecture
3. candidate methods considered
4. selected OSD method
5. exact model/version/repository where applicable
6. dataset and split
7. ground-truth methodology
8. streaming configuration
9. commands executed
10. measured OSD metrics
11. detection latency
12. computational cost
13. failure examples
14. tests
15. smoke-test status
16. notebook status
17. limitations
18. unresolved research questions
19. whether the OSD interface is ready to feed Phase 6
20. exact recommended next step

The next phase will consume the OSD output to build the **Adaptive Streaming ASR Pipeline**:

```text
Audio Stream
      ↓
Streaming VAD
      ↓
Streaming OSD
      ↓
┌─────┴────────┐
│              │
Single        Overlap
│              │
WhisperRT     overlap-aware /
              multi-talker path
│              │
└─────┬────────┘
      ↓
Incremental Transcript
```

Do not implement this routing during Phase 5.
