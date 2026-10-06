# PHASE 7 — Streaming Overlap-Aware / Multi-Talker ASR

## Objective

Implement the first **actual overlap-aware / multi-talker ASR branch** for the English Streaming ASR project.

The goal of this phase is to move beyond the Phase 6 routing prototype and introduce a real ASR component capable of handling overlapping speakers more appropriately than ordinary single-stream WhisperRT decoding.

The system should remain **streaming-first** and should operate on continuous audio chunks rather than treating the problem as:

```text
offline audio → artificially split into chunks → offline ASR
```

The main research question for this phase is:

> Can an overlap-aware or multi-talker ASR model be integrated into the existing streaming VAD + OSD + adaptive routing architecture while preserving incremental operation and providing a measurable improvement over the ordinary WhisperRT baseline on overlapping speech?

Do not assume that a particular model is suitable. First investigate candidate models and select one based on actual technical compatibility with this project's requirements.

---

# 1. Scope

Implement:

1. Literature/model investigation for overlap-aware and multi-talker ASR.
2. Selection of one primary candidate model.
3. A model adapter/backend abstraction.
4. Streaming or low-latency inference for the selected model.
5. Integration with the existing adaptive pipeline.
6. Multi-speaker/overlap-aware output representation.
7. Evaluation on synthetic overlap data.
8. Comparison against the existing WhisperRT baseline.
9. Comparison against Phase 6 adaptive routing.
10. Appropriate multi-talker ASR metrics.
11. Latency and computational measurements.
12. Robust handling of transitions between normal and overlap branches.
13. Tests.
14. Configuration.
15. Kaggle-compatible demonstration notebook.
16. Documentation and phase report.

Preserve all functionality from Phases 1–6.

---

# 2. Explicitly Out of Scope

Do NOT implement the following in this phase unless they are absolutely required by the selected model's inference mechanism:

* speaker enrollment
* known-speaker identification
* speaker verification
* full diarization
* speaker clustering
* source separation as an independent research objective
* large-scale model pretraining
* reinforcement learning
* RLHF
* PEFT/fine-tuning as the primary contribution
* final thesis-scale ablation study
* production deployment
* microphone-specific optimization
* multilingual ASR

The target language remains:

**English**

The system should recognize multiple simultaneously active speakers, but it does not need to determine the real-world identity of those speakers.

For example:

```text
Speaker A
Speaker B
```

is sufficient.

You do not need:

```text
Alice
Bob
```

---

# 3. FIRST STEP — Investigate Candidate Models

Before implementing the overlap-aware branch, inspect current research and available implementations.

The investigation should focus on models/research directions such as:

* multi-talker ASR
* overlap-aware ASR
* streaming multi-talker ASR
* continuous multi-talker ASR
* target-speaker-independent ASR
* serialized output training
* token-level speaker-attributed ASR
* multi-talker end-to-end ASR
* streaming conversational ASR
* recent Whisper-based multi-talker approaches
* SURT / serialized output approaches
* t-SOT / token-level serialized output approaches
* recent models specifically designed for overlapping speech

Potential candidates may include approaches such as:

* SURT
* t-SOT
* Whisper-based multi-talker approaches
* recent streaming multi-talker architectures
* other publicly available models with reproducible inference

Do NOT assume that these are automatically suitable.

For each candidate investigate:

* paper
* publication year
* official implementation
* GitHub repository
* Hugging Face availability
* pretrained checkpoint availability
* model size
* input format
* output format
* whether it supports multiple simultaneous speakers
* whether it is genuinely streaming
* whether it is chunk-based/low-latency
* whether it requires full utterance context
* whether it requires speaker embeddings
* whether it performs separation internally
* whether it supports English
* dataset used for training/evaluation
* whether inference is reproducible on Kaggle
* GPU requirements
* license
* installation complexity
* compatibility with Python/PyTorch environment already established

Create a short comparison table in:

```text
docs/multitalker_candidates.md
```

Use factual descriptions.

Do not select a model merely because it has a high reported benchmark score.

The primary selection criteria should be:

1. overlap/multi-talker capability
2. streaming or low-latency compatibility
3. publicly available implementation/checkpoint
4. reproducible inference
5. English support
6. reasonable compute requirements
7. compatibility with the existing project
8. ability to produce outputs that can be evaluated against the synthetic overlap ground truth

If no candidate provides genuine streaming inference, explicitly document this limitation.

In that case, select the most defensible low-latency/online-compatible candidate and clearly distinguish:

```text
TRUE STREAMING
```

from:

```text
LOW-LATENCY WINDOWED
```

and:

```text
OFFLINE
```

Do not describe an offline model as streaming.

---

# 4. Model Selection Decision

After investigating candidates, select exactly one primary implementation candidate for Phase 7.

Document:

```text
Selected model:
Repository:
Paper:
Checkpoint:
Model size:
Input:
Output:
Streaming capability:
Overlap capability:
Expected latency:
GPU requirements:
License:
Reason for selection:
Known limitations:
```

If the best candidate cannot satisfy all requirements, explicitly state which requirement is relaxed and why.

Do not modify the architecture simply to make an unsuitable model appear streaming-compatible.

---

# 5. Model Adapter

Create:

```text
src/streaming_asr/models/multitalker.py
```

with an abstraction similar to:

```python
class StreamingMultiTalkerASR:
    def process(self, audio_chunk):
        ...

    def finalize(self):
        ...
```

The exact interface may be extended if the selected model requires additional state.

The adapter should hide model-specific implementation details from the rest of the project.

The rest of the pipeline should not directly depend on the selected model's internal API.

---

# 6. Output Representation

The overlap-aware branch must produce a structured output.

Do NOT return only a plain string.

At minimum, represent:

```text
timestamp
speaker/stream identifier
text
confidence if available
start time if available
end time if available
final/partial status
```

For example:

```python
{
    "timestamp": ...,
    "streams": [
        {
            "stream_id": 0,
            "text": "...",
            "start": ...,
            "end": ...,
            "is_final": ...
        },
        {
            "stream_id": 1,
            "text": "...",
            "start": ...,
            "end": ...,
            "is_final": ...
        }
    ]
}
```

The exact schema should follow the selected model's actual output semantics.

Do not fabricate speaker attribution if the model does not provide it.

If the model only provides multiple transcription streams without reliable speaker identity, document that explicitly.

---

# 7. Streaming State

The overlap-aware model must maintain state across audio chunks where the model architecture supports stateful processing.

Investigate and document:

* recurrent state
* KV cache
* encoder state
* decoder state
* attention context
* overlapping input windows
* look-ahead
* context retention
* output stabilization
* partial hypothesis revision

Do not reset the model on every chunk unless the model explicitly requires independent chunk processing.

The system should distinguish:

```text
model state
audio buffer state
partial hypothesis state
final hypothesis state
```

---

# 8. Chunking and Context

Implement configurable streaming parameters.

Example:

```yaml
chunk_duration_ms:
context_duration_ms:
lookahead_ms:
hop_duration_ms:
```

Do not hard-code these values.

The implementation must support experimentation with different settings.

If the selected model has its own recommended streaming configuration, document the source of those recommendations.

If the model is not genuinely streaming, implement the smallest defensible rolling-window strategy and explicitly label it:

```text
LOW-LATENCY WINDOWED INFERENCE
```

rather than streaming.

---

# 9. Incremental Output

The system must avoid repeatedly emitting the same text.

For example, if the model produces:

```text
partial 1:
hello

partial 2:
hello how

partial 3:
hello how are

partial 4:
hello how are you
```

the final pipeline must not treat these as four independent transcripts.

Implement an appropriate hypothesis-management mechanism.

The mechanism should handle:

* partial hypotheses
* revised hypotheses
* finalized tokens
* duplicated tokens
* token overlap
* stream-specific output
* end-of-segment finalization

If the selected model has an existing streaming output mechanism, use it rather than implementing an unnecessary second decoding algorithm.

---

# 10. Integration with Phase 6 Adaptive Pipeline

The architecture should become:

```text
Audio Stream
      ↓
Streaming Audio Buffer
      ↓
Streaming VAD
      ↓
Streaming OSD
      ↓
Adaptive Router
      ↓
 ┌───────────────┴────────────────┐
 │                                │
SINGLE SPEAKER                  OVERLAP
 │                                │
WhisperRT                 Multi-Talker ASR
Streaming                      Branch
 │                                │
 └───────────────┬────────────────┘
                 ↓
        Incremental Transcript
                 ↓
             Evaluation
```

The normal branch remains:

```text
MLSpeech/WhisperRT-Streaming
```

The overlap branch becomes the actual Phase 7 multi-talker model.

Do not replace WhisperRT.

The project must retain both branches.

---

# 11. Branch Transition Handling

This is a critical part of the phase.

The router may transition:

```text
SINGLE_SPEAKER → OVERLAP
```

or:

```text
OVERLAP → SINGLE_SPEAKER
```

The implementation must investigate how to handle these transitions without losing audio or generating duplicate output.

Consider:

* pre-roll audio
* overlap onset delay
* OSD detection latency
* buffering
* branch warm-up
* branch state initialization
* branch finalization
* transcript synchronization
* duplicate hypothesis suppression
* transition timestamps

If OSD detects overlap after the overlap has already started, the overlap branch may need access to a small amount of preceding audio.

Implement a configurable:

```text
overlap_pre_roll_ms
```

or equivalent mechanism if technically appropriate.

Document its effect.

---

# 12. Branch Warm-Up

If the multi-talker model requires initialization time, measure it.

The system should distinguish:

```text
OSD detection time
→ router decision time
→ overlap branch activation time
→ first overlap-aware output time
```

Do not hide initialization latency inside generic inference time.

---

# 13. Baseline Conditions

Phase 7 must compare at least these conditions.

### Condition A — Clean WhisperRT

```text
Audio
→ WhisperRT
```

Use the clean/single-speaker evaluation condition from earlier phases.

---

### Condition B — WhisperRT on Overlap

```text
Audio
→ WhisperRT
```

Use the Phase 4 implementation.

This is the primary overlap baseline.

---

### Condition C — Multi-Talker ASR Without Adaptive Routing

If technically possible:

```text
Audio
→ Multi-Talker ASR
```

Run the selected model directly on the overlap test set.

This isolates the effect of the multi-talker model from the routing system.

---

### Condition D — Adaptive Routing + Multi-Talker ASR

```text
Audio
→ VAD
→ OSD
→ Router
→ WhisperRT / Multi-Talker ASR
```

This is the main Phase 7 integrated condition.

---

### Condition E — Oracle Routing + Multi-Talker ASR

Where technically possible:

```text
Audio
→ Ground-Truth Overlap State
→ Router
→ WhisperRT / Multi-Talker ASR
```

Clearly label this:

```text
ORACLE ROUTING — NOT A DEPLOYABLE SYSTEM
```

This measures the upper bound of the routing mechanism independently of OSD errors.

---

# 14. Evaluation Dataset

Primary evaluation data:

```text
Phase 3 Synthetic Overlap Dataset
```

Use the existing controlled overlap conditions.

At minimum evaluate:

```text
no overlap / control
low overlap
medium overlap
high overlap
```

where those conditions exist in the generated dataset.

Do not regenerate the dataset unnecessarily.

Reuse the existing manifests and metadata.

Maintain speaker-disjoint evaluation splits.

---

# 15. Evaluation Metrics

## 15.1 ASR Metrics

For ordinary single-speaker data:

```text
WER
CER if useful
```

For multi-talker output, determine the correct metric based on the selected model's output representation.

Potential metrics include:

* cpWER
* permutation-invariant WER
* speaker-attributed WER
* stream-wise WER
* utterance-level WER
* token-level metrics

Do not force ordinary single-reference WER onto a two-speaker overlap example.

If the output format cannot support a valid metric, record:

```text
NOT MEASURED
```

and explain why.

Do not invent an evaluation method simply to produce a number.

---

# 16. Streaming Metrics

Measure:

```text
Real-Time Factor
First Output Latency
Average Output Latency
End-of-Utterance Latency
Chunk Processing Time
Model Inference Time
Routing Overhead
Memory Usage
GPU Memory Usage
```

Where appropriate also measure:

```text
Time to First Correct Token
Token Latency
Partial Hypothesis Stability
Revision Rate
```

Only report metrics that can actually be measured.

---

# 17. Overlap-Specific Analysis

For each overlap regime investigate:

* recognition degradation
* missed speaker
* speaker dominance
* missing tokens
* cross-speaker token ordering
* repetitions
* hallucinations
* partial-output instability
* delayed recognition
* failure to activate overlap branch
* incorrect return to single-speaker branch
* branch transition errors

Produce qualitative examples.

Do not cherry-pick examples without documenting the selection method.

---

# 18. Routing vs Recognition Analysis

Separate the two sources of errors:

### Routing errors

Examples:

```text
OVERLAP detected as SINGLE
SINGLE detected as OVERLAP
late overlap detection
late return to SINGLE
```

### Recognition errors

Examples:

```text
correct routing + bad transcription
correct overlap detection + missed speaker
correct routing + unstable multi-talker decoding
```

This distinction is important for the thesis.

The experiment should help answer:

> Is recognition failing because the router selected the wrong branch, or because the selected overlap-aware recognizer itself fails?

---

# 19. Oracle vs Predicted OSD

Compare:

```text
Oracle OSD
```

against:

```text
Predicted OSD
```

If the results differ substantially, investigate whether the main bottleneck is:

```text
OSD
```

rather than:

```text
Multi-Talker ASR
```

Do not conclude causality from a small experiment without supporting evidence.

---

# 20. Adaptive Routing Ablation

Where feasible compare:

```text
Always WhisperRT
```

with:

```text
Always Multi-Talker
```

and:

```text
Adaptive Routing
```

This helps separate:

* model capability
* routing capability
* computational cost

from each other.

If the multi-talker model is too expensive to run continuously, measure the computational benefit of activating it only during detected overlap.

---

# 21. Compute Analysis

Record:

```text
CPU
GPU
GPU memory
model parameter count if available
model loading time
warm-up time
average inference time
peak memory
RTF
```

Compare:

```text
Always Multi-Talker
```

against:

```text
Adaptive Multi-Talker
```

if the experiment is feasible.

The purpose is not to claim deployment readiness, but to quantify the cost of adaptive activation.

---

# 22. Configuration

Create or update:

```text
configs/multitalker.yaml
```

and, if necessary:

```text
configs/adaptive.yaml
```

Configuration should include:

```yaml
model:
  name:
  checkpoint:
  device:
  dtype:

streaming:
  chunk_duration_ms:
  context_duration_ms:
  lookahead_ms:
  hop_duration_ms:

overlap:
  pre_roll_ms:
  threshold:
  stabilization_ms:

evaluation:
  dataset:
  split:
  metrics:
```

Do not hard-code experiment settings in Python.

---

# 23. Implementation Files

Create or update only the files required for this phase.

Likely files include:

```text
src/streaming_asr/models/multitalker.py
src/streaming_asr/pipeline/overlap.py
src/streaming_asr/pipeline/adaptive.py
src/streaming_asr/evaluation/evaluator.py
src/streaming_asr/metrics/asr.py
src/streaming_asr/metrics/streaming.py
src/streaming_asr/metrics/overlap.py
src/streaming_asr/utils/config.py

scripts/run_multitalker.py
scripts/run_adaptive.py
scripts/benchmark.py

configs/multitalker.yaml
configs/adaptive.yaml

notebooks/08_multitalker.ipynb

tests/test_multitalker.py
tests/test_pipeline.py
tests/test_streaming.py
tests/test_metrics.py

docs/multitalker_candidates.md
docs/phase_reports/phase_7_report.md
```

Do not create unnecessary duplicate abstractions.

Reuse existing infrastructure from earlier phases wherever appropriate.

---

# 24. Notebook

Create:

```text
notebooks/08_multitalker.ipynb
```

The notebook must be executable on Kaggle.

It should demonstrate:

1. environment verification
2. model loading
3. small overlap sample
4. streaming/chunked processing
5. multi-talker output
6. comparison with WhisperRT
7. latency measurement
8. evaluation
9. visualization
10. limitations

Keep the notebook small enough to run on a practical Kaggle GPU.

Do not download the entire dataset.

Use a small configurable subset.

---

# 25. Tests

Add unit tests for:

### Model adapter

* initialization
* input validation
* output schema
* state handling
* finalize behavior

### Streaming

* chunk processing
* state persistence
* partial output
* final output
* duplicate prevention

### Multi-talker output

* multiple streams
* empty streams
* partial/final hypotheses
* timestamps
* malformed model output

### Adaptive routing

* SINGLE → OVERLAP
* OVERLAP → SINGLE
* NO_SPEECH transitions
* hysteresis
* pre-roll
* branch initialization
* branch finalization
* duplicate prevention

### Metrics

* single-speaker WER
* multi-talker metric where applicable
* invalid-reference handling
* latency calculation

Use mocks/dummy models for unit tests.

Do not require downloading a large checkpoint for normal unit tests.

---

# 26. Real Integration Test

Create a separate optional integration test that:

1. loads the actual selected model,
2. loads a very small overlap example,
3. processes it incrementally,
4. produces structured output,
5. verifies that inference completes,
6. measures basic latency.

This test should be clearly separated from the normal unit-test suite.

Example:

```text
pytest tests/
```

must remain lightweight.

An explicit real-model test may require:

```text
pytest tests/test_multitalker_integration.py
```

or an equivalent command.

---

# 27. Model Download Policy

Never commit:

* model weights
* checkpoints
* large datasets
* generated audio collections
* large prediction files

The repository should contain only:

* configuration
* code
* small examples where appropriate
* manifests
* documentation
* reproducibility metadata

Models should be downloaded through:

* Hugging Face
* Kaggle
* configured local paths

Use the project's existing backend abstraction.

---

# 28. Hugging Face and Kaggle Compatibility

The model implementation must work through the project's existing model-loading abstraction.

Support:

```text
Hugging Face
Kaggle
Local
```

where technically applicable.

Do not create a separate model-loading mechanism that bypasses the project's backend abstraction unless the selected model genuinely requires it.

If Kaggle cannot reproduce the model because of dependency, hardware, licensing, or checkpoint limitations, document the exact limitation.

---

# 29. Reproducibility

Every experiment must record:

```text
experiment ID
timestamp
git commit if available
random seed
model
checkpoint
dataset
dataset split
configuration
hardware
software environment
metrics
runtime
output path
```

Use the existing experiment-tracking infrastructure.

Do not manually type final metrics into reports.

Generate reports from experiment results where practical.

---

# 30. Failure Handling

The system must fail clearly when:

* checkpoint is unavailable
* model dependencies are missing
* GPU memory is insufficient
* model does not support the requested streaming mode
* output format is incompatible
* sample rate is unsupported
* input shape is invalid

Do not silently fall back to ordinary Whisper.

A fallback that changes the scientific condition invalidates the experiment.

If a fallback is useful for development, label it explicitly:

```text
DEVELOPMENT FALLBACK — NOT USED FOR RESULTS
```

---

# 31. Research Integrity

This phase is research-oriented.

Never fabricate:

* model capabilities
* streaming support
* benchmark scores
* WER
* latency
* memory usage
* dataset statistics
* citations
* experimental improvements

If something has not been measured:

```text
NOT MEASURED
```

If something is not yet decided:

```text
RESEARCH DECISION PENDING
```

If a candidate model cannot satisfy a requirement:

```text
LIMITATION
```

Do not describe low-latency windowed inference as true streaming.

Do not describe multiple transcription streams as speaker identification.

Do not claim the adaptive architecture improves ASR unless the measured results support that statement.

---

# 32. Important Scientific Distinction

Maintain these three concepts separately:

### Overlap Detection

```text
Is more than one speaker active?
```

### Multi-Talker ASR

```text
What is each active speaker saying?
```

### Diarization

```text
Who is speaking when?
```

Phase 7 addresses the second problem.

It does not turn the project into a diarization system.

---

# 33. Experiment Matrix

Start with a small, reproducible matrix.

At minimum:

| Condition                | VAD          | OSD    | Router | ASR                      |
| ------------------------ | ------------ | ------ | ------ | ------------------------ |
| Clean baseline           | No           | No     | No     | WhisperRT                |
| Overlap baseline         | No           | No     | No     | WhisperRT                |
| Multi-talker             | No           | No     | No     | Multi-Talker             |
| Adaptive + predicted OSD | Yes          | Yes    | Yes    | WhisperRT + Multi-Talker |
| Adaptive + oracle OSD    | Yes/optional | Oracle | Yes    | WhisperRT + Multi-Talker |

Evaluate on a manageable subset first.

Only expand the experiment after the entire pipeline works.

---

# 34. Visualization

Add useful visualizations such as:

### Routing timeline

```text
time →
────────────────────────────
SINGLE | SINGLE | OVERLAP | OVERLAP | SINGLE
```

### Ground truth vs OSD

```text
Ground Truth:  SINGLE SINGLE OVERLAP OVERLAP SINGLE
Predicted:     SINGLE SINGLE SINGLE OVERLAP SINGLE
```

### Recognition comparison

Compare:

```text
WhisperRT
Multi-Talker ASR
Adaptive
Oracle Adaptive
```

using appropriate metrics.

### Latency

Compare:

```text
WhisperRT
Multi-Talker
Adaptive
```

where meaningful.

Do not create plots for metrics that were not validly measured.

---

# 35. README Updates

Update the main:

```text
README.md
```

with:

* Phase 7 status
* selected multi-talker model
* architecture
* installation requirements
* inference instructions
* streaming behavior
* output format
* evaluation procedure
* limitations
* reproducibility instructions

Also update:

```text
docs/architecture.md
docs/datasets.md
docs/experiments.md
docs/thesis_mapping.md
```

where necessary.

---

# 36. Phase Report

Create:

```text
docs/phase_reports/phase_7_report.md
```

The report must contain:

## 1. Objective

What Phase 7 attempted to implement.

## 2. Candidate Models

Short comparison of investigated models.

## 3. Selected Model

Exact model and justification.

## 4. Architecture

Show:

```text
Audio
 ↓
VAD
 ↓
OSD
 ↓
Router
 ├── WhisperRT
 └── Multi-Talker ASR
 ↓
Output
```

## 5. Streaming Behavior

Explain whether the selected model is:

```text
TRUE STREAMING
```

or:

```text
LOW-LATENCY WINDOWED
```

and why.

## 6. Dataset

Describe the exact Phase 3 subset used.

## 7. Experiments

List all executed conditions.

## 8. Metrics

List measured metrics.

## 9. Results

Report actual measurements only.

## 10. Latency

Report actual latency/RTF measurements.

## 11. Routing

Analyze oracle vs predicted OSD.

## 12. Failure Analysis

Document representative failure categories.

## 13. Resource Usage

Report hardware and memory where measured.

## 14. Tests

Report test results.

## 15. Limitations

Be explicit.

## 16. Phase 8 Requirements

Identify exactly what must be addressed next.

---

# 37. Completion Criteria

Phase 7 is complete only if:

* candidate models were investigated
* one primary model was selected
* the selection is documented
* the model adapter works
* streaming/low-latency behavior is explicitly characterized
* the model processes real overlap audio
* structured multi-talker output is produced
* WhisperRT remains functional
* Phase 6 adaptive routing remains functional
* overlap branch is connected to the router
* branch transitions are handled
* baseline experiments run
* multi-talker experiments run
* adaptive experiments run where technically feasible
* valid ASR metrics are calculated
* streaming metrics are calculated
* routing metrics remain available
* tests pass
* notebook runs
* Kaggle/HF loading works where supported
* README is updated
* architecture documentation is updated
* phase report is written
* `scripts/smoke_test.py` still works
* previous phase functionality is not broken

---

# 38. Final Agent Response

After implementation, stop.

Do not automatically begin Phase 8.

Report:

1. candidate models investigated
2. selected model
3. paper/repository/checkpoint
4. why it was selected
5. streaming capability classification
6. model architecture/interface
7. output representation
8. files created
9. files modified
10. adaptive architecture changes
11. branch-transition strategy
12. datasets used
13. experiment conditions
14. exact commands executed
15. ASR metrics
16. multi-talker metrics
17. latency/RTF
18. GPU/CPU/memory usage
19. routing results
20. oracle vs predicted OSD results
21. failure examples
22. tests
23. smoke-test status
24. notebook status
25. README/documentation status
26. limitations
27. unresolved technical issues
28. exact recommended objectives for Phase 8

Do not claim success merely because the code runs.

The final report must distinguish:

```text
IMPLEMENTED
```

from:

```text
MEASURED
```

from:

```text
NOT MEASURED
```

from:

```text
LIMITATION
```

and:

```text
FUTURE WORK
```

The central scientific objective of this phase is to establish a reproducible, measurable **overlap-aware/multi-talker ASR branch** that can genuinely participate in the existing streaming adaptive architecture.
