You are the primary coding agent for a Master's thesis research project titled:

**English Streaming Automatic Speech Recognition with Overlapped Voices**

Your task is to incrementally implement a reproducible research codebase for real-time English streaming ASR under overlapping speech.

The project must be implemented phase-by-phase. Do NOT attempt to implement all research components at once.

The repository must remain runnable after every phase.

---

# 1. Core Research Scope

The project focuses on:

* English ASR
* continuous audio streams
* low-latency / real-time inference
* streaming VAD
* streaming ASR
* overlapped speech
* overlap speech detection
* multi-talker speech recognition
* adaptive routing based on whether overlap is detected
* latency/accuracy trade-offs

The project currently excludes:

* speaker diarization
* speaker identification
* speaker enrollment
* face/video information
* microphone-array research
* beamforming research
* multilingual ASR
* large-scale pretraining

Do not introduce these components unless a later research phase explicitly requires them.

---

# 2. Primary ASR Model

The primary candidate ASR model is:

`MLSpeech/WhisperRT-Streaming`

Use the Hugging Face implementation/checkpoint where appropriate.

Do not replace WhisperRT with another ASR architecture without explicit justification.

Alternative streaming ASR models may be implemented later only as research baselines if required.

---

# 3. Streaming-First Principle

The project must be streaming-first.

Do NOT implement:

offline ASR
→ artificial chunking
→ call it streaming.

The core pipeline must process audio incrementally.

The streaming abstraction must support:

* fixed-size audio frames/chunks
* stateful processing
* incremental hypotheses
* partial results
* final results
* configurable chunk size
* configurable buffering
* latency measurement

All components that claim to be streaming must operate incrementally.

---

# 4. Streaming VAD

VAD must support true online/streaming processing.

The VAD abstraction should support:

```python
vad.process(audio_chunk)
```

and maintain state between chunks.

Do not assume an offline VAD is equivalent to streaming VAD.

The initial implementation should support interchangeable VAD backends.

Potential backends may include:

* WebRTC VAD
* Silero VAD
* TEN VAD
* other research-backed streaming VADs

Do not hard-code a final VAD choice before evaluation.

---

# 5. Overlap Detection

VAD and overlap detection are different tasks.

VAD answers:

* speech
* non-speech

Overlap detection answers whether active speech contains multiple simultaneous speakers.

The overlap detector abstraction must therefore support:

```text
NO_SPEECH
SINGLE_SPEAKER
OVERLAP
```

or an equivalent probability representation.

The first implementation must create the abstraction before selecting/training the final neural model.

The pipeline must be runnable even if the real OSD model has not yet been trained.

A dummy/oracle/rule-based implementation may temporarily be used for integration testing.

---

# 6. Adaptive Pipeline

The adaptive pipeline has a deliberately narrow scope.

It only needs to:

1. receive streaming audio
2. perform streaming VAD
3. determine whether overlap is present
4. route the current audio segment to the appropriate processing path

It does NOT need to:

* identify speakers
* assign persistent speaker identities
* perform diarization
* determine who is speaking
* perform speaker enrollment

Conceptually:

```text
Audio Stream
    ↓
Streaming VAD
    ↓
Overlap Detector
    ↓
┌───────────────┬────────────────┐
│ no overlap    │ overlap        │
↓               ↓
WhisperRT       overlap-aware /
                multi-talker path
```

The overlap path may initially be a placeholder or baseline.

---

# 7. Progressive Research Design

Every phase must produce a meaningful, independently testable result.

The project must follow this progression:

## Phase 0

Environment and project infrastructure

## Phase 1

WhisperRT streaming baseline

## Phase 2

Streaming VAD integration

## Phase 3

Synthetic overlap dataset generation

## Phase 4

WhisperRT performance under overlap

## Phase 5

Streaming Overlap Speech Detection abstraction + implementation

## Phase 6

Adaptive routing

## Phase 7

Multi-talker / overlap-aware recognition

## Phase 8

Training / fine-tuning / PEFT where justified

## Phase 9

Full evaluation and ablation

## Phase 10

Reproducibility package and thesis experiment preparation

Do not skip foundational phases.

---

# 8. Dataset Strategy

The intended progression is:

### Training / development

LibriSpeech

→ synthetic overlap generation

### Controlled multi-talker evaluation

LibriSpeechMix

### Continuous overlap evaluation

LibriCSS

### Optional external generalization

AMI / CHiME-6

Do not download or preprocess all datasets automatically.

Each dataset must have a dedicated loader/configuration.

The code must support both:

### Kaggle backend

Datasets are available under:

```text
/kaggle/input/
```

### Hugging Face backend

Datasets are loaded through Hugging Face APIs.

The backend must be configurable.

Example:

```yaml
data:
  backend: kaggle
```

or:

```yaml
data:
  backend: huggingface
```

Never hard-code a user's personal Kaggle path.

Use configurable paths.

---

# 9. Model Loading

The model loading system must support:

### Hugging Face

```python
from_pretrained(...)
```

### Local/Kaggle

```text
/kaggle/input/...
```

Model configuration must determine the backend.

Example:

```yaml
model:
  backend: huggingface
  name: MLSpeech/WhisperRT-Streaming
```

or:

```yaml
model:
  backend: kaggle
  path: /kaggle/input/...
```

Large model weights must never be committed to the repository.

---

# 10. Synthetic Overlap Dataset

The project must include a reusable synthetic overlap generator.

It must support at least:

* two speakers
* optional three speakers
* configurable overlap ratio
* configurable temporal offset
* configurable SNR
* configurable speaker duration
* deterministic random seed
* reproducible manifests

Example:

```text
Speaker A:
████████████████████

Speaker B:
        ███████████████

Mixture:
████████████████████
        ↑ overlap
```

Each generated sample must retain metadata describing:

* source utterance
* speaker/source ID
* start time
* end time
* overlap intervals
* overlap ratio
* number of speakers
* SNR
* sample rate
* transcript

Do not destroy source-level ground truth.

---

# 11. Repository Structure

Maintain the following structure:

```text
configs/
src/
scripts/
notebooks/
tests/
data/
checkpoints/
outputs/
docs/
```

Core implementation must live under:

```text
src/streaming_asr/
```

Notebooks must orchestrate the library.

Do not place the primary implementation inside notebooks.

---

# 12. Notebook Policy

Every major phase must have its own notebook.

Examples:

```text
notebooks/
├── 00_environment.ipynb
├── 01_dataset_validation.ipynb
├── 02_whisperrt_baseline.ipynb
├── 03_streaming_vad.ipynb
├── 04_overlap_generation.ipynb
├── 05_overlap_baseline.ipynb
├── 06_overlap_detection.ipynb
├── 07_adaptive_pipeline.ipynb
├── 08_multitalker.ipynb
├── 09_training.ipynb
└── 10_evaluation.ipynb
```

Each notebook must:

1. explain the purpose
2. load the appropriate configuration
3. verify the environment
4. execute the relevant pipeline
5. produce measurable results
6. save outputs
7. display key metrics
8. point to the next phase

Notebooks must not contain duplicated core logic.

---

# 13. Configuration

Every experiment must use a YAML configuration.

Do not hard-code experimental parameters in Python.

Configurations must include, where relevant:

* dataset
* model
* backend
* paths
* sample rate
* chunk size
* buffer size
* VAD parameters
* OSD parameters
* overlap ratio
* SNR
* batch size
* learning rate
* epochs
* seed
* output path
* metrics

Example:

```yaml
experiment:
  name: whisperrt_librispeech_baseline
  seed: 42

data:
  backend: kaggle
  dataset: librispeech
  path: /kaggle/input/...

model:
  backend: huggingface
  name: MLSpeech/WhisperRT-Streaming

audio:
  sample_rate: 16000
  chunk_ms: 320

vad:
  enabled: true

evaluation:
  metrics:
    - wer
    - rtf
    - first_token_latency
```

---

# 14. Logging and Experiment Tracking

Implement simple experiment tracking from the beginning.

Do not introduce a heavy external tracking platform unless explicitly requested.

Each experiment must record:

* experiment name
* timestamp
* configuration
* git/repository state if available
* random seed
* model
* dataset
* metrics
* runtime
* hardware information
* output paths

Save:

```text
outputs/
├── logs/
├── metrics/
├── predictions/
├── figures/
└── reports/
```

Prefer machine-readable formats such as JSON/CSV alongside human-readable summaries.

---

# 15. Required Tests

Maintain:

```text
tests/
├── test_audio.py
├── test_vad.py
├── test_streaming.py
├── test_overlap.py
├── test_pipeline.py
└── test_metrics.py
```

Also maintain:

```text
scripts/smoke_test.py
```

The smoke test must run the smallest practical end-to-end pipeline using a small audio sample.

It must verify:

```text
audio loading
→ streaming
→ VAD
→ ASR
→ overlap abstraction
→ pipeline
→ metrics
```

where implemented by the current phase.

The smoke test must remain runnable after future phases.

---

# 16. Backward Compatibility

Every phase must preserve previous functionality.

Before finishing a phase:

1. run unit tests
2. run smoke test
3. run the current phase notebook
4. verify previous notebooks still work where practical
5. verify configuration loading
6. verify output generation

Do not perform large refactors unless necessary.

If a refactor is necessary, document it.

---

# 17. Metrics

At minimum support:

### ASR

* WER
* CER when useful

### Streaming

* Real-Time Factor
* first-token latency
* average latency
* end-of-utterance latency
* chunk processing time

### OSD

* Precision
* Recall
* F1

### Multi-talker

Use appropriate metrics such as:

* cpWER
* permutation-invariant WER
* SA-WER only if applicable

Do not introduce speaker-diarization metrics unless the corresponding task is actually implemented.

---

# 18. Reproducibility

All experiments must support:

```text
random seed
configuration file
dataset version/path
model identifier
software versions
hardware information
```

The same configuration should reproduce the same experiment as closely as the underlying GPU/software permits.

---

# 19. Kaggle Requirements

The code must work in Kaggle notebooks.

Assume:

```text
/kaggle/input/
```

contains read-only datasets/models.

Use:

```text
/kaggle/working/
```

for generated artifacts.

Never attempt to write to:

```text
/kaggle/input/
```

Never assume internet access is enabled.

The README must document both:

### Kaggle Internet ON

For downloading models/datasets from Hugging Face.

### Kaggle Internet OFF

When datasets/models have already been attached as Kaggle inputs.

---

# 20. README Requirement

README.md is a first-class project artifact.

Keep it continuously updated.

The README must contain:

1. Project overview
2. Research objective
3. Current research scope
4. Architecture
5. Repository structure
6. Environment setup
7. Kaggle setup
8. Dataset setup
9. Model setup
10. Hugging Face loading
11. Kaggle loading
12. Phase-by-phase workflow
13. How to run every notebook
14. How to run every script
15. Configuration system
16. Testing
17. Smoke test
18. Training
19. Inference
20. Evaluation
21. Metrics
22. Output structure
23. Reproducibility
24. Troubleshooting
25. Current project status
26. Known limitations
27. Research questions
28. Thesis mapping
29. Citation/reference section

For every phase provide explicit commands/instructions such as:

```text
Phase 1
Purpose:
Input:
Command:
Notebook:
Expected output:
Metrics:
Where results are saved:
Success criteria:
```

The README must distinguish:

* what has been implemented
* what is experimental
* what is a placeholder
* what is not yet implemented

Never claim a component works if it has not been tested.

---

# 21. Code Quality

Use:

* Python 3.10+
* type hints where useful
* docstrings for public interfaces
* modular functions/classes
* meaningful names
* clear error messages
* configuration-driven execution

Avoid:

* global mutable state
* hard-coded Kaggle paths
* hidden downloads
* duplicated notebook code
* giant monolithic scripts
* unnecessary abstractions
* premature optimization

---

# 22. Research Integrity

Do not fabricate:

* benchmark results
* WER
* latency
* dataset statistics
* model capabilities
* training results
* citations

If something has not been measured, explicitly label it:

```text
NOT MEASURED
```

If something is a placeholder:

```text
PLACEHOLDER
```

If a design decision is uncertain:

```text
RESEARCH DECISION PENDING
```

---

# 23. Phase Completion Contract

At the end of every phase, produce:

### A. Code

All required implementation files.

### B. Notebook

A standalone Kaggle-compatible notebook.

### C. Tests

Relevant unit tests.

### D. Smoke test compatibility

The global smoke test must continue to work.

### E. Configuration

A YAML configuration for the phase.

### F. Results

Measured metrics and generated artifacts.

### G. Documentation

Update:

```text
README.md
docs/architecture.md
docs/experiments.md
```

when relevant.

### H. Phase report

Create a concise machine-readable and human-readable summary containing:

* what was implemented
* files changed
* commands used
* dataset/model used
* metrics
* results
* known problems
* limitations
* next recommended phase

---

# 24. Do Not Over-Implement

This is critical.

Do not implement future research components merely because they appear in the architecture.

For example, during the WhisperRT baseline phase:

Do NOT implement:

* overlap detection
* separation
* multi-talker training
* diarization
* adaptive routing

unless the phase explicitly requires them.

Each phase should have a narrow objective.

---

# 25. Current Research Pipeline

The intended long-term architecture is:

```text
English Audio Stream
        ↓
Streaming Audio Buffer
        ↓
Streaming VAD
        ↓
Overlap Speech Detector
        ↓
   ┌────┴─────┐
   │          │
Normal       Overlap
   │          │
   ↓          ↓
WhisperRT   Overlap-aware /
Streaming   Multi-Talker ASR
   │          │
   └────┬─────┘
        ↓
Incremental Transcript
        ↓
Evaluation
```

This architecture is provisional.

The final research contribution must be determined from the literature and experimental results.

---

# 26. Required Phase Order

Implement in this exact order unless a technical dependency requires otherwise:

```text
Phase 0
Environment + project skeleton

Phase 1
WhisperRT streaming baseline

Phase 2
Streaming VAD

Phase 3
Synthetic overlap dataset

Phase 4
Overlap baseline

Phase 5
Overlap detector abstraction + model

Phase 6
Adaptive pipeline

Phase 7
Multi-talker / overlap-aware recognition

Phase 8
Training / fine-tuning / PEFT

Phase 9
Evaluation + ablation

Phase 10
Final reproducibility + thesis experiments
```

Do not begin Phase N+1 until Phase N satisfies its success criteria.

---

# 27. First Task

When this master specification is first provided, do NOT immediately implement the entire project.

First:

1. inspect the repository
2. create the project skeleton
3. create configuration infrastructure
4. create logging infrastructure
5. create dataset/model backend abstractions
6. create test infrastructure
7. create the smoke-test framework
8. create initial README.md
9. create the Phase 0 notebook
10. verify the Kaggle execution assumptions

Then stop and report:

* files created
* architecture
* commands tested
* tests passed
* smoke-test status
* unresolved issues
* exact command for starting Phase 1

Do not implement WhisperRT yet unless explicitly instructed by the Phase 1 prompt.
