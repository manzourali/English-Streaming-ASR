# PHASE 0 — Project Infrastructure & Skeleton

You are implementing **Phase 0** of the project:

> **English Streaming ASR with Overlapped Speech**

This phase operates strictly under the previously defined **Master Prompt / Project Contract**. Do not redefine the project scope or introduce architectural decisions that conflict with it.

The purpose of Phase 0 is to create a **clean, modular, testable, Kaggle-compatible project foundation** that can support all later phases.

## 0. Critical Scope Boundary

Phase 0 is **infrastructure only**.

Do NOT implement:

* WhisperRT inference
* WhisperRT fine-tuning
* actual ASR inference
* actual VAD inference
* actual OSD inference
* speaker diarization
* speaker identification
* speaker enrollment
* multi-talker ASR
* speech separation
* overlap-aware recognition
* adaptive routing logic based on a trained model
* dataset downloading at repository initialization
* large model downloads
* large dataset downloads
* training loops
* real benchmark experiments

You may create **interfaces, abstract classes, configuration entries, placeholders, mocks, and minimal lifecycle implementations** required by later phases.

Every future component must have a clean integration point.

---

# 1. First: Inspect the Existing Repository

Before modifying anything:

1. Inspect the complete repository structure.
2. Identify:

   * existing Python files
   * existing notebooks
   * existing configuration files
   * existing README
   * existing tests
   * existing scripts
   * existing requirements/environment files
   * existing source code
   * existing datasets/manifests
   * existing model-related code
3. Determine whether the repository is:

   * empty
   * partially implemented
   * already contains useful components
4. Preserve useful existing work.
5. Do not blindly overwrite existing files.
6. If existing code conflicts with the Master Prompt, document the conflict and refactor only when necessary.
7. Do not delete potentially useful research code merely because it is incomplete.

Before implementation, produce an internal inventory of the repository.

If the repository is already partially implemented, integrate the Phase 0 architecture with it instead of unnecessarily rebuilding everything.

---

# 2. Target Repository Structure

Create or complete the following structure:

```text
english-streaming-asr-overlap/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── kaggle.yaml
│   ├── whisperrt_baseline.yaml
│   ├── streaming_vad.yaml
│   ├── overlap_dataset.yaml
│   ├── overlap_detection.yaml
│   ├── adaptive.yaml
│   ├── multitalker.yaml
│   ├── training.yaml
│   └── evaluation.yaml
│
├── src/
│   └── streaming_asr/
│       ├── __init__.py
│       │
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── stream.py
│       │   ├── buffer.py
│       │   └── preprocessing.py
│       │
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   ├── manifests.py
│       │   ├── overlap_generator.py
│       │   └── validators.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── whisperrt.py
│       │   ├── vad.py
│       │   ├── overlap_detector.py
│       │   └── multitalker.py
│       │
│       ├── streaming/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── state.py
│       │   └── scheduler.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── baseline.py
│       │   ├── streaming.py
│       │   ├── overlap.py
│       │   └── adaptive.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── peft.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   ├── benchmark.py
│       │   └── reports.py
│       │
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── asr.py
│       │   ├── streaming.py
│       │   └── overlap.py
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── audio.py
│       │   └── plots.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── logging.py
│           ├── paths.py
│           └── reproducibility.py
│
├── scripts/
│   ├── check_environment.py
│   ├── prepare_dataset.py
│   ├── generate_overlap.py
│   ├── run_baseline.py
│   ├── run_streaming.py
│   ├── train_osd.py
│   ├── train_asr.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── smoke_test.py
│
├── notebooks/
│   ├── 00_environment.ipynb
│   ├── 01_dataset_validation.ipynb
│   ├── 02_whisperrt_baseline.ipynb
│   ├── 03_streaming_vad.ipynb
│   ├── 04_overlap_generation.ipynb
│   ├── 05_overlap_baseline.ipynb
│   ├── 06_overlap_detection.ipynb
│   ├── 07_adaptive_pipeline.ipynb
│   ├── 08_multitalker.ipynb
│   ├── 09_training.ipynb
│   └── 10_evaluation.ipynb
│
├── tests/
│   ├── test_audio.py
│   ├── test_vad.py
│   ├── test_streaming.py
│   ├── test_overlap.py
│   ├── test_pipeline.py
│   └── test_metrics.py
│
├── data/
│   ├── manifests/
│   └── samples/
│
├── checkpoints/
│
├── outputs/
│   ├── logs/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   └── reports/
│
└── docs/
    ├── architecture.md
    ├── datasets.md
    ├── experiments.md
    └── thesis_mapping.md
```

Create directories that are needed for the repository to work.

Do not place large datasets or model checkpoints in Git.

---

# 3. Python Package Design

Use a proper `src/` layout.

The project must be importable as:

```python
from streaming_asr...
```

Configure `pyproject.toml` accordingly.

Use modern Python packaging.

Target Python version should be explicitly documented.

Prefer a Python version that is compatible with:

* PyTorch
* Hugging Face ecosystem
* Kaggle
* later WhisperRT dependencies

Do not unnecessarily pin every transitive dependency to an exact version.

---

# 4. Dependency Strategy

Phase 0 should remain lightweight.

Separate dependencies conceptually into:

### Core dependencies

Only dependencies genuinely needed for Phase 0, such as:

* PyYAML
* NumPy
* soundfile if needed
* pytest

### Future ML dependencies

Document, but do not necessarily require them for Phase 0:

* torch
* transformers
* datasets
* accelerate
* evaluate
* peft
* librosa
* sounddevice
* torchaudio
* etc.

Do not force a large model stack to be installed merely to execute:

```bash
python scripts/smoke_test.py
```

The smoke test must run without downloading a model.

---

# 5. Configuration System

Implement:

```text
src/streaming_asr/utils/config.py
```

The configuration system must support:

* YAML loading
* base configuration
* optional environment-specific configuration
* nested configuration access
* configuration merging
* validation of required fields
* conversion to a normal Python dictionary where useful

Example:

```bash
python scripts/smoke_test.py --config configs/base.yaml
```

and:

```bash
python scripts/smoke_test.py \
    --config configs/base.yaml \
    --override configs/kaggle.yaml
```

If a different CLI design is technically cleaner, use it, but it must support equivalent functionality.

The system must allow configuration such as:

```yaml
experiment:
  name: phase0_infrastructure
  seed: 42

runtime:
  environment: local

data:
  backend: huggingface

model:
  backend: huggingface

audio:
  sample_rate: 16000
  chunk_ms: 320

streaming:
  enabled: true

logging:
  level: INFO
```

Support backend switching through configuration rather than hard-coded Python conditionals.

---

# 6. Configuration Files

Create at minimum:

```text
configs/base.yaml
configs/kaggle.yaml
configs/whisperrt_baseline.yaml
configs/streaming_vad.yaml
configs/overlap_dataset.yaml
configs/overlap_detection.yaml
configs/adaptive.yaml
configs/multitalker.yaml
configs/training.yaml
configs/evaluation.yaml
```

Only `base.yaml` and `kaggle.yaml` need meaningful Phase 0 behavior.

The remaining configuration files should be valid, documented placeholders for future phases.

Do not pretend that future settings are already implemented.

Clearly mark future configuration fields when appropriate.

---

# 7. Local and Kaggle Path Management

Implement:

```text
src/streaming_asr/utils/paths.py
```

The path manager must distinguish between:

### Local environment

Example:

```text
./data
./outputs
./checkpoints
```

### Kaggle environment

Input datasets are normally under:

```text
/kaggle/input/
```

Outputs must NOT be written to `/kaggle/input`.

Use appropriate writable locations such as:

```text
/kaggle/working/
```

The system should allow configuration like:

```yaml
runtime:
  environment: kaggle
```

and automatically resolve appropriate directories.

Do not hard-code the user's personal filesystem paths.

Do not assume a specific Kaggle dataset slug.

---

# 8. Dataset Backend Abstraction

Create an abstraction capable of supporting:

```text
Kaggle
Hugging Face
Local filesystem
```

at minimum.

Relevant files include:

```text
src/streaming_asr/datasets/loaders.py
src/streaming_asr/datasets/manifests.py
src/streaming_asr/datasets/validators.py
```

Create a conceptual interface such as:

```python
class DatasetBackend:
    def load(self, config):
        raise NotImplementedError
```

or a cleaner equivalent.

Implement lightweight backend adapters:

```text
HuggingFaceDatasetBackend
KaggleDatasetBackend
LocalDatasetBackend
```

Phase 0 does NOT need to download or process LibriSpeech.

The purpose is only to establish the architecture.

The backend should make later usage possible through configuration:

```yaml
data:
  backend: huggingface
```

or:

```yaml
data:
  backend: kaggle
```

Avoid making later phases dependent on one specific data-access mechanism.

---

# 9. Model Backend Abstraction

Create:

```text
src/streaming_asr/models/
```

and establish a model-loading abstraction supporting:

```text
Hugging Face
Kaggle/local checkpoint
```

Do NOT download WhisperRT in Phase 0.

Create an interface conceptually similar to:

```python
class ModelBackend:
    def load(self, model_config):
        raise NotImplementedError
```

Possible adapters:

```text
HuggingFaceModelBackend
LocalModelBackend
KaggleModelBackend
```

The exact naming can differ if a cleaner architecture is justified.

The key requirement is:

> Later phases must be able to change the model source through configuration without rewriting the pipeline.

Example:

```yaml
model:
  backend: huggingface
  name: MLSpeech/WhisperRT-Streaming
```

Phase 0 must not instantiate this model.

---

# 10. Audio Abstractions

Implement the foundational audio interfaces.

Relevant files:

```text
audio/loader.py
audio/stream.py
audio/buffer.py
audio/preprocessing.py
```

Provide basic concepts for:

* audio metadata
* sample rate
* mono/stereo representation
* PCM/audio array representation
* chunk representation
* timestamps
* chunk duration
* audio stream lifecycle

A chunk should carry enough metadata for later streaming evaluation.

Conceptually:

```python
AudioChunk(
    samples=...,
    sample_rate=16000,
    start_time=...,
    end_time=...,
    index=...
)
```

Do not over-engineer this.

The interface should remain simple and extensible.

---

# 11. Streaming Engine Skeleton

Implement:

```text
src/streaming_asr/streaming/engine.py
src/streaming_asr/streaming/state.py
src/streaming_asr/streaming/scheduler.py
```

The streaming engine must support incremental processing.

Conceptually:

```python
engine.start()

for chunk in stream:
    output = engine.process(chunk)

engine.finalize()
```

Create a state object that can later hold:

* current stream time
* chunk index
* buffer state
* VAD state
* ASR state
* overlap state
* emitted transcript state
* timing information

However, Phase 0 should only initialize the generic state.

Do not implement actual VAD/ASR behavior.

---

# 12. Streaming State Requirements

Create a state object with enough structure to support later phases.

For example:

```python
StreamingState:
    started
    finalized
    chunk_index
    stream_time
    metadata
```

It may later be extended with:

```text
vad_state
osd_state
asr_state
decoder_state
buffer_state
routing_state
metrics_state
```

Do not implement those future algorithms yet.

---

# 13. Pipeline Skeleton

Create:

```text
pipeline/baseline.py
pipeline/streaming.py
pipeline/overlap.py
pipeline/adaptive.py
```

These files should establish the future architecture.

At Phase 0, they may contain interfaces/placeholders.

Create a generic lifecycle such as:

```python
pipeline.start()

result = pipeline.process(audio_chunk)

pipeline.finalize()
```

The pipeline should not require an actual ASR model in Phase 0.

---

# 14. Future Component Interfaces

Create clean interfaces/placeholders for:

### Streaming ASR

```python
class StreamingASR:
    def process(self, audio_chunk):
        ...
    
    def finalize(self):
        ...
```

### Streaming VAD

```python
class StreamingVAD:
    def process(self, audio_chunk):
        ...
```

### Streaming OSD

```python
class StreamingOverlapDetector:
    def process(self, audio_chunk):
        ...
```

### Pipeline

```python
class StreamingASRPipeline:
    def process(self, audio_chunk):
        ...
```

Do not implement real inference.

A dummy implementation may be used for Phase 0 testing.

---

# 15. Metric Interfaces

Create:

```text
src/streaming_asr/metrics/asr.py
src/streaming_asr/metrics/streaming.py
src/streaming_asr/metrics/overlap.py
```

Establish interfaces for future metrics.

ASR:

* WER
* CER

Streaming:

* RTF
* first-token latency
* average token latency
* end-of-utterance latency
* chunk processing time
* memory/GPU measurements where possible

Overlap:

* precision
* recall
* F1

Phase 0 does NOT need to calculate real WER or OSD metrics unless a tiny dependency-free implementation is useful.

Do not fabricate values.

---

# 16. Experiment Logging

Implement simple experiment logging.

Relevant file:

```text
src/streaming_asr/utils/logging.py
```

Each experiment should have a run directory similar to:

```text
outputs/
└── logs/
    └── 2026-XX-XX_phase0_infrastructure/
        ├── run.log
        ├── config.yaml
        └── metadata.json
```

The exact structure can differ slightly.

Record at minimum:

* experiment name
* timestamp
* Python version
* operating system
* runtime environment
* seed
* configuration
* package/project version if available

Do not introduce MLflow/W&B/etc. unless explicitly required later.

The project should remain lightweight.

---

# 17. Reproducibility Utilities

Implement:

```text
src/streaming_asr/utils/reproducibility.py
```

Provide a simple seed-setting mechanism.

It should support at least:

* Python random
* NumPy

If PyTorch is installed, support PyTorch conditionally without making PyTorch mandatory for Phase 0.

Do not fail merely because PyTorch is unavailable.

---

# 18. Environment Check Script

Implement:

```text
scripts/check_environment.py
```

It should report:

* Python version
* project importability
* operating system
* whether running on Kaggle
* available writable directories
* whether optional packages are installed
* basic CPU information
* GPU availability if PyTorch is installed
* CUDA information if available

Do not treat missing future-phase packages as fatal errors.

Clearly distinguish:

```text
REQUIRED
OPTIONAL
NOT INSTALLED
```

---

# 19. Smoke Test

Implement:

```text
scripts/smoke_test.py
```

This is a critical component.

The Phase 0 smoke test must:

1. load the configuration
2. initialize paths
3. initialize logging
4. create a synthetic audio signal
5. divide it into streaming chunks
6. pass chunks through the streaming engine
7. exercise the pipeline lifecycle
8. exercise dummy VAD/OSD interfaces where appropriate
9. verify state progression
10. finalize the pipeline
11. write a small run artifact
12. exit successfully

It must NOT:

* download a model
* download a dataset
* require internet
* require GPU
* require Kaggle
* require WhisperRT
* require VAD model weights

A successful Phase 0 smoke test should demonstrate that the architecture can support a streaming workflow.

Example conceptual flow:

```text
Synthetic Audio
      ↓
AudioChunk Generator
      ↓
Streaming Engine
      ↓
Dummy VAD
      ↓
Dummy OSD
      ↓
Dummy ASR
      ↓
Pipeline Output
      ↓
Run Metadata
```

Use deterministic synthetic audio.

For example:

* short sine wave
* silence
* another short waveform

Do not use an actual human speech recording.

The test should remain fast.

---

# 20. Tests

Create and implement:

```text
tests/test_audio.py
tests/test_vad.py
tests/test_streaming.py
tests/test_overlap.py
tests/test_pipeline.py
tests/test_metrics.py
```

Tests should verify interfaces and lifecycle behavior.

## test_audio.py

Test:

* audio chunk creation
* sample rate
* timestamps
* chunk duration
* buffer behavior
* invalid input handling where appropriate

## test_vad.py

Because there is no real VAD yet:

* test interface
* test dummy implementation
* test state lifecycle
* test expected output schema

Do NOT claim actual VAD accuracy.

## test_streaming.py

Test:

* stream initialization
* sequential chunks
* chunk index progression
* timestamps
* finalize behavior
* repeated finalize behavior if defined

## test_overlap.py

Test:

* OSD interface
* dummy classification schema
* supported states

For example:

```text
NO_SPEECH
SINGLE_SPEAKER
OVERLAP
```

## test_pipeline.py

Test:

* start
* process
* finalize
* state propagation
* dummy components

## test_metrics.py

Test metric interfaces and simple deterministic cases where implemented.

Do not add meaningless placeholder assertions simply to increase test count.

---

# 21. Notebook 00 — Environment

Create:

```text
notebooks/00_environment.ipynb
```

This notebook must be independently executable in Kaggle.

It should demonstrate:

1. Python version
2. environment information
3. project import
4. configuration loading
5. path resolution
6. package availability
7. synthetic audio generation
8. streaming chunk creation
9. smoke-test-style pipeline execution
10. output directory verification

It should NOT:

* download WhisperRT
* download LibriSpeech
* perform ASR
* train anything

The notebook should clearly explain:

> Phase 0 establishes the environment and infrastructure only.

Use the repository source package rather than duplicating implementation logic in notebook cells.

The notebook should call project code wherever possible.

---

# 22. README.md

Create a comprehensive initial:

```text
README.md
```

It must describe:

## 22.1 Project Overview

English streaming ASR with overlapped speech.

## 22.2 Research Objective

Explain the long-term objective without claiming that future components are already implemented.

## 22.3 Current Phase

Clearly state:

```text
Current phase: Phase 0 — Infrastructure
Status: Infrastructure implementation
```

## 22.4 Scope

Explicitly state:

* English
* streaming ASR
* overlapped speech
* streaming VAD
* OSD
* multi-talker recognition later
* adaptive routing later
* diarization currently out of scope

## 22.5 Architecture

Include the conceptual architecture:

```text
Audio Stream
    ↓
Streaming Buffer
    ↓
Streaming VAD
    ↓
Overlap Speech Detection
    ↓
Normal ASR / Overlap-aware ASR
    ↓
Incremental Transcript
    ↓
Evaluation
```

Clearly indicate which components are currently implemented and which are future phases.

## 22.6 Repository Structure

Explain major directories.

## 22.7 Installation

Include local installation.

## 22.8 Kaggle Setup

Explain:

* `/kaggle/input`
* `/kaggle/working`
* repository setup
* dataset inputs
* no writing to `/kaggle/input`

## 22.9 Hugging Face Integration

Explain the intended future backend.

Do not claim Phase 0 downloads models.

## 22.10 Configuration

Explain YAML configuration and backend switching.

## 22.11 Smoke Test

Provide exact command.

## 22.12 Tests

Provide exact command.

Example:

```bash
pytest -q
```

## 22.13 Environment Check

Provide:

```bash
python scripts/check_environment.py
```

## 22.14 Phase Roadmap

Document:

```text
Phase 0 — Infrastructure
Phase 1 — WhisperRT streaming baseline
Phase 2 — Streaming VAD
Phase 3 — Synthetic overlap generation
Phase 4 — Overlap benchmark
Phase 5 — OSD
Phase 6 — Adaptive routing
Phase 7 — Multi-talker / overlap-aware ASR
Phase 8 — Training / PEFT
Phase 9 — Evaluation / ablation
Phase 10 — Reproducibility / thesis experiments
```

## 22.15 Research Integrity

State explicitly:

* no fabricated results
* unmeasured values must remain unmeasured
* placeholders must be labeled
* research decisions pending must be identified

---

# 23. Documentation

Create:

## docs/architecture.md

Describe:

* module boundaries
* data flow
* streaming lifecycle
* backend abstraction
* future extension points
* current limitations

## docs/datasets.md

Describe planned dataset roles:

```text
LibriSpeech
Synthetic Overlap
LibriSpeechMix
LibriCSS
AMI / CHiME-6
```

Do not claim that they have been downloaded or validated in Phase 0.

Clearly distinguish:

```text
planned
available
implemented
validated
```

## docs/experiments.md

Define:

* experiment configuration
* run directory
* seeds
* metrics
* output artifacts
* reproducibility expectations

## docs/thesis_mapping.md

Explain how the software phases map conceptually to the thesis research.

Do not claim scientific contributions before they are experimentally demonstrated.

---

# 24. `.gitignore`

Ensure the repository ignores at least:

```text
__pycache__/
*.py[cod]
.pytest_cache/
.ipynb_checkpoints/
.venv/
venv/
.env

checkpoints/*
outputs/*
data/*
```

If useful, preserve:

```text
data/manifests/.gitkeep
data/samples/.gitkeep
outputs/.gitkeep
```

Do not accidentally ignore source configuration files or notebooks.

Do not commit:

* model weights
* datasets
* generated audio
* experiment outputs
* Kaggle secrets
* Hugging Face tokens
* API keys

---

# 25. Security / Credentials

Never hard-code:

* Hugging Face tokens
* Kaggle API tokens
* passwords
* private paths
* credentials

If authentication is needed later, use environment variables or the relevant platform mechanism.

Do not create fake credentials for testing.

---

# 26. Kaggle Compatibility

Phase 0 must be executable in Kaggle.

Assume:

```text
/kaggle/input
/kaggle/working
```

may exist.

Do not assume that:

* Internet is enabled
* a dataset is attached
* GPU is enabled
* Hugging Face credentials exist

The infrastructure must fail gracefully when optional resources are unavailable.

The Phase 0 smoke test must work without Internet.

---

# 27. Error Handling

Use meaningful exceptions and error messages.

Examples:

```text
ConfigurationError
BackendNotAvailableError
InvalidAudioChunkError
StreamingStateError
```

You do not need to create a large custom exception hierarchy.

Do not silently swallow errors.

Optional dependencies should be detected explicitly.

---

# 28. Type Hints and Code Quality

Use type hints throughout the new code.

Prefer:

```python
def process(self, chunk: AudioChunk) -> ...
```

over untyped interfaces.

Use concise docstrings for public classes/functions.

Avoid excessive abstraction.

The goal is:

> research-grade modularity without enterprise-level overengineering.

---

# 29. No Premature Implementation

Do not implement future functionality merely because the corresponding files exist.

For example:

```text
models/whisperrt.py
```

may contain an interface or placeholder.

It must NOT download or initialize:

```text
MLSpeech/WhisperRT-Streaming
```

during Phase 0.

Likewise:

```text
models/vad.py
models/overlap_detector.py
models/multitalker.py
```

should contain interfaces or dummy components only.

---

# 30. Future Model Identifier

Reserve configuration support for:

```yaml
model:
  backend: huggingface
  name: MLSpeech/WhisperRT-Streaming
```

But do not download or run this model in Phase 0.

Phase 1 will implement the actual WhisperRT integration.

---

# 31. Future Dataset Backends

Reserve configuration support for:

```yaml
data:
  backend: huggingface
```

and:

```yaml
data:
  backend: kaggle
```

Also support local data where useful:

```yaml
data:
  backend: local
```

Phase 0 only needs to establish the interfaces.

---

# 32. Output Contract

Every Phase 0 run should be able to produce structured information such as:

```json
{
  "phase": "phase0",
  "experiment": "phase0_infrastructure",
  "status": "success",
  "environment": "local",
  "seed": 42,
  "tests": {
    "passed": true
  },
  "smoke_test": {
    "passed": true
  }
}
```

The exact schema may differ, but it must be machine-readable.

Do not include fake model metrics.

---

# 33. Validation Sequence

After implementation, run the following in order.

## Step 1 — Environment

```bash
python scripts/check_environment.py
```

## Step 2 — Import test

Verify:

```python
import streaming_asr
```

## Step 3 — Unit tests

```bash
pytest -q
```

## Step 4 — Smoke test

```bash
python scripts/smoke_test.py --config configs/base.yaml
```

## Step 5 — Kaggle configuration validation

Validate:

```bash
python scripts/smoke_test.py --config configs/kaggle.yaml
```

This must validate configuration/path behavior without requiring an attached dataset or GPU.

## Step 6 — Notebook

Execute or validate:

```text
notebooks/00_environment.ipynb
```

If automated notebook execution is unavailable, validate its cells and imports programmatically as far as practical.

---

# 34. Expected Phase 0 Result

At the end of Phase 0, the repository must have:

### Working

* project package
* configuration system
* local/Kaggle path handling
* logging
* reproducibility utilities
* dataset backend abstraction
* model backend abstraction
* audio abstraction
* streaming state
* streaming engine skeleton
* pipeline skeleton
* metric interfaces
* unit tests
* smoke test
* environment check
* Phase 0 notebook
* initial README
* architecture documentation

### Not implemented yet

* WhisperRT
* real ASR
* real VAD
* real OSD
* overlap training
* multi-talker ASR
* adaptive routing
* PEFT
* real evaluation benchmark

---

# 35. Phase 0 Success Criteria

Phase 0 is considered complete only if all of the following are true:

* [ ] Existing repository was inspected before modification.
* [ ] Useful existing code was preserved.
* [ ] `src/` package imports successfully.
* [ ] Configuration loading works.
* [ ] Configuration merging works.
* [ ] Local paths work.
* [ ] Kaggle path configuration works.
* [ ] `/kaggle/input` is treated as read-only.
* [ ] Output paths are writable.
* [ ] Dataset backend abstraction exists.
* [ ] Hugging Face backend interface exists.
* [ ] Kaggle backend interface exists.
* [ ] Local backend exists or is appropriately supported.
* [ ] Model backend abstraction exists.
* [ ] Audio chunk abstraction works.
* [ ] Streaming state works.
* [ ] Streaming engine lifecycle works.
* [ ] Pipeline lifecycle works.
* [ ] Dummy components can pass data through the pipeline.
* [ ] Metric interfaces exist.
* [ ] Logging works.
* [ ] Reproducibility utilities work.
* [ ] Environment checker works.
* [ ] `pytest -q` passes.
* [ ] Phase 0 smoke test passes.
* [ ] Kaggle configuration smoke test passes without requiring GPU/data.
* [ ] `00_environment.ipynb` is valid.
* [ ] README is complete enough for a new researcher to run Phase 0.
* [ ] Documentation reflects actual implementation status.
* [ ] No fabricated metrics/results exist.
* [ ] No model weights or datasets are committed.
* [ ] No credentials are committed.
* [ ] Phase 1 has NOT been implemented.

---

# 36. Final Phase 0 Report

After implementation and validation, provide a concise report containing:

## A. Repository changes

List the important files created or modified.

## B. Architecture

Summarize the implemented Phase 0 architecture.

## C. Commands executed

List exact commands used.

## D. Test results

Report:

```text
pytest: PASS/FAIL
smoke test: PASS/FAIL
Kaggle config validation: PASS/FAIL
notebook validation: PASS/FAIL
```

Use actual measured results only.

## E. Environment

Report:

* Python version
* OS
* GPU availability
* CUDA availability if applicable

## F. Known issues

List unresolved issues honestly.

If there are none, say:

```text
None identified during Phase 0 validation.
```

## G. Research decisions pending

Identify any decisions intentionally deferred to later phases.

## H. Phase 1 readiness

State whether the repository is ready for:

> **Phase 1 — WhisperRT Streaming Baseline**

Do not implement Phase 1.

---

# 37. Final Instruction

Implement **only Phase 0**.

Do not jump ahead.

Do not install unnecessary heavyweight ML dependencies.

Do not download WhisperRT.

Do not download LibriSpeech.

Do not train anything.

Do not fabricate benchmark results.

Do not claim streaming ASR functionality merely because the interfaces exist.

The objective is to leave a **clean, tested, reproducible, extensible foundation** on which Phase 1 can implement the real:

```text
MLSpeech/WhisperRT-Streaming
```

streaming ASR baseline.

When Phase 0 is complete, stop and provide the Phase 0 report described above.
