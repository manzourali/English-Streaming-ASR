# PHASE 1 — WhisperRT Streaming Baseline

You are implementing **Phase 1** of the project:

> **English Streaming ASR with Overlapped Speech**

This phase operates strictly under the previously defined **master_prompt.md (Project Contract)** and after the completion of **Phase 0 — Project Infrastructure & Skeleton**.

The purpose of this phase is to integrate the real:

```text
MLSpeech/WhisperRT-Streaming
```

model into the Phase 0 architecture and establish a **reproducible, measurable English streaming ASR baseline using LibriSpeech**.

This baseline will become the reference point for all later phases involving VAD, overlap detection, overlap-aware recognition, adaptive routing, and multi-talker ASR.

---

# 1. Phase 1 Objective

The primary objective is:

> Implement and validate a real streaming ASR pipeline using `MLSpeech/WhisperRT-Streaming` on English LibriSpeech audio, while measuring both recognition quality and streaming/latency behavior.

The implementation must demonstrate that the project can:

```text
LibriSpeech Audio
       ↓
Audio Loader
       ↓
Streaming Chunker
       ↓
WhisperRT-Streaming
       ↓
Incremental / Streaming Transcript
       ↓
WER + Streaming Metrics
       ↓
Experiment Report
```

The result must be a **real streaming baseline**, not an offline Whisper model wrapped in an artificial chunking loop.

---

# 2. Strict Scope Boundary

Phase 1 must implement:

* real WhisperRT-Streaming integration
* model loading
* Hugging Face model backend integration
* local/Kaggle model loading where practical
* English LibriSpeech loading
* audio preprocessing required by WhisperRT
* streaming chunk generation
* streaming inference
* incremental transcript handling
* final transcript generation
* WER evaluation
* basic streaming latency measurements
* RTF measurement
* experiment configuration
* reproducible benchmark
* Phase 1 notebook
* Phase 1 tests
* README updates
* experiment report

Phase 1 must NOT implement:

* streaming VAD
* Silero VAD
* WebRTC VAD
* TEN VAD
* OSD model
* overlap generation
* LibriSpeechMix processing
* LibriCSS evaluation
* speaker diarization
* speaker identification
* speaker enrollment
* multi-talker ASR
* speech separation
* adaptive routing
* overlap-aware routing
* PEFT
* ASR fine-tuning
* RL
* custom WhisperRT training
* model architecture modification

These belong to later phases.

The only "VAD" allowed in Phase 1 is functionality that is inherently required by the actual WhisperRT implementation itself. Do not build a separate VAD subsystem.

---

# 3. First Step — Inspect Phase 0

Before implementation:

1. Inspect the repository.
2. Verify that Phase 0 exists.
3. Inspect:

   * configuration system
   * model backend abstraction
   * dataset backend abstraction
   * audio abstractions
   * streaming engine
   * pipeline interfaces
   * metrics
   * tests
   * smoke test
   * README
4. Do not rebuild Phase 0 unnecessarily.
5. Extend the existing abstractions.
6. Preserve working Phase 0 behavior.

Run:

```bash
python scripts/check_environment.py
pytest -q
python scripts/smoke_test.py --config configs/base.yaml
```

If Phase 0 is broken, diagnose and fix the minimal necessary issue before implementing Phase 1.

Do not silently ignore Phase 0 regressions.

---

# 4. Critical Technical Requirement — Real Streaming

This is the most important requirement of Phase 1.

Do NOT implement:

```text
load complete audio
        ↓
split into chunks
        ↓
run offline Whisper independently on every chunk
```

That is not an acceptable implementation of streaming ASR.

The implementation must use the actual causal/streaming behavior provided by:

```text
MLSpeech/WhisperRT-Streaming
```

and preserve the model's streaming state across successive audio chunks.

The conceptual lifecycle should be:

```python
stream.start()

for chunk in audio_stream:
    partial = stream.process(chunk)

stream.finalize()
```

The exact API must follow the actual `MLSpeech/WhisperRT-Streaming` implementation.

Do not invent an API based on assumptions.

---

# 5. Verify the Actual WhisperRT Implementation

Before coding against the model, inspect the actual Hugging Face repository/model implementation.

Verify:

* model class
* processor/tokenizer
* expected sampling rate
* input format
* streaming API
* state management
* chunk size requirements
* context requirements
* look-ahead/look-back behavior if present
* incremental output behavior
* finalization mechanism
* supported device types
* dtype requirements
* generation configuration
* repository dependencies
* installation requirements

Do not assume that the model exposes a generic Hugging Face `pipeline()` API.

If the repository contains official inference examples, follow them as the primary reference.

If documentation and source behavior differ:

1. inspect the implementation,
2. determine the actual working API,
3. document the finding in the Phase 1 report.

Do not fabricate unsupported functionality.

---

# 6. Model Integration

Extend:

```text
src/streaming_asr/models/whisperrt.py
```

The model adapter should encapsulate WhisperRT-specific behavior.

The rest of the project should not need to know WhisperRT implementation details.

Conceptually:

```python
class WhisperRTStreamingASR(StreamingASR):
    def __init__(self, config):
        ...

    def start(self):
        ...

    def process(self, audio_chunk):
        ...

    def finalize(self):
        ...
```

The exact interface may be adapted to the actual WhisperRT API.

The adapter must:

* load the model
* load required processor/tokenizer
* configure device
* configure dtype
* maintain streaming state
* accept incremental audio chunks
* return incremental output
* finalize the stream
* expose timing information where possible

---

# 7. Model Backend Integration

Integrate the existing Phase 0 model backend abstraction.

The configuration must support:

```yaml
model:
  backend: huggingface
  name: MLSpeech/WhisperRT-Streaming
```

The implementation should not hard-code:

```python
from_pretrained("MLSpeech/WhisperRT-Streaming")
```

inside the pipeline.

The model name should come from configuration.

Support local/checkpoint loading if the actual model implementation makes this possible.

The architecture should remain:

```text
Configuration
     ↓
Model Backend
     ↓
WhisperRT Adapter
     ↓
Streaming ASR Interface
     ↓
Pipeline
```

---

# 8. Hugging Face Model Loading

The preferred Phase 1 model source is:

```text
MLSpeech/WhisperRT-Streaming
```

from Hugging Face.

Use the appropriate official loading mechanism discovered from the repository.

Do not assume that:

```python
AutoModel.from_pretrained(...)
```

is necessarily sufficient.

If custom code is required by the repository, use it only when verified from the model's official implementation.

Document:

* exact model identifier
* loading method
* required packages
* required `trust_remote_code` behavior if any
* device
* dtype
* model revision if explicitly pinned

Do not pin an arbitrary revision without justification.

---

# 9. Device Handling

Support at minimum:

```text
CPU
CUDA GPU
```

The implementation must detect the available device.

Configuration should allow:

```yaml
runtime:
  device: auto
```

with possible values:

```text
auto
cpu
cuda
```

If CUDA is requested but unavailable, fail with a clear error rather than silently falling back unless fallback behavior is explicitly configured.

---

# 10. Precision / Dtype

Allow configuration such as:

```yaml
model:
  dtype: auto
```

Possible implementation choices may include:

```text
float32
float16
bfloat16
auto
```

Do not assume that every dtype is supported by every device.

The adapter should validate the combination.

For example:

* CPU may require float32 depending on implementation.
* CUDA may support float16/bfloat16 depending on hardware and model.

Record the actual dtype in experiment metadata.

---

# 11. Audio Requirements

The baseline should use:

```text
English
16 kHz
mono
PCM floating-point waveform
```

unless the actual WhisperRT implementation explicitly requires another representation.

The loader must normalize the input appropriately.

Do not resample unnecessarily if the input is already at the target rate.

Audio preprocessing should be deterministic.

---

# 12. LibriSpeech Dataset

Use **LibriSpeech** as the primary Phase 1 benchmark dataset.

The goal is to establish a clean English single-speaker streaming baseline before introducing overlap.

Do not mix overlapping speakers in this phase.

Preferred evaluation subset:

```text
LibriSpeech test-clean
```

Use another subset only if there is a technical reason.

If a smaller subset is needed for development/debugging, support configuration such as:

```yaml
data:
  split: test-clean
  max_samples: 20
```

But the full benchmark configuration must remain available.

---

# 13. Dataset Backend

Use the Phase 0 dataset abstraction.

Support:

```text
Hugging Face
Kaggle
Local
```

where technically appropriate.

The benchmark must not depend on a hard-coded local filesystem path.

For example:

```yaml
data:
  backend: huggingface
  dataset: librispeech
  split: test.clean
```

or an equivalent configuration based on the actual dataset API.

If Kaggle input is used:

```yaml
data:
  backend: kaggle
  path: /kaggle/input/...
```

Do not write anything to:

```text
/kaggle/input
```

---

# 14. Dataset Loader

Extend:

```text
src/streaming_asr/datasets/loaders.py
```

so that Phase 1 can obtain:

```text
audio
reference transcript
sample ID
speaker ID if available
duration
sampling rate
```

Speaker identity must NOT be used by the ASR system.

It may be retained as metadata for dataset analysis if useful.

The evaluation should primarily operate on:

```text
audio + reference transcript
```

---

# 15. Dataset Validation

Before running the benchmark, validate:

* audio exists
* transcript exists
* sampling rate
* number of channels
* duration
* empty audio
* malformed examples

Produce a small dataset validation summary.

Example:

```text
Dataset: LibriSpeech test-clean
Samples evaluated: N
Total duration: X minutes
Sample rate: 16000 Hz
Invalid samples: 0
```

Use actual values.

Never fabricate them.

---

# 16. Streaming Chunker

Implement or extend:

```text
src/streaming_asr/audio/stream.py
src/streaming_asr/audio/buffer.py
```

to provide deterministic streaming chunks.

The chunk duration must be configurable.

For example:

```yaml
audio:
  sample_rate: 16000
  chunk_ms: 320
```

Do not assume 320 ms is the optimal value.

It is only a configurable starting point.

Support later experimentation with multiple chunk sizes.

Potential benchmark values can include:

```text
160 ms
320 ms
640 ms
```

but do not automatically run a large grid unless configured.

The actual chunk sizes should respect WhisperRT's requirements.

---

# 17. Streaming Buffer

The streaming buffer must preserve:

* sample order
* timestamps
* chunk index
* stream position

Avoid copying the entire accumulated audio on every chunk if the WhisperRT API does not require it.

The architecture should allow bounded/incremental processing.

---

# 18. WhisperRT State

The implementation must preserve model streaming state between chunks.

Do not recreate the model for each chunk.

Do not reset the decoder between chunks unless the actual WhisperRT API requires it.

The lifecycle should conceptually be:

```text
create model
      ↓
initialize stream
      ↓
chunk 1 → state update
      ↓
chunk 2 → state update
      ↓
chunk 3 → state update
      ↓
...
      ↓
finalize
```

This requirement must be tested.

---

# 19. Incremental Transcript Handling

WhisperRT may emit:

* partial hypotheses
* updated hypotheses
* tokens
* text segments
* finalized text

depending on the actual API.

Implement a transcript accumulator that can distinguish:

```text
partial output
finalized output
```

Do not simply concatenate repeated partial hypotheses.

For example, if the model emits:

```text
"the"
"the quick"
"the quick brown"
```

the final transcript must not become:

```text
"the the quick the quick brown"
```

Determine the actual output semantics of WhisperRT and implement the correct accumulation logic.

Document this behavior.

---

# 20. Transcript Object

Create a structured representation for streaming recognition output.

It should contain, where available:

```text
text
start_time
end_time
is_final
chunk_index
latency information
```

Keep the representation generic enough for future VAD/OSD integration.

---

# 21. Baseline Pipeline

Implement the actual Phase 1 pipeline in:

```text
src/streaming_asr/pipeline/streaming.py
```

The baseline pipeline should be:

```text
Audio
  ↓
Audio Loader
  ↓
Streaming Chunker
  ↓
WhisperRT Streaming ASR
  ↓
Transcript Accumulator
  ↓
Evaluation
```

There is deliberately NO:

```text
VAD
OSD
Adaptive Router
Multi-Talker
```

in this pipeline.

---

# 22. Baseline Script

Implement:

```text
scripts/run_streaming.py
```

It should support configuration-driven execution.

Example:

```bash
python scripts/run_streaming.py \
    --config configs/whisperrt_baseline.yaml
```

Optional development restriction:

```bash
python scripts/run_streaming.py \
    --config configs/whisperrt_baseline.yaml \
    --max-samples 10
```

If the existing configuration system supports overrides, use that instead.

The script must:

1. load configuration
2. initialize logging
3. load dataset
4. load model
5. initialize streaming pipeline
6. process each utterance
7. finalize each utterance
8. collect predictions
9. calculate metrics
10. save outputs
11. save experiment metadata
12. generate a concise report

---

# 23. Baseline Configuration

Update:

```text
configs/whisperrt_baseline.yaml
```

with real Phase 1 settings.

Example structure:

```yaml
experiment:
  name: whisperrt_streaming_librispeech_baseline
  seed: 42

runtime:
  environment: auto
  device: auto

data:
  backend: huggingface
  dataset: librispeech
  split: test-clean
  max_samples: null

model:
  backend: huggingface
  name: MLSpeech/WhisperRT-Streaming
  dtype: auto

audio:
  sample_rate: 16000
  channels: 1
  chunk_ms: 320

streaming:
  enabled: true

evaluation:
  metrics:
    - wer
    - rtf
    - first_token_latency
    - end_of_utterance_latency

output:
  save_predictions: true
  save_metrics: true
  save_report: true
```

Adapt field names to the Phase 0 configuration system.

Do not blindly copy this example if the actual WhisperRT API requires different settings.

---

# 24. Baseline Evaluation

At minimum calculate:

## Recognition

### WER

Word Error Rate:

```text
WER = (S + D + I) / N
```

where:

* S = substitutions
* D = deletions
* I = insertions
* N = reference word count

Use a reliable implementation rather than writing a fragile custom implementation unless there is a strong reason.

Document:

* text normalization
* punctuation handling
* case handling
* number normalization if any

The same normalization must be applied consistently.

---

# 25. Text Normalization

Implement a deterministic normalization function.

At minimum consider:

* lowercase
* punctuation normalization/removal
* whitespace normalization

Do not perform aggressive linguistic normalization that changes the meaning.

Do not silently normalize numbers, contractions, or special tokens unless documented.

Store both:

```text
raw_reference
normalized_reference
raw_prediction
normalized_prediction
```

where practical.

---

# 26. Streaming Metrics

Measure at minimum:

### Real-Time Factor

```text
RTF = processing_time / audio_duration
```

Interpretation:

```text
RTF < 1
```

means processing is faster than real-time under the tested conditions.

Do not describe this as a universal property of the model.

It is an experimental measurement under a particular:

* hardware
* batch size
* chunk size
* dtype
* software environment

---

# 27. Latency Metrics

Implement measurements for:

### First-token latency

Time from the relevant audio arrival/start point until the first meaningful recognized output.

### End-of-utterance latency

Time between the point at which the final required audio has been supplied and the finalized transcript becomes available.

### Chunk processing latency

Processing time for each streaming chunk.

Where precise token-level timing is unavailable from the model API, clearly state the measurement definition.

Do not fabricate token timestamps.

---

# 28. Latency Measurement Methodology

Clearly distinguish:

```text
audio time
wall-clock processing time
model inference time
queue/scheduling time
finalization time
```

If only wall-clock measurements are available, report wall-clock measurements.

Use a monotonic clock such as:

```python
time.perf_counter()
```

for timing.

Do not use system wall-clock timestamps for duration calculations.

---

# 29. Warm-up

GPU inference may have initialization overhead.

Support an optional warm-up stage.

Do not mix model loading time into normal streaming inference latency unless explicitly reported as a separate metric.

Report separately:

```text
model_load_time
warmup_time
streaming_processing_time
finalization_time
```

if available.

---

# 30. Batch Size

The streaming baseline should conceptually operate with:

```text
batch_size = 1
```

per audio stream.

Do not batch unrelated utterances in a way that invalidates streaming latency measurements.

If batching is supported for dataset throughput, distinguish that benchmark from the true streaming benchmark.

---

# 31. Single-Utterance Streaming Evaluation

The primary Phase 1 benchmark should process each LibriSpeech utterance independently:

```text
reset stream
    ↓
feed chunks sequentially
    ↓
finalize
    ↓
compare with reference
```

Do not allow transcript/state leakage between utterances.

This is a critical test.

---

# 32. Streaming State Reset Test

Add a test ensuring that:

```text
utterance A
```

does not affect:

```text
utterance B
```

The second utterance must start from a clean streaming state.

This should be tested both at the pipeline and model-adapter levels where practical.

---

# 33. Phase 1 Tests

Extend the existing test suite.

At minimum:

```text
tests/test_audio.py
tests/test_vad.py
tests/test_streaming.py
tests/test_overlap.py
tests/test_pipeline.py
tests/test_metrics.py
```

Do not remove Phase 0 tests.

Add Phase 1-specific tests where appropriate.

---

# 34. WhisperRT Tests

Create tests that do not require downloading the real model for every unit test.

Use mocks/stubs where appropriate.

Test:

* model adapter interface
* initialization configuration
* device selection
* state lifecycle
* chunk processing contract
* finalization
* transcript accumulation
* repeated partial hypothesis handling

The full real model should be tested separately as an integration test.

Do not make the entire unit-test suite depend on a large model download.

---

# 35. Real Model Integration Test

Create a lightweight integration test or validation path that can run the actual model on a very small audio example.

It may be:

```text
manual integration test
```

or:

```text
pytest -m integration
```

depending on the architecture.

Do not require the real model in ordinary:

```bash
pytest -q
```

unless the project environment explicitly supports this.

---

# 36. Smoke Test Evolution

Update:

```text
scripts/smoke_test.py
```

The smoke test must remain fast and must not unexpectedly download the full model.

Phase 1 should introduce an optional real-model smoke test.

For example:

```bash
python scripts/smoke_test.py --phase 0
```

should remain lightweight.

A separate command may exist for:

```bash
python scripts/smoke_test.py --phase 1 --real-model
```

or an equivalent mechanism.

The real-model smoke test should use:

* a tiny audio sample
* one utterance
* minimal output

Do not make the default smoke test dependent on network/model availability.

---

# 37. Kaggle Execution

Phase 1 must be designed to run on Kaggle.

The Kaggle workflow should support:

```text
Kaggle Notebook
      ↓
Repository
      ↓
Hugging Face model
      ↓
LibriSpeech dataset
      ↓
Streaming inference
      ↓
Metrics
      ↓
/kaggle/working outputs
```

Do not assume Internet access is enabled.

Document both:

### Internet-enabled workflow

Download model/dataset through Hugging Face.

### Attached-input workflow

Use Kaggle dataset/model inputs when available.

---

# 38. No Large Files in Git

Do not commit:

* WhisperRT weights
* LibriSpeech
* generated audio
* prediction dumps
* model caches
* large benchmark outputs

Use configuration to point to external data/model locations.

---

# 39. Experiment Output Structure

Use the Phase 0 run-directory mechanism.

A Phase 1 run should produce something similar to:

```text
outputs/
├── logs/
│   └── <run_id>/
│       ├── run.log
│       ├── config.yaml
│       └── metadata.json
│
├── predictions/
│   └── <run_id>.jsonl
│
├── metrics/
│   └── <run_id>.json
│
└── reports/
    └── <run_id>.md
```

The exact layout may follow the existing Phase 0 implementation.

---

# 40. Prediction Format

Save per-utterance results.

A useful JSONL record should contain fields such as:

```json
{
  "sample_id": "...",
  "reference": "...",
  "prediction": "...",
  "audio_duration": 12.34,
  "processing_time": 4.56,
  "rtf": 0.37,
  "first_output_latency": 0.82,
  "finalization_latency": 0.21
}
```

Only include fields that were actually measured.

Do not insert fake values.

---

# 41. Benchmark Report

Generate a Phase 1 report containing:

## Dataset

* dataset name
* split
* number of samples
* total audio duration

## Model

* model identifier
* model revision if known
* device
* dtype
* relevant generation configuration

## Streaming

* chunk size
* sample rate
* stream mode
* batch size

## Metrics

* WER
* RTF
* first-output latency
* end-of-utterance latency
* average chunk processing time

## Environment

* Python
* OS
* GPU
* CUDA
* relevant package versions

## Limitations

Explicitly document anything that was not measurable.

---

# 42. Chunk Size Experiment

Phase 1 should support changing chunk size through configuration.

At minimum make this configurable:

```yaml
audio:
  chunk_ms: 320
```

Do not automatically conduct a large experiment grid.

However, the implementation should make it easy to run:

```text
160 ms
320 ms
640 ms
```

or other valid values later.

If a small chunk-size comparison is executed during Phase 1, report it as an experiment rather than silently selecting a winner.

Do not rank chunk sizes unless the user later requests comparative analysis and the experimental evidence supports the comparison.

---

# 43. Baseline Experiment Design

The minimum baseline experiment is:

```text
Model:
MLSpeech/WhisperRT-Streaming

Dataset:
LibriSpeech test-clean

Language:
English

Speakers:
single-speaker utterances

Streaming:
enabled

Chunk:
configurable

VAD:
not separately implemented

OSD:
not implemented

Overlap:
none

WER:
measured

RTF:
measured

Latency:
measured
```

This experiment establishes the clean speech reference.

---

# 44. Important Baseline Limitation

Explicitly document:

> Phase 1 evaluates clean single-speaker speech. It does not establish performance under overlapping speech.

This is important because later phases will compare the baseline against overlap conditions.

Do not make claims about overlap robustness based on Phase 1.

---

# 45. README Update

Update `README.md` with a new section:

```text
## Phase 1 — WhisperRT Streaming Baseline
```

Include:

* objective
* model
* dataset
* architecture
* installation
* model loading
* dataset loading
* configuration
* commands
* Kaggle workflow
* expected outputs
* metrics
* benchmark methodology
* limitations
* current status

Clearly mark measured results only after actually running the benchmark.

---

# 46. Documentation Updates

Update:

```text
docs/architecture.md
docs/datasets.md
docs/experiments.md
docs/thesis_mapping.md
```

### architecture.md

Document:

```text
Audio → Chunker → WhisperRT Streaming Adapter → Transcript
```

### datasets.md

Document the actual LibriSpeech subset used.

### experiments.md

Document:

* experiment configuration
* timing methodology
* WER normalization
* output structure

### thesis_mapping.md

Explain that Phase 1 establishes:

> a clean single-speaker streaming ASR baseline against which later overlap-aware methods can be evaluated.

Do not claim novelty.

---

# 47. Notebook 02 — WhisperRT Baseline

Implement:

```text
notebooks/02_whisperrt_baseline.ipynb
```

The notebook must be independently understandable.

It should contain sections:

1. Environment
2. Configuration
3. Model loading
4. Dataset loading
5. Single utterance test
6. Streaming chunk demonstration
7. Streaming inference
8. Transcript output
9. WER calculation
10. latency measurement
11. RTF measurement
12. benchmark summary
13. artifact locations
14. limitations

The notebook should use project code.

Do not duplicate the entire implementation in notebook cells.

---

# 48. Notebook Development Mode

The notebook should support a small development run such as:

```yaml
data:
  max_samples: 3
```

or equivalent.

This allows the user to verify the complete pipeline before running the full test-clean benchmark.

---

# 49. Full Benchmark Mode

The notebook should also make it clear how to run the full configured benchmark.

Do not automatically run the full dataset merely because the notebook is opened.

Use an explicit configuration variable or configuration file.

---

# 50. Reproducibility

Record:

* random seed
* model name
* dataset name
* dataset split
* model revision if available
* chunk size
* device
* dtype
* Python version
* package versions
* hardware

The benchmark should be reproducible as far as the underlying model/framework allows.

---

# 51. Determinism

Where deterministic behavior is supported, configure it.

However, do not claim exact deterministic inference if the hardware/framework does not guarantee it.

Document limitations.

---

# 52. Model Cache

Support normal Hugging Face caching.

Do not copy model weights into the repository.

On Kaggle, document that model caching may depend on the notebook environment and Internet availability.

---

# 53. Performance Considerations

Do not optimize prematurely.

However:

* avoid reloading the model for each utterance
* avoid unnecessary resampling
* avoid copying complete accumulated audio repeatedly
* use inference mode/no-gradient mode where appropriate
* keep batch size appropriate for streaming
* release temporary resources correctly

Use the inference context recommended by the actual model implementation.

---

# 54. Memory Measurement

If practical, report GPU memory usage.

For CUDA, measurements may include:

```text
allocated memory
reserved memory
peak memory
```

Do not make GPU memory a mandatory metric if the environment does not expose it.

For CPU-only runs, report that GPU memory is not applicable.

---

# 55. Error Handling

The implementation should provide clear errors for:

* missing model
* invalid model backend
* unavailable CUDA
* unsupported dtype
* missing dataset
* malformed audio
* unsupported sample rate
* incompatible WhisperRT API
* missing optional dependency

Do not silently fall back to offline Whisper.

This is especially important.

---

# 56. No Offline Whisper Fallback

Do NOT implement:

```text
if WhisperRT fails:
    use normal Whisper
```

This would invalidate the baseline.

If WhisperRT cannot be loaded:

```text
FAIL CLEARLY
```

and explain the dependency/configuration problem.

A fallback may be introduced in a future experiment only if explicitly requested and clearly labeled as a different baseline.

---

# 57. Validation Protocol

After implementation, run:

## A. Phase 0 regression

```bash
pytest -q
```

## B. Environment

```bash
python scripts/check_environment.py
```

## C. Phase 1 development run

Run the smallest real WhisperRT + LibriSpeech experiment.

Use:

```text
1–3 utterances
```

or another minimal configured subset.

Verify:

* model loads
* audio loads
* streaming works
* transcript is produced
* finalization works
* metrics are calculated
* artifacts are saved

## D. Full baseline

Run the configured LibriSpeech test-clean benchmark if hardware/time permits.

Do not claim completion if only the development run was executed.

---

# 58. Required Evidence

The Phase 1 report must distinguish between:

```text
IMPLEMENTED
TESTED
MEASURED
NOT MEASURED
```

For example:

```text
WhisperRT integration: IMPLEMENTED + TESTED
Full test-clean WER: NOT MEASURED
3-sample WER: MEASURED
CUDA benchmark: NOT MEASURED
```

Do not infer full-dataset performance from a tiny development run.

---

# 59. Phase 1 Success Criteria

Phase 1 is complete only when:

* [ ] Phase 0 remains functional.
* [ ] Actual `MLSpeech/WhisperRT-Streaming` is integrated.
* [ ] Model loading uses the project backend abstraction.
* [ ] Hugging Face loading works.
* [ ] Device selection works.
* [ ] Dtype configuration works where supported.
* [ ] LibriSpeech loader works.
* [ ] English audio preprocessing works.
* [ ] Streaming chunking works.
* [ ] WhisperRT state persists across chunks.
* [ ] Stream finalization works.
* [ ] Transcript accumulation does not duplicate partial hypotheses.
* [ ] Utterance state resets correctly.
* [ ] No offline Whisper fallback exists.
* [ ] WER is calculated.
* [ ] Text normalization is documented.
* [ ] RTF is calculated.
* [ ] latency measurement is implemented.
* [ ] model loading time is separated from inference timing.
* [ ] predictions are saved.
* [ ] metrics are saved.
* [ ] experiment configuration is saved.
* [ ] Phase 1 report is generated.
* [ ] `02_whisperrt_baseline.ipynb` exists and is valid.
* [ ] unit tests pass.
* [ ] a real WhisperRT integration run succeeds.
* [ ] at least one real LibriSpeech utterance has been transcribed.
* [ ] actual measured results are documented.
* [ ] README is updated.
* [ ] documentation is updated.
* [ ] no large model/data files are committed.
* [ ] no credentials are committed.
* [ ] VAD/OSD/overlap/multi-talker functionality has NOT been implemented.

---

# 60. Phase 1 Completion Report

After implementation, produce a report with exactly these sections:

## 1. Implementation Summary

What was implemented.

## 2. Files Created/Modified

List important files.

## 3. WhisperRT Integration

Explain:

* actual API used
* model loading mechanism
* processor/tokenizer
* streaming state handling
* finalization behavior

## 4. Dataset

Report:

* LibriSpeech subset
* backend
* sample count actually evaluated
* total duration actually evaluated

## 5. Experimental Configuration

Report:

* chunk size
* sample rate
* device
* dtype
* batch size
* generation settings

## 6. Measured Results

Use a table:

| Metric                   | Value | Dataset subset | Hardware | Notes |
| ------------------------ | ----: | -------------- | -------- | ----- |
| WER                      |   ... | ...            | ...      | ...   |
| RTF                      |   ... | ...            | ...      | ...   |
| First-output latency     |   ... | ...            | ...      | ...   |
| End-of-utterance latency |   ... | ...            | ...      | ...   |
| Avg. chunk latency       |   ... | ...            | ...      | ...   |

If a metric was not measured:

```text
NOT MEASURED
```

Do not invent a value.

## 7. Tests

Report:

```text
pytest: PASS/FAIL
Phase 0 smoke test: PASS/FAIL
WhisperRT integration: PASS/FAIL
LibriSpeech development run: PASS/FAIL
Full test-clean benchmark: PASS/FAIL/NOT RUN
Kaggle validation: PASS/FAIL/NOT RUN
```

## 8. Problems Encountered

Document actual issues.

## 9. Limitations

Especially:

* clean speech only
* no overlap
* no VAD
* no OSD
* no multi-talker recognition

## 10. Phase 2 Readiness

Explain whether the project is ready for:

> **Phase 2 — Streaming VAD**

Do not implement Phase 2.

---

# 61. Research Integrity Rules

These rules are mandatory.

Never fabricate:

* WER
* RTF
* latency
* throughput
* memory
* dataset statistics
* model capabilities
* benchmark results

If something was not measured, write:

```text
NOT MEASURED
```

If something is only a planned design:

```text
PLANNED
```

If something depends on a future research decision:

```text
RESEARCH DECISION PENDING
```

If a number comes from external documentation rather than the experiment, clearly label it as an externally documented value and cite its source in the documentation.

---

# 62. Important Scientific Interpretation Rule

Do not make claims such as:

> WhisperRT is superior to Whisper.

or:

> This proves the model is real-time.

Instead report the measured result under the tested conditions.

For example:

> On the tested hardware and configuration, the measured RTF was X.

This distinction is important for thesis-quality experimentation.

---

# 63. Final Scope Boundary

At the end of Phase 1 the architecture should conceptually be:

```text
                 PHASE 1
                    │
                    ▼
              Audio Stream
                    │
                    ▼
             Audio Buffer
                    │
                    ▼
           Streaming Chunker
                    │
                    ▼
       WhisperRT-Streaming ASR
                    │
                    ▼
        Transcript Accumulator
                    │
                    ▼
               Evaluation
```

The following remain future work:

```text
Streaming VAD
     ↓
Overlap Speech Detection
     ↓
Overlap-aware routing
     ↓
Multi-talker ASR
```

Do not implement those components in Phase 1.

---

# 64. Final Instruction

Implement **ONLY Phase 1**.

The central deliverable is a **real, reproducible, measured WhisperRT streaming baseline on English LibriSpeech**.

Do not replace WhisperRT with offline Whisper.

Do not introduce VAD.

Do not introduce overlap detection.

Do not introduce multi-talker recognition.

Do not fine-tune the model.

Do not fabricate results.

Preserve the Phase 0 architecture and tests.

After the implementation and validation are complete, stop and provide the Phase 1 completion report.

The next phase will be:

> **PHASE 2 — Streaming VAD**

and must be treated as a separate implementation step.
