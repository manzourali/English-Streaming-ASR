# PHASE 3 — Synthetic Overlap Dataset Generation

You are implementing **Phase 3** of the English Streaming ASR with Overlapped Voices research project.

The project operates under the previously defined **Master Prompt / project contract**. Follow all architectural, reproducibility, testing, Kaggle/Hugging Face, documentation, and research-integrity requirements from that contract.

## Current Project State

Phases 0–2 have been specified as:

* **Phase 0:** project infrastructure and repository skeleton
* **Phase 1:** `MLSpeech/WhisperRT-Streaming` streaming ASR baseline
* **Phase 2:** streaming VAD

Phase 3 now introduces the **synthetic overlap data generation layer**.

The intended dataset progression is:

```text
LibriSpeech
    ↓
Synthetic Overlap
    ↓
LibriSpeechMix
    ↓
LibriCSS
    ↓
AMI / CHiME-6 (if needed)
```

Do not implement later phases prematurely.

---

# 1. Phase Objective

Build a reproducible system that takes clean English single-speaker utterances and generates **synthetic overlapping speech mixtures**.

The generated data must preserve enough metadata to know:

* which source utterances were mixed
* which speaker/source contributed to each mixture
* when each source starts
* when each source ends
* the overlap intervals
* the amount of overlap
* the mixture/reference audio
* the individual source references
* the corresponding transcripts

The primary purpose is to create controlled data for later experiments involving:

1. WhisperRT under overlapping speech
2. overlap detection
3. adaptive routing
4. multi-talker recognition

Phase 3 itself must **not** implement those systems.

---

# 2. Research Question for This Phase

The main question is:

> Can we construct a reproducible and controllable synthetic English speech-overlap dataset from LibriSpeech that provides reliable ground-truth timing and transcript metadata for later streaming overlap experiments?

The generated dataset should allow future experiments to vary:

* overlap ratio
* relative source timing
* number of speakers
* signal-to-signal ratio / relative gain
* utterance duration
* amount of silence
* mixture duration

Do not assume in advance which overlap configuration will be optimal for the thesis.

---

# 3. Scope

Implement only the following:

### Required

* LibriSpeech input loader
* source utterance selection
* deterministic pairing/selection
* audio normalization/resampling where necessary
* synthetic mixture generation
* controlled temporal offsets
* controlled overlap duration/ratio
* optional relative gain/SNR control
* mixture metadata
* source/reference metadata
* manifest generation
* train/validation/test organization
* reproducibility controls
* dataset validation
* overlap statistics
* audio sanity checks
* tests
* configuration
* Kaggle notebook
* README/documentation updates
* phase report

### Explicitly NOT in scope

Do **not** implement:

* OSD
* overlap detector
* diarization
* speaker identification
* speaker embeddings
* speaker enrollment
* multi-talker ASR
* speech separation
* source separation model
* WhisperRT modifications
* WhisperRT fine-tuning
* ASR training
* PEFT
* reinforcement learning
* adaptive routing
* new VAD models

Those belong to later phases.

---

# 4. Important Conceptual Distinction

Synthetic overlap generation is a **data-generation problem**, not an overlap-detection problem.

The generator knows the ground truth because it creates the mixture.

For example:

```text
Speaker A:
|---------speech---------|
0s                       4s

Speaker B:
        |---------speech---------|
        1s                       5s

Mixture:
|-----------------------------|
0s                            5s

Overlap:
        |-----3s------|
        1s             4s
```

The generator must record this timing information explicitly.

Do not infer overlap labels later from the mixed waveform when the generator already knows the source timing.

---

# 5. Source Dataset

Use **LibriSpeech** as the primary source dataset.

Prefer:

* `train-clean-100` for generation/training experiments where practical
* `dev-clean` for validation
* `test-clean` for evaluation

However, do not unnecessarily duplicate the complete LibriSpeech dataset into the repository.

The system must support:

### Hugging Face

Loading through the Hugging Face dataset ecosystem.

### Kaggle

Loading from `/kaggle/input/...` or an equivalent configurable Kaggle dataset path.

### Local

Loading from a configurable local dataset path.

Do not hard-code a personal filesystem path.

---

# 6. Speaker Separation Requirement

When creating a two-speaker mixture, prefer selecting utterances from **different speakers**.

Do not generate synthetic overlap by simply mixing two utterances from the same speaker unless there is an explicit experimental reason and it is separately labeled.

The metadata should contain source speaker identifiers where the source dataset provides them.

For example:

```json
{
  "speaker_a": "...",
  "speaker_b": "..."
}
```

The exact LibriSpeech speaker-ID representation should be determined from the actual dataset/API being used rather than assumed.

---

# 7. Dataset Generator Design

Create a reusable generator rather than putting generation logic directly inside a notebook.

The notebook should call the project code.

A suitable abstraction is:

```python
class OverlapGenerator:
    def generate(...):
        ...
```

You may introduce smaller abstractions if they improve maintainability.

For example:

```text
SourceDataset
      ↓
UtteranceSelector
      ↓
OverlapScheduler
      ↓
AudioMixer
      ↓
ManifestWriter
      ↓
DatasetValidator
```

Do not over-engineer this.

---

# 8. Mixture Types

Initially support **two-speaker mixtures**.

Do not implement arbitrary N-speaker mixtures in this phase unless the architecture makes it essentially trivial.

The first implementation should be reliable for:

```text
Speaker A + Speaker B
```

The architecture may leave room for future extension to:

```text
Speaker A + Speaker B + Speaker C
```

but Phase 3 should focus on the two-speaker case.

---

# 9. Overlap Control

The generator must support controlled overlap.

Do not use only random offsets with no knowledge of the resulting overlap.

The configuration should allow specifying a target overlap regime.

Possible approaches include:

```yaml
overlap:
  mode: ratio
  target_ratio: 0.5
```

or a duration-based configuration.

The implementation may choose the most robust formulation after inspecting the source audio durations.

The important requirement is:

> The resulting overlap must be measurable from the generated source timing metadata.

For each mixture calculate actual values such as:

```text
source_a_duration
source_b_duration
overlap_duration
mixture_duration
overlap_ratio
```

Define the exact overlap-ratio formula clearly in the documentation.

Do not silently use multiple incompatible definitions.

---

# 10. Multiple Overlap Conditions

Create configurable overlap conditions rather than generating only one fixed overlap.

At minimum, support several controlled regimes such as:

```text
low overlap
medium overlap
high overlap
```

The exact numerical ranges should be configurable.

For example, the configuration could contain:

```yaml
overlap:
  regimes:
    low:
      ...
    medium:
      ...
    high:
      ...
```

Do not hard-code the example values above unless justified by the implementation.

The generator must record the actual overlap value for every generated example.

---

# 11. Temporal Placement

Support controlled temporal placement.

At minimum, the generator should be capable of creating:

### Partial overlap

```text
A: |-------------|
B:       |-------------|
```

### Strong overlap

```text
A: |-------------|
B:   |-------------|
```

### Minimal/near-zero overlap

```text
A: |-------------|

B:              |-------------|
```

The last case may be useful as a negative/control condition.

However, do not redefine non-overlap as an overlap sample.

Clearly label:

```text
overlap = true
overlap = false
```

or an equivalent ground-truth representation.

---

# 12. Relative Gain / SNR

Support configurable relative source gain.

For example:

```text
Speaker A: 0 dB
Speaker B: -3 dB
Speaker B: -6 dB
```

The exact experimental values should be configurable.

Do not assume that equal-energy mixing is representative of all real-world overlap.

However, keep Phase 3 focused on generating controlled mixtures rather than attempting to model every acoustic condition.

---

# 13. Audio Processing

All generated audio should have a clearly defined format.

Use the project's standard:

```text
sample rate: 16 kHz
```

Ensure:

* consistent sampling rate
* consistent channel format
* numerical stability
* no accidental clipping
* appropriate waveform normalization
* deterministic processing where applicable

Do not destroy the relative amplitude relationship between speakers through inappropriate per-source normalization immediately before mixing.

If normalization is necessary, document exactly where and why it occurs.

---

# 14. Clipping and Amplitude Handling

The mixer must explicitly handle amplitude overflow/clipping.

Do not simply add waveforms and write the result without checking amplitude.

Implement an appropriate strategy such as:

* controlled gain
* peak normalization after mixing
* headroom
* another technically justified method

The chosen method must be documented.

Be careful that post-mix normalization does not invalidate the intended relative source gain/SNR condition.

---

# 15. Ground-Truth Metadata

Every generated mixture must have machine-readable metadata.

A record should contain information equivalent to:

```json
{
  "mixture_id": "...",
  "audio_path": "...",

  "sample_rate": 16000,
  "duration": 5.21,

  "num_speakers": 2,

  "sources": [
    {
      "speaker_id": "...",
      "source_id": "...",
      "audio_path": "...",
      "start": 0.0,
      "end": 4.2,
      "duration": 4.2,
      "transcript": "..."
    },
    {
      "speaker_id": "...",
      "source_id": "...",
      "audio_path": "...",
      "start": 1.1,
      "end": 5.2,
      "duration": 4.1,
      "transcript": "..."
    }
  ],

  "overlap": {
    "exists": true,
    "start": 1.1,
    "end": 4.2,
    "duration": 3.1,
    "ratio": 0.74
  },

  "mix": {
    "relative_gain_db": 0.0
  }
}
```

This is illustrative.

Adapt the schema to the actual implementation.

The important point is that future phases must be able to recover exact source timing and transcript references without reverse engineering the waveform.

---

# 16. Manifest Format

Use a machine-readable manifest.

JSONL is preferred for individual mixture records because it is easy to stream and process.

You may additionally create CSV summaries where useful.

For example:

```text
data/manifests/
    train.jsonl
    validation.jsonl
    test.jsonl
```

or an equivalent structure.

Document the schema.

---

# 17. Train / Validation / Test Leakage

Avoid speaker leakage where practical.

If the source dataset provides speaker identities, ensure that the generated dataset split strategy does not accidentally place the same source speaker in both training and test sets.

Prefer:

```text
train speakers
    ≠
validation speakers
    ≠
test speakers
```

if feasible with the selected LibriSpeech subsets.

This is particularly important because the later research concerns multi-speaker/overlap robustness.

Document the actual split strategy.

Do not claim speaker-disjointness unless the implementation verifies it.

---

# 18. Deterministic Generation

The generator must support a random seed.

For example:

```yaml
experiment:
  seed: 42
```

Running the same generation configuration with the same seed should produce reproducible selection and timing decisions, subject to documented audio-library/platform limitations.

Do not use uncontrolled global randomness.

---

# 19. Dataset Size

Do not attempt to generate a massive dataset by default.

The default configuration should generate a **small development dataset** suitable for:

* Kaggle
* CI/smoke tests
* debugging
* quick experimentation

Provide configuration parameters such as:

```yaml
generation:
  num_samples: ...
```

The full-scale generation should be configurable separately.

The repository must never contain the generated large dataset.

---

# 20. Development / Smoke Dataset

Create a very small deterministic dataset suitable for testing.

It should be possible to run something conceptually similar to:

```bash
python scripts/generate_overlap.py --config configs/overlap_dataset.yaml
```

and produce a small set of mixtures.

The exact CLI can differ if the project's existing CLI conventions require another form.

The output should be predictable enough for automated validation.

---

# 21. Dataset Validation

Implement validation utilities.

The validator should check at least:

### Audio

* file exists
* audio can be loaded
* sample rate is correct
* expected channel format
* finite numeric values
* no invalid waveform values
* duration is positive

### Metadata

* required fields exist
* source files exist where expected
* source timing is valid
* `start < end`
* source duration agrees reasonably with metadata
* mixture duration is consistent
* overlap interval is valid

### Overlap

Verify that the recorded overlap agrees mathematically with the source timing.

For example:

```text
overlap_start = max(start_A, start_B)
overlap_end   = min(end_A, end_B)
```

if:

```text
overlap_end > overlap_start
```

then:

```text
overlap_duration = overlap_end - overlap_start
```

Do not trust metadata generated by the same function without independently checking it.

---

# 22. Dataset Statistics

Create a statistics/reporting function.

At minimum report:

* number of mixtures
* total duration
* mean duration
* min/max duration
* number of overlapping examples
* number of non-overlapping/control examples
* mean overlap duration
* overlap-ratio distribution
* speaker count
* relative gain/SNR distribution if implemented

The report should help determine whether the generated dataset actually matches the requested configuration.

Do not fabricate statistics.

All reported statistics must be computed from generated data.

---

# 23. Visualization

Add lightweight visualization where useful.

The notebook should be able to show examples such as:

```text
Speaker A activity
Speaker B activity
Overlap region
Mixture waveform
```

A simple timeline visualization is sufficient.

Do not build a sophisticated GUI.

---

# 24. Dataset Quality Checks

Create several automatic checks, for example:

```text
✓ all files readable
✓ all sample rates correct
✓ no NaN/Inf
✓ metadata valid
✓ source timing valid
✓ overlap timing valid
✓ expected number of examples generated
✓ expected overlap regimes represented
✓ no unintended speaker leakage
```

The checks should fail clearly when an invariant is violated.

---

# 25. Integration With Phase 2

Phase 3 must integrate cleanly with the existing Phase 2 infrastructure.

The generated mixtures should be usable as audio input for the existing streaming pipeline.

Do not modify the VAD implementation to become an overlap detector.

The distinction remains:

```text
VAD:
speech vs non-speech

Phase 3:
generate single/overlapping speech with ground truth

Future OSD:
single-speaker vs overlap
```

---

# 26. Phase 2 Compatibility

Do not break:

* Phase 1 WhisperRT baseline
* Phase 2 streaming VAD
* existing tests
* existing configs
* existing smoke-test behavior

If an interface must change, preserve backward compatibility where practical and document the change.

---

# 27. Project Structure

Use the existing repository structure.

Likely affected files include:

```text
src/streaming_asr/
    datasets/
        loaders.py
        manifests.py
        overlap_generator.py
        validators.py

configs/
    overlap_dataset.yaml

scripts/
    generate_overlap.py

notebooks/
    04_overlap_generation.ipynb

tests/
    test_overlap.py
    test_audio.py
    test_metrics.py   # only if necessary

docs/
    datasets.md
    experiments.md
    architecture.md

README.md
```

Do not create unnecessary modules merely to satisfy a theoretical architecture.

---

# 28. Configuration

Create:

```text
configs/overlap_dataset.yaml
```

It should expose the important experimental variables without requiring source-code edits.

At minimum consider:

```yaml
experiment:
  name: synthetic_overlap
  seed: 42

data:
  backend: ...
  source_dataset: librispeech
  source_split: ...

generation:
  num_samples: ...
  num_speakers: 2

audio:
  sample_rate: 16000

overlap:
  ...

mix:
  ...

output:
  ...
```

Use the project's existing configuration system from Phase 0 rather than creating a second configuration mechanism.

---

# 29. Kaggle Support

The Phase 3 notebook must be usable in Kaggle.

It should support both:

### Kaggle dataset input

```text
/kaggle/input/...
```

and, where permitted/configured:

### Hugging Face download

The notebook must not assume one specific acquisition method.

Clearly document:

* where the source dataset is expected
* how to configure its path
* whether Kaggle Internet is required
* how to run with a pre-attached dataset
* where generated outputs are written

Do not download a huge dataset automatically merely because the notebook starts.

---

# 30. Hugging Face Support

If the source dataset is loaded through Hugging Face, use the project's dataset backend abstraction.

Do not bypass it with unrelated dataset-loading code inside the notebook.

The notebook should demonstrate the supported backend rather than create a separate implementation.

---

# 31. Notebook

Create:

```text
notebooks/04_overlap_generation.ipynb
```

The notebook should be an executable demonstration of the Phase 3 pipeline.

Suggested flow:

```text
1. Environment check
2. Load configuration
3. Load small source subset
4. Inspect source examples
5. Generate synthetic mixtures
6. Inspect metadata
7. Play/inspect mixture examples
8. Visualize timing/overlap
9. Validate generated dataset
10. Compute statistics
11. Save manifest
12. Demonstrate output structure
```

Keep the notebook concise.

The real implementation belongs in `src/`.

---

# 32. Tests

Extend the existing test suite.

At minimum add tests covering:

### Generator

* deterministic generation
* valid two-speaker mixture
* expected output files
* controlled timing

### Overlap calculation

Test cases such as:

```text
no overlap
partial overlap
complete containment
different durations
```

### Metadata

* schema validity
* source timing
* overlap timing
* transcript preservation

### Audio

* sample rate
* finite values
* no invalid waveform output

### Dataset split

* no unintended speaker leakage

Tests must not require downloading a large dataset.

Use small synthetic fixtures or tiny/local test audio where possible.

---

# 33. Smoke Test

Update:

```text
scripts/smoke_test.py
```

so that the project remains incrementally testable.

The smoke test should be able to execute the smallest practical version of:

```text
source fixture
    ↓
overlap generator
    ↓
generated mixture
    ↓
metadata
    ↓
validation
```

Do not make the default smoke test depend on a large external dataset.

If real LibriSpeech access is necessary, keep it as an optional integration test rather than a mandatory unit/smoke dependency.

---

# 34. Metrics

Phase 3 is not primarily an ASR evaluation phase.

Do not report WER as a Phase 3 result unless you are merely verifying compatibility with an existing pipeline.

The primary Phase 3 measurements are dataset properties:

* number of mixtures
* duration
* overlap duration
* overlap ratio
* overlap-regime distribution
* source-speaker distribution
* relative gain/SNR distribution
* generation time
* storage size

Mark anything not measured as:

```text
NOT MEASURED
```

Do not invent expected values.

---

# 35. Research Integrity

This phase is particularly important for avoiding misleading experimental results.

Do not claim that synthetic overlap represents real conversational overlap.

Synthetic mixtures are a controlled experimental environment.

The documentation must explicitly distinguish:

```text
Synthetic overlap
        vs
Real-world overlap
```

Real-world evaluation will be addressed later using datasets such as LibriCSS and potentially AMI/CHiME-6.

---

# 36. Important Design Question

Before implementing the mixer, inspect the actual source dataset structure and determine:

* how speaker IDs are represented
* how transcript information is exposed
* how audio is loaded
* whether source audio is already 16 kHz
* what split metadata is available
* how speaker-disjoint generation can be implemented

Do not assume these details.

Use the actual dataset/API available in the environment.

---

# 37. Avoid Premature Research Decisions

Do not decide in this phase:

* which OSD model will be used
* which multi-talker ASR architecture will be used
* whether source separation will be necessary
* whether WhisperRT will be fine-tuned
* whether a unified or modular final architecture will win
* final overlap ratios for the thesis
* final benchmark datasets

Phase 3 should provide a flexible experimental data layer that allows those decisions later.

---

# 38. Documentation

Update:

```text
README.md
docs/datasets.md
docs/experiments.md
docs/architecture.md
```

Document:

* purpose of synthetic overlap
* source dataset
* generation process
* overlap definition
* metadata schema
* configuration
* reproducibility
* train/validation/test split
* limitations
* how to generate data
* how to validate data
* how to inspect examples
* output directory structure

Add a clear statement that this dataset is **synthetically generated** and should not be treated as a substitute for real conversational overlap data.

---

# 39. Phase Report

Create a concise Phase 3 report, for example:

```text
docs/phase_reports/phase_3_report.md
```

Include:

### Implemented

What was actually implemented.

### Files changed

List important files.

### Dataset

Actual source dataset and subset used.

### Generation configuration

Actual configuration used.

### Results

Actual measured statistics.

### Validation

Which checks passed.

### Tests

Test command and actual result.

### Smoke test

Command and actual result.

### Issues

Any technical issues encountered.

### Limitations

Known limitations of synthetic overlap.

### Research implications

What this phase enables for Phase 4.

### Next phase

Exact recommended command/prompt transition for Phase 4.

Never claim successful execution if you did not actually execute it.

---

# 40. Completion Criteria

Phase 3 is complete only when all of the following are true:

* [ ] Synthetic two-speaker overlap generation works.
* [ ] LibriSpeech can be used through the supported dataset backend.
* [ ] Kaggle/local/HF configuration is documented and works where applicable.
* [ ] Source speakers are tracked.
* [ ] Source timing is tracked.
* [ ] Overlap intervals are tracked.
* [ ] Overlap ratios are computed.
* [ ] Relative gain/SNR control is implemented or explicitly documented as not implemented.
* [ ] Generated audio is valid.
* [ ] Generated metadata is valid.
* [ ] Dataset validation works.
* [ ] Dataset statistics are generated from actual data.
* [ ] Train/validation/test leakage is checked.
* [ ] Deterministic generation is supported.
* [ ] Small smoke-test dataset works without a huge download.
* [ ] `notebooks/04_overlap_generation.ipynb` works.
* [ ] `tests/test_overlap.py` works.
* [ ] Existing Phase 1 and Phase 2 tests still pass.
* [ ] Existing smoke test still passes.
* [ ] README/docs are updated.
* [ ] Phase 3 report is written.
* [ ] No OSD/multi-talker/diarization implementation has been introduced prematurely.

---

# 41. Final Execution Rule

Do not implement Phase 4 functionality.

At the end of Phase 3, stop.

Report:

1. files created/modified
2. architecture implemented
3. commands executed
4. dataset/backend used
5. generated dataset statistics
6. validation results
7. test results
8. smoke-test result
9. notebook status
10. known issues
11. limitations
12. exact recommended next step

The next phase will use this synthetic dataset to establish the **WhisperRT-under-overlap baseline**.

Do not implement that baseline during Phase 3.
