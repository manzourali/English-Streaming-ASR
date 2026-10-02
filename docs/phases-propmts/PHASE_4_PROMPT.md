# PHASE 4 — WhisperRT Under Synthetic Overlap Baseline

You are implementing **Phase 4** of the English Streaming ASR with Overlapped Voices research project.

The project operates under the previously defined **Master Prompt / project contract**. Follow all architecture, reproducibility, testing, Kaggle/Hugging Face, documentation, and research-integrity requirements from that contract.

Do not redesign the overall project architecture unless an actual implementation problem requires it.

---

# 1. Current Project State

The project has progressed through:

### Phase 0

Project infrastructure, configuration, logging, backend abstractions, testing, smoke-test framework.

### Phase 1

`MLSpeech/WhisperRT-Streaming` streaming ASR baseline on clean English speech.

### Phase 2

Streaming VAD integrated as a separate component while preserving the WhisperRT baseline.

### Phase 3

Synthetic two-speaker overlap dataset generation using controlled mixtures with source timing and overlap ground truth.

Phase 4 now asks:

> **How does the existing streaming WhisperRT ASR system behave when the input contains overlapping speech?**

This phase is a **baseline/evaluation phase**.

It is not a model-improvement phase.

---

# 2. Main Objective

Evaluate the existing `MLSpeech/WhisperRT-Streaming` pipeline on the synthetic overlap dataset generated in Phase 3.

The goal is to establish a reproducible baseline for:

* clean speech
* non-overlapping speech/control mixtures
* low-overlap speech
* medium-overlap speech
* high-overlap speech
* different relative speaker levels, if supported by Phase 3
* different overlap durations

The results will provide the empirical basis for deciding what the later OSD/adaptive/multi-talker components actually need to address.

---

# 3. Core Principle

Do **not** attempt to solve overlap in this phase.

The baseline should answer:

> What does WhisperRT do when it receives overlapping speech without an overlap-aware mechanism?

Do not add:

* OSD
* speaker diarization
* speaker identification
* source separation
* multi-talker decoding
* overlap-aware decoding
* adaptive routing
* fine-tuning
* PEFT
* RL
* special overlap prompts
* post-processing intended to improve overlap recognition

If an existing component from an earlier phase is required for the baseline, use it without changing its behavior.

---

# 4. Baseline Comparison

The evaluation should compare at least:

```text
A. Clean single-speaker speech
B. Non-overlapping two-speaker control
C. Overlapping two-speaker speech
```

The exact available conditions depend on the Phase 3 dataset.

Where Phase 3 generated multiple overlap regimes, evaluate them separately.

For example:

```text
Clean
No overlap
Low overlap
Medium overlap
High overlap
```

Do not assume these labels exist. Read the actual Phase 3 metadata/configuration.

---

# 5. Two Important Baselines

Maintain two distinct conceptual baselines.

## Baseline A — Existing Clean-Speech Baseline

Use the Phase 1 streaming WhisperRT pipeline on the clean LibriSpeech evaluation data.

This establishes the reference performance.

Do not regenerate or alter the Phase 1 baseline unnecessarily.

## Baseline B — WhisperRT on Synthetic Mixtures

Feed the synthetic mixture waveform directly into the same streaming WhisperRT system.

Conceptually:

```text
Synthetic mixture
       ↓
Streaming audio chunks
       ↓
WhisperRT
       ↓
Incremental transcript
       ↓
Evaluation
```

No overlap-aware processing should be inserted between the audio stream and WhisperRT.

---

# 6. Streaming Requirement

The experiment must preserve the project's **true streaming** requirement.

Do not:

```text
load complete file
→ run offline ASR
→ pretend it was streaming
```

Instead:

```text
audio source
→ streaming chunks
→ stateful WhisperRT
→ incremental outputs
```

Use the same streaming architecture and chunking mechanism established in Phase 1.

If Phase 1 uses configurable chunk duration, expose that configuration rather than creating a second chunking implementation.

---

# 7. VAD Handling

Phase 2 introduced streaming VAD.

Do not automatically make VAD mandatory for this Phase 4 baseline.

The most important baseline should be:

```text
Audio
  ↓
WhisperRT
```

because this isolates the effect of overlap on the recognizer.

If useful, additionally evaluate:

```text
Audio
  ↓
Streaming VAD
  ↓
WhisperRT
```

but keep it as a **separate experimental condition**.

Do not mix the results together.

The experiment report must clearly state whether VAD was enabled for each run.

---

# 8. Ground-Truth Challenge

Synthetic overlap provides multiple valid reference concepts.

For a mixture:

```text
Speaker A transcript = "..."
Speaker B transcript = "..."
```

there may not be a single naturally ordered reference transcript equivalent to ordinary single-speaker ASR.

Therefore, do not blindly calculate ordinary WER between:

```text
WhisperRT output
vs
"A transcript + B transcript"
```

unless the ordering/reference construction is explicitly justified.

Instead, design the evaluation around the available ground truth.

---

# 9. Evaluation Levels

Implement evaluation at multiple levels.

## Level 1 — Clean ASR

Use ordinary:

```text
WER
```

for clean single-speaker speech.

## Level 2 — Control / Non-overlap

For two-speaker non-overlapping mixtures, evaluate the recognized transcript against an appropriately constructed reference.

Document the reference construction.

## Level 3 — Overlap

For overlapping mixtures, report metrics that are valid for the actual task.

At minimum provide:

* hypothesis transcript
* source transcripts
* timing metadata
* mixture condition
* overlap ratio
* overlap duration
* speaker/source count

Do not force a conventional single-reference WER metric if it is mathematically inappropriate.

If a suitable overlap-aware metric can be implemented reliably from the available references, use it and document it.

Otherwise mark the unsupported metric as:

```text
NOT MEASURED
```

rather than inventing a methodology.

---

# 10. Important Multi-Talker Evaluation Decision

Investigate how existing literature evaluates two-speaker/multi-talker ASR.

Potential metrics may include concepts such as:

* cpWER
* permutation-invariant WER
* source-level WER
* concatenated-reference WER
* other multi-talker metrics

However:

**Do not automatically implement all of them.**

Determine which metric is appropriate for the exact output format of WhisperRT in this phase.

WhisperRT may produce a single transcript stream rather than explicit speaker-separated transcripts.

If the model cannot produce speaker-separated hypotheses, do not pretend that source-level WER or cpWER directly represents its output without defining a valid mapping.

Document this limitation.

This is a research decision that may affect Phase 7.

---

# 11. Overlap-Specific Measurements

For every mixture, preserve and report the Phase 3 metadata:

* mixture ID
* speaker/source IDs
* source start/end
* source durations
* overlap start/end
* overlap duration
* overlap ratio
* relative gain/SNR where available

Then associate the ASR output with those conditions.

This enables analysis such as:

```text
WER vs overlap ratio
WER vs overlap duration
WER vs relative speaker level
```

Do not assume a relationship before measuring it.

---

# 12. Primary Metrics

At minimum evaluate:

### ASR quality

* WER where valid

### Streaming behavior

* Real-Time Factor (RTF)
* first-output latency
* end-of-utterance latency
* average processing time per chunk
* number of output updates, if meaningful

### System behavior

* memory usage where practical
* GPU utilization/resource information where practical
* inference duration
* audio duration

Use the same metric definitions established in Phase 1.

Do not create competing definitions.

---

# 13. Error Analysis

Phase 4 should not only produce one aggregate score.

Collect examples of failure modes.

Potential categories include:

* missing one speaker
* partially transcribing one speaker
* speaker dominance
* hallucinated words
* repeated words
* deletions
* insertions
* cross-speaker word ordering
* unstable incremental output
* delayed output
* failure under strong overlap
* degradation under unequal speaker levels

These are **analysis categories**, not assumptions.

Only report a category if actual examples support it.

---

# 14. Incremental Transcript Analysis

Because this is a streaming project, inspect the incremental outputs.

Do not evaluate only the final transcript.

Where the Phase 1 implementation exposes incremental hypotheses, record information such as:

```text
timestamp
partial hypothesis
final hypothesis
```

This can help determine whether overlap causes:

* unstable partial hypotheses
* excessive revisions
* delayed recognition
* repeated text
* missing text

Do not modify the decoder merely to make these outputs easier to analyze.

---

# 15. Experiment Matrix

Create a configurable experiment matrix.

For example:

```text
Clean
No overlap
Low overlap
Medium overlap
High overlap
```

combined, where available, with:

```text
relative speaker level
chunk size
VAD enabled/disabled
```

However, avoid an unnecessarily huge combinatorial experiment.

The default experiment should be small enough for Kaggle.

A larger sweep should be configurable.

---

# 16. Chunk Size Analysis

If the Phase 1 system supports multiple streaming chunk sizes, allow Phase 4 to compare a small number of configurations.

For example:

```text
small
medium
large
```

Use the project's actual configured values rather than inventing arbitrary defaults.

The purpose is to determine whether overlap degradation changes with streaming granularity.

Do not optimize chunk size in this phase.

This is an evaluation variable, not a tuning exercise.

---

# 17. Clean vs Overlap Comparison

Produce a direct comparison between:

```text
clean speech
vs
synthetic overlap
```

using compatible metrics.

The report should make it possible to determine empirically:

* whether recognition quality degrades
* how degradation changes with overlap
* whether latency changes
* whether incremental output becomes less stable
* whether speaker imbalance affects behavior

Do not summarize these as conclusions until the experiment has actually been run.

---

# 18. Per-Example Results

Store machine-readable per-example results.

A record should contain information similar to:

```json
{
  "mixture_id": "...",
  "condition": "medium_overlap",
  "overlap_ratio": 0.52,
  "overlap_duration": 2.1,
  "relative_gain_db": -3.0,
  "reference": {
    "speaker_a": "...",
    "speaker_b": "..."
  },
  "hypothesis": "...",
  "wer": null,
  "rtf": 0.0,
  "first_output_latency": 0.0,
  "end_of_utterance_latency": 0.0
}
```

Adapt this to the actual evaluation implementation.

Do not store meaningless `WER` values where the metric is not valid.

Use `null`, `NOT_MEASURED`, or the project's established representation.

---

# 19. Output Structure

Use the existing output structure.

For example:

```text
outputs/
├── predictions/
│   └── phase4/
├── metrics/
│   └── phase4/
├── figures/
│   └── phase4/
├── reports/
│   └── phase4/
└── logs/
```

Do not create a separate unrelated output system.

---

# 20. Experiment Configuration

Create:

```text
configs/overlap_baseline.yaml
```

It should control:

* source dataset
* generated overlap dataset
* model
* device
* dtype
* streaming chunk size
* VAD enabled/disabled
* evaluation subset
* conditions
* output path
* seed

Use the project's existing config infrastructure.

Do not hard-code experimental settings in Python.

---

# 21. Script

Create a script such as:

```text
scripts/run_overlap_baseline.py
```

or use the project's established naming conventions if another name is more appropriate.

It should support a configurable experiment.

Conceptually:

```bash
python scripts/run_overlap_baseline.py \
    --config configs/overlap_baseline.yaml
```

The command should:

1. load configuration
2. load model
3. load dataset manifest
4. run streaming inference
5. collect outputs
6. calculate valid metrics
7. save per-example results
8. save aggregate results
9. save logs
10. generate a concise report

---

# 22. Notebook

Create:

```text
notebooks/05_overlap_baseline.ipynb
```

The notebook should demonstrate the complete Phase 4 experiment.

Suggested flow:

```text
1. Environment check
2. Load configuration
3. Load Phase 3 manifest
4. Inspect overlap conditions
5. Load WhisperRT
6. Run a clean example
7. Run a non-overlap example
8. Run overlap examples
9. Display incremental/final hypotheses
10. Calculate valid metrics
11. Compare conditions
12. Plot relevant results
13. Inspect failure examples
14. Save results
```

Keep the notebook relatively small.

The actual implementation must remain in `src/` and `scripts/`.

---

# 23. Visualizations

Generate useful, non-excessive plots.

Potential plots:

### WER vs overlap ratio

Only if WER is valid for the evaluated condition.

### Latency vs overlap ratio

Where enough measurements exist.

### RTF by condition

For example:

```text
clean
no-overlap
low
medium
high
```

### Example timeline

Show:

```text
Speaker A
Speaker B
Overlap
WhisperRT output timing
```

Do not generate plots for metrics that were not actually measured.

---

# 24. Tests

Extend the test suite without requiring large model downloads.

Add tests for:

### Evaluation

* result schema
* aggregation
* condition grouping
* missing metric handling

### Reference handling

* clean reference
* two-source reference
* invalid reference cases
* overlap metadata association

### Streaming measurements

* latency calculation
* RTF calculation
* chunk timing

### Output persistence

* JSON/JSONL result writing
* reproducible result paths

Do not make unit tests depend on WhisperRT weights.

Keep actual model execution as a separate integration test.

---

# 25. Real Model Integration Test

Add or preserve a clearly separated integration test for:

```text
Phase 3 mixture
→ WhisperRT
→ hypothesis
→ evaluation
```

It may require:

* model download
* GPU
* internet
* Hugging Face access

It must not be part of the default lightweight unit-test suite unless the project's testing strategy explicitly supports that.

Document the requirement.

---

# 26. Kaggle Compatibility

The notebook and scripts must work in Kaggle with an appropriate environment.

Support:

* Kaggle-attached synthetic overlap dataset
* Kaggle-attached source/model artifacts where applicable
* Hugging Face model loading when Internet is enabled
* configurable paths

Do not require the user to modify Python source code to change dataset paths.

Do not include model weights or generated datasets in Git.

---

# 27. Hugging Face Model Handling

Use:

```text
MLSpeech/WhisperRT-Streaming
```

through the model abstraction established in Phase 0/1.

Do not create a second Whisper loading mechanism.

If the actual current model API differs from assumptions made in earlier phases, inspect the installed/current implementation and adapt the adapter rather than bypassing the architecture.

Do not silently substitute another Whisper model.

---

# 28. Reproducibility

Every experiment must record:

* experiment name
* timestamp
* seed
* model identifier/version where available
* dataset identifier/version or manifest
* configuration
* device
* dtype
* chunk size
* VAD status
* number of evaluated samples
* software/environment information where practical

Save the effective configuration with the results.

The result directory must contain enough information to understand how it was produced.

---

# 29. Statistical Reporting

Do not report only a single aggregate number.

For each condition report at least:

* number of examples
* total audio duration
* mean/median relevant metric
* variation where meaningful
* invalid/missing metric count

For WER, use an appropriate corpus-level formulation if that is the established project metric rather than blindly averaging per-example percentages.

Document the aggregation method.

---

# 30. Important Evaluation Limitation

The output of a single-stream WhisperRT recognizer is not necessarily equivalent to a two-stream multi-talker ASR system.

Therefore:

> Phase 4 is measuring the behavior of a single-stream streaming ASR model under overlapped acoustic input, not claiming that it performs explicit multi-talker recognition.

This distinction must appear in the documentation and report.

---

# 31. No Optimization

Do not:

* tune WhisperRT weights
* fine-tune the model
* change decoding specifically for overlap
* add separation
* add OSD
* add prompt engineering specifically for overlap
* add speaker-aware processing

If you discover a promising improvement, document it as a **future experiment** rather than implementing it.

The baseline must remain interpretable.

---

# 32. Research Questions

Phase 4 should provide evidence for questions such as:

### RQ1

How does streaming WhisperRT behave when two English speakers overlap?

### RQ2

How does performance vary as overlap increases?

### RQ3

How does relative speaker level affect recognition?

### RQ4

Does overlap affect streaming latency or real-time behavior?

### RQ5

What observable failure modes justify introducing an explicit overlap-detection/adaptive mechanism?

Do not answer these questions before running the experiments.

---

# 33. Documentation

Update:

```text
README.md
docs/datasets.md
docs/experiments.md
docs/architecture.md
```

Document:

* Phase 4 objective
* experiment design
* baseline definition
* reference construction
* metrics
* limitations
* commands
* configuration
* outputs
* how to reproduce the experiment

Do not insert fabricated result numbers into the documentation.

Use:

```text
NOT MEASURED
```

where necessary.

---

# 34. Phase Report

Create:

```text
docs/phase_reports/phase_4_report.md
```

Include:

## Objective

What Phase 4 tested.

## Experimental Setup

* model
* dataset
* conditions
* chunk size
* VAD state
* hardware
* configuration

## Results

Actual measured results.

## Baseline Comparison

Clean vs control vs overlap.

## Failure Analysis

Actual observed failure examples.

## Streaming Analysis

Latency/RTF/incremental-output observations supported by measurements.

## Limitations

Especially limitations of evaluating a single-stream recognizer on two-speaker mixtures.

## Research Implications

What the measurements suggest should be investigated next.

Do not turn these implications into claims that have not been demonstrated.

## Next Phase

Identify Phase 5 as the next stage:

**Overlap Speech Detection / OSD**

---

# 35. Completion Criteria

Phase 4 is complete only when:

* [ ] Phase 3 synthetic overlap data can be consumed by the streaming pipeline.
* [ ] WhisperRT is evaluated directly on synthetic mixtures.
* [ ] Clean baseline is preserved.
* [ ] Non-overlap/control condition is evaluated where available.
* [ ] Multiple overlap conditions are evaluated where available.
* [ ] Streaming inference remains genuinely stateful/incremental.
* [ ] VAD is clearly separated from the core overlap baseline.
* [ ] Valid ASR metrics are calculated.
* [ ] Invalid/inapplicable metrics are explicitly avoided or marked `NOT MEASURED`.
* [ ] Streaming latency/RTF metrics are collected.
* [ ] Per-example predictions/results are saved.
* [ ] Aggregate results are saved.
* [ ] Failure examples can be inspected.
* [ ] Experiment configuration is reproducible.
* [ ] Kaggle workflow is documented.
* [ ] Hugging Face model loading works through the existing abstraction.
* [ ] Tests pass.
* [ ] Existing Phase 1/2 functionality remains intact.
* [ ] `scripts/smoke_test.py` remains functional.
* [ ] `notebooks/05_overlap_baseline.ipynb` is created and validated.
* [ ] README/docs are updated.
* [ ] Phase 4 report is created.
* [ ] No OSD, diarization, separation, multi-talker model, or overlap-specific optimization is implemented.

---

# 36. Final Execution Rule

Do not implement Phase 5 during this task.

At the end of Phase 4, stop and report:

1. files created/modified
2. baseline architecture used
3. model actually used
4. dataset actually used
5. experiment conditions
6. exact commands executed
7. measured results
8. metric definitions
9. test results
10. smoke-test result
11. failure examples
12. limitations
13. unresolved research questions
14. exact recommended next step

The next phase will introduce the **Overlap Speech Detection (OSD) abstraction and model-selection/implementation work**.

Do not implement OSD during Phase 4.
