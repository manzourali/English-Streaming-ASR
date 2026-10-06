# PHASE 9 — Comprehensive Evaluation, Ablation, and Robustness Study

## Objective

Perform the main scientific evaluation of the English Streaming ASR project.

Phases 1–8 progressively established:

```text
WhisperRT Streaming ASR
        ↓
Streaming VAD
        ↓
Synthetic Overlap Dataset
        ↓
WhisperRT Under Overlap
        ↓
Streaming OSD
        ↓
Adaptive Routing
        ↓
Multi-Talker / Overlap-Aware ASR
        ↓
Training / Fine-Tuning
```

Phase 9 should now stop adding major architectural components and instead answer the project's central research questions through controlled experiments.

The primary objective is:

> Determine whether the proposed streaming architecture improves recognition of overlapping English speech, under what conditions it works, what component contributes to the improvement, what computational cost it introduces, and where the system fails.

This phase should produce the primary experimental evidence that can later be used in the thesis.

---

# 1. Core Research Questions

The evaluation must explicitly address these questions.

### RQ1 — Streaming ASR

How does the selected streaming ASR system perform on ordinary English speech in terms of:

* WER
* RTF
* latency
* output stability
* computational cost?

---

### RQ2 — Effect of Overlap

How much does overlapping speech degrade ordinary WhisperRT recognition?

Compare:

```text
Clean Speech
vs
Low Overlap
vs
Medium Overlap
vs
High Overlap
```

---

### RQ3 — OSD

How accurately does the streaming overlap speech detector identify overlapping speech?

Measure:

* precision
* recall
* F1
* false positives
* false negatives
* event detection delay

---

### RQ4 — Adaptive Routing

Does adaptive routing provide a useful trade-off between recognition quality and computational cost?

Compare:

```text
Always WhisperRT
Always Multi-Talker
Adaptive Routing
Oracle Adaptive Routing
```

---

### RQ5 — Multi-Talker ASR

Does the selected overlap-aware / multi-talker model outperform ordinary WhisperRT on overlapping speech?

Use the correct multi-talker metric.

---

### RQ6 — Fine-Tuning

Does Phase 8 training improve the overlap-aware model?

Compare:

```text
Base Multi-Talker
vs
Fine-Tuned Multi-Talker
```

while also measuring clean-speech degradation or improvement.

---

### RQ7 — Component Contribution

Which components actually matter?

Perform ablations for:

```text
VAD
OSD
Adaptive Routing
Multi-Talker ASR
Fine-Tuning
```

---

### RQ8 — Streaming Trade-Off

What is the relationship between:

```text
chunk size
context
lookahead
latency
RTF
WER
```

?

---

### RQ9 — Robustness

Does the system remain effective when overlap characteristics change?

Investigate:

* overlap ratio
* relative speaker energy
* temporal offset
* speaker combinations
* speech duration
* acoustic conditions if available

---

### RQ10 — Generalization

Does a model trained on synthetic overlap generalize beyond the exact synthetic conditions used for training?

If an appropriate external dataset is available, evaluate it separately.

---

# 2. Important Phase Boundary

Phase 9 is primarily an **evaluation phase**.

Do NOT introduce major new model architectures.

Do NOT:

* replace the selected ASR model
* replace the selected OSD model
* redesign the routing architecture
* start a new training strategy
* perform RL
* introduce diarization
* introduce speaker enrollment
* start a completely new dataset-generation methodology

If a serious implementation bug is discovered, fix it.

If a scientific weakness is discovered, document it.

Do not silently redesign the system to improve the results.

---

# 3. Freeze the Experimental System

Before running the final experiments, establish a frozen experimental configuration.

Record:

```text
Git commit
Model checkpoints
OSD checkpoint/configuration
VAD configuration
Routing configuration
Training configuration
Dataset manifests
Random seeds
Software environment
Hardware
```

Create:

```text
configs/evaluation.yaml
```

This configuration should reference the exact versions/checkpoints used for the final experiments.

The final evaluation should be reproducible from this configuration.

---

# 4. Experimental Dataset Policy

Use clearly separated:

```text
Training
Validation
Test
```

sets.

The test set must not be used for:

* hyperparameter tuning
* threshold selection
* checkpoint selection
* model selection
* routing-policy tuning

If thresholds or hyperparameters must be changed after seeing test results, rerun the evaluation using a new untouched test condition or clearly label the result as exploratory.

---

# 5. Primary Dataset

The main controlled evaluation should use the Phase 3 synthetic overlap dataset.

Maintain the established conditions:

```text
No Overlap
Low Overlap
Medium Overlap
High Overlap
```

Use exactly the overlap definitions established in the project configuration.

Do not silently change overlap thresholds for the final evaluation.

---

# 6. Synthetic Dataset Stratification

Where metadata allows, stratify the test set by:

### Overlap Ratio

```text
low
medium
high
```

### Relative Speaker Energy

For example:

```text
balanced
moderately dominant
strongly dominant
```

using the project's actual mixing configuration.

### Temporal Relationship

Investigate:

```text
partial overlap
long overlap
short overlap
```

if such categories are already represented.

Do not create unsupported categories.

---

# 7. External Evaluation

If technically feasible, evaluate on at least one real-world or independently generated dataset.

Candidates may include:

* LibriCSS
* AMI
* CHiME-6
* another publicly available English overlapping-speech corpus

Select based on:

* license
* availability
* transcription format
* overlap annotation
* compatibility with the model
* compute requirements

Do not force external evaluation if preprocessing is unreasonable.

If used, keep it completely separate from synthetic results.

Example:

```text
Synthetic Results
-----------------
LibriSpeech-based overlap

External Results
----------------
LibriCSS
```

Never combine the datasets into one score without a defensible methodology.

---

# 8. Primary Baselines

The final evaluation must include these baselines.

## Baseline 1 — Clean WhisperRT

```text
Audio
 ↓
WhisperRT
```

Purpose:

Establish ordinary streaming ASR performance.

---

## Baseline 2 — WhisperRT on Overlap

```text
Overlapping Audio
 ↓
WhisperRT
```

Purpose:

Measure degradation caused by overlap.

---

## Baseline 3 — Multi-Talker Model

```text
Overlapping Audio
 ↓
Multi-Talker ASR
```

Purpose:

Measure the capability of the overlap-aware model independently of routing.

---

## Baseline 4 — Fine-Tuned Multi-Talker

```text
Overlapping Audio
 ↓
Fine-Tuned Multi-Talker
```

Purpose:

Measure the effect of Phase 8 training.

---

## Baseline 5 — Adaptive Pipeline

```text
Audio
 ↓
VAD
 ↓
OSD
 ↓
Router
 ├── WhisperRT
 └── Multi-Talker
```

Purpose:

Measure the complete proposed architecture.

---

# 9. Oracle Conditions

Maintain oracle experiments where useful.

At minimum:

```text
Oracle OSD
```

and, if supported:

```text
Oracle Routing
```

The purpose is to determine the upper bound of the routing architecture.

Clearly label:

```text
ORACLE — NOT DEPLOYABLE
```

Do not include oracle results in the primary deployable-system comparison.

---

# 10. Main Ablation Study

Perform controlled component ablations.

At minimum evaluate:

| Configuration | VAD | OSD | Router | Multi-Talker | Fine-Tuned |
| ------------- | --: | --: | -----: | -----------: | ---------: |
| A             |  No |  No |     No |           No |         No |
| B             | Yes |  No |     No |           No |         No |
| C             | Yes | Yes |     No |           No |         No |
| D             | Yes | Yes |    Yes |           No |         No |
| E             | Yes | Yes |    Yes |          Yes |         No |
| F             | Yes | Yes |    Yes |          Yes |        Yes |

The exact matrix may be adjusted if some combinations are technically meaningless.

The important principle is:

> Change one major component at a time whenever possible.

---

# 11. VAD Ablation

Compare:

```text
Without VAD
```

against:

```text
With VAD
```

Measure:

* WER
* latency
* RTF
* compute overhead
* false suppression of speech
* impact on OSD

Do not claim that VAD improves ASR simply because it reduces computation.

Separate:

```text
Recognition Quality
```

from:

```text
Computational Efficiency
```

---

# 12. OSD Ablation

Compare:

```text
No OSD
```

against:

```text
Predicted OSD
```

against:

```text
Oracle OSD
```

Measure:

* routing accuracy
* overlap precision
* overlap recall
* F1
* detection delay
* recognition performance
* computational overhead

This should reveal how much OSD quality limits the complete system.

---

# 13. Routing Ablation

Compare:

```text
Always WhisperRT
```

```text
Always Multi-Talker
```

```text
Adaptive Routing
```

```text
Oracle Adaptive Routing
```

Measure:

* WER/cpWER
* RTF
* GPU memory
* latency
* branch utilization
* number of transitions

The main question is:

> Can adaptive routing achieve a useful quality/efficiency trade-off compared with always running the more expensive overlap-aware model?

---

# 14. Fine-Tuning Ablation

Compare:

```text
Base Multi-Talker
```

against:

```text
Fine-Tuned Multi-Talker
```

and, where available:

```text
Clean Fine-Tuned
Overlap Fine-Tuned
Mixed Fine-Tuned
```

Measure separately:

```text
Clean Speech
Overlap Speech
```

The evaluation must reveal whether fine-tuning causes catastrophic forgetting.

---

# 15. Streaming Parameter Study

Investigate the effect of streaming parameters.

At minimum vary one parameter at a time:

```text
chunk duration
```

and, if supported:

```text
context duration
lookahead
```

Example conceptual matrix:

```text
small chunk
medium chunk
large chunk
```

Do not blindly use specific values; use the ranges supported by the actual implementation.

For each setting measure:

* WER
* overlap metric
* first-output latency
* average latency
* end-of-utterance latency
* RTF
* memory

---

# 16. Latency–Accuracy Trade-Off

Create a trade-off analysis.

For each streaming configuration calculate:

```text
Latency
vs
WER
```

and, where meaningful:

```text
Latency
vs
cpWER
```

The goal is to identify whether the system has a useful operating region.

Do not select the "best" configuration using test data.

Use validation data for configuration selection.

The final test results should be reported for the selected configuration.

---

# 17. OSD Threshold Study

If the OSD model exposes a threshold:

Perform threshold analysis on the validation set.

Measure:

```text
threshold
precision
recall
F1
detection delay
routing behavior
```

Select the operating point using validation data.

Then freeze it before test evaluation.

Do not tune the threshold on the test set.

---

# 18. Hysteresis Study

If the adaptive router uses hysteresis/stabilization:

Measure the effect of:

```text
no hysteresis
small hysteresis
larger hysteresis
```

where supported.

Investigate:

* false transitions
* missed transitions
* delayed transitions
* branch switching frequency
* duplicate output
* latency

The goal is to determine whether stabilization improves system behavior.

---

# 19. Overlap Ratio Analysis

Plot recognition performance as a function of overlap ratio.

For example:

```text
Overlap Ratio →
WER / cpWER →
```

Compare:

```text
WhisperRT
Multi-Talker
Fine-Tuned Multi-Talker
Adaptive
```

This is one of the most important figures in the project.

Do not interpolate values that were not measured.

---

# 20. Speaker Dominance Analysis

Where relative gain metadata exists, investigate:

```text
balanced speakers
```

versus:

```text
dominant speaker
```

Measure whether the system disproportionately recognizes the louder speaker.

Analyze:

* speaker-specific recognition
* missed speaker rate
* token imbalance
* cpWER
* stream assignment

Do not claim speaker-level fairness unless the evaluation actually supports that claim.

---

# 21. Error Taxonomy

Create a structured error taxonomy.

At minimum:

```text
1. Missed speaker
2. Missing words
3. Speaker dominance
4. Hallucination
5. Repetition
6. Cross-speaker mixing
7. Incorrect stream assignment
8. Incorrect temporal order
9. Delayed output
10. Duplicate output
11. OSD false positive
12. OSD false negative
13. Incorrect route transition
14. Branch initialization failure
```

Annotate a representative subset of failures.

Do not manually inspect every test sample unless practical.

---

# 22. Routing Error vs ASR Error

For each failed adaptive example determine, where possible:

```text
Was the routing decision correct?
```

If no:

```text
Routing Failure
```

If yes:

```text
ASR Failure
```

This distinction should appear in the final analysis.

The purpose is to determine whether future improvements should target:

```text
OSD
```

or:

```text
Multi-Talker ASR
```

---

# 23. Speaker-Miss Analysis

For overlap examples with two speakers, measure whether:

```text
both speakers recognized
```

or:

```text
only one speaker substantially recognized
```

If the evaluation format supports it, define an objective criterion.

Do not manually classify results using an undocumented subjective rule.

---

# 24. Multi-Talker Evaluation

Use the metric appropriate to the selected model.

Potential metrics:

* cpWER
* permutation-invariant WER
* stream-wise WER
* speaker-attributed WER
* token-level accuracy

Document exactly:

```text
metric definition
reference format
hypothesis format
permutation handling
speaker assignment method
```

If cpWER is used, explain how permutations are handled.

Do not compare incompatible metrics directly.

For example:

```text
ordinary WER
```

and:

```text
cpWER
```

must not be presented as though they were identical quantities.

---

# 25. Statistical Analysis

Where the test set is sufficiently large, report uncertainty.

Possible methods:

* bootstrap confidence intervals
* paired bootstrap
* paired sample analysis

For WER-like metrics, use a defensible paired evaluation methodology.

Report:

```text
metric
point estimate
confidence interval
```

where practical.

Do not rely only on a single aggregate number.

---

# 26. Significance of Improvements

If claiming that one system is better than another, investigate whether the observed difference is meaningful.

Avoid statements such as:

> Model A is better than Model B.

Prefer:

> Model A achieved X compared with Y for Model B under the same evaluation condition.

If statistical testing is performed, document the test and assumptions.

Do not perform statistical significance testing merely for appearance.

---

# 27. Per-Example Evaluation Records

Every final evaluation example should have a machine-readable result.

Include, where available:

```text
sample_id
condition
overlap_ratio
speaker_count
reference
hypothesis
metric
route
ground_truth_overlap
predicted_overlap
routing_correct
latency
rtf
model
checkpoint
```

Store results under:

```text
outputs/metrics/
outputs/predictions/
```

Do not rely exclusively on aggregate tables.

---

# 28. Experiment Reproducibility

Every final experiment must record:

```text
experiment ID
timestamp
git commit
config
model checkpoint
dataset manifest
seed
hardware
software versions
metrics
runtime
```

The evaluation script must be able to regenerate the final tables from raw result files.

Do not manually type the final numbers into the report.

---

# 29. Evaluation Script

Create or substantially extend:

```text
scripts/evaluate.py
```

It should support:

```text
dataset selection
model/checkpoint selection
evaluation condition
metric selection
output directory
config loading
```

Example:

```bash
python scripts/evaluate.py \
    --config configs/evaluation.yaml
```

---

# 30. Benchmark Script

Update:

```text
scripts/benchmark.py
```

to compare:

* latency
* RTF
* memory
* throughput
* model loading time
* branch activation cost

where meaningful.

Do not mix benchmark and ASR-quality evaluation logic unnecessarily.

---

# 31. Evaluation Module

Organize evaluation logic under:

```text
src/streaming_asr/evaluation/
```

Suggested responsibilities:

```text
evaluator.py
benchmark.py
reports.py
```

The evaluation code should:

* load predictions
* align references
* calculate metrics
* aggregate results
* stratify results
* generate machine-readable outputs
* generate report-ready tables

---

# 32. Visualization

Create thesis-quality visualizations.

At minimum:

### Figure 1 — Architecture

Final adaptive architecture.

---

### Figure 2 — WER vs Overlap Ratio

Compare major systems.

---

### Figure 3 — Multi-Talker Metric vs Overlap Ratio

Use cpWER or the appropriate metric.

---

### Figure 4 — Latency vs WER

Show the streaming trade-off.

---

### Figure 5 — OSD Precision/Recall/F1

Show detector performance.

---

### Figure 6 — Routing Timeline

Show a representative example:

```text
Ground Truth
OSD
Router
```

---

### Figure 7 — Computational Cost

Compare:

```text
Always WhisperRT
Always Multi-Talker
Adaptive
```

---

### Figure 8 — Ablation Results

Show contribution of:

```text
VAD
OSD
Routing
Multi-Talker
Fine-Tuning
```

Only generate figures supported by valid measurements.

---

# 33. Tables

Generate final machine-readable and human-readable tables.

At minimum:

### Table 1 — Main ASR Results

| System                  | Clean | Low Overlap | Medium Overlap | High Overlap |
| ----------------------- | ----: | ----------: | -------------: | -----------: |
| WhisperRT               |       |             |                |              |
| Multi-Talker            |       |             |                |              |
| Fine-Tuned Multi-Talker |       |             |                |              |
| Adaptive                |       |             |                |              |

Use the correct metric in each column.

---

### Table 2 — Streaming Performance

| System | RTF | First Output | Avg Latency | EOU Latency | Memory |
| ------ | --: | -----------: | ----------: | ----------: | -----: |

---

### Table 3 — OSD

| Metric          | Value |
| --------------- | ----: |
| Precision       |       |
| Recall          |       |
| F1              |       |
| Detection Delay |       |

---

### Table 4 — Ablation

| Configuration | ASR | OSD | Routing | RTF |
| ------------- | --: | --: | ------: | --: |

---

### Table 5 — Fine-Tuning

| Model | Training Data | Clean Metric | Overlap Metric | RTF |
| ----- | ------------- | -----------: | -------------: | --: |

Only include values actually measured.

---

# 34. Robustness Tests

Where practical, test sensitivity to:

* different overlap ratios
* different speaker energy ratios
* different chunk sizes
* different OSD thresholds
* different context sizes
* different random seeds

Do not run every combination.

Prioritize the parameters most relevant to the research question.

---

# 35. Seed Stability

Run multiple seeds for at least the most important training experiment if computationally feasible.

Compare:

```text
mean
standard deviation
```

or an appropriate confidence interval.

If only one seed is practical, clearly state:

```text
Single-seed experiment
```

Do not imply statistical robustness from one seed.

---

# 36. Generalization to Unseen Speakers

Ensure the primary test set contains speakers not used during training.

Report this explicitly.

The experiment should answer:

> Does the overlap-aware model generalize to unseen speakers?

Do not confuse:

```text
unseen speakers
```

with:

```text
unseen acoustic conditions
```

They are different forms of generalization.

---

# 37. Generalization to Unseen Overlap Conditions

Where feasible, construct or use a test configuration with overlap characteristics that differ from training.

For example:

```text
training:
controlled overlap ratios

testing:
different ratios / temporal relationships
```

Do not modify the training set based on test results.

---

# 38. External Dataset Analysis

If LibriCSS or another real-world dataset is used, analyze:

* domain mismatch
* microphone/environment differences
* speaker behavior
* overlap characteristics
* transcription conventions
* model preprocessing compatibility

Do not simply report one external WER and call it generalization.

---

# 39. Compute Efficiency Analysis

Calculate approximate computational trade-offs.

For adaptive routing, estimate:

```text
fraction of time in SINGLE branch
fraction of time in OVERLAP branch
```

and compare:

```text
always Multi-Talker
```

against:

```text
adaptive
```

where meaningful.

The goal is to determine whether routing reduces unnecessary computation.

---

# 40. Memory Analysis

Measure peak GPU memory for:

```text
WhisperRT
Multi-Talker
Fine-Tuned Multi-Talker
Adaptive
```

where applicable.

If adaptive memory remains high because both branches remain loaded simultaneously, document that.

Do not claim memory savings merely because computation is conditionally routed.

---

# 41. Branch Loading Strategy

If the adaptive system supports:

```text
both models loaded
```

versus:

```text
lazy loading
```

do not redesign the system in this phase unless necessary.

Instead, measure/document the current behavior.

If it significantly affects the results, record it as a limitation or future optimization.

---

# 42. Real-Time Constraint

Define an explicit real-time criterion.

For example:

```text
RTF < 1
```

may indicate that processing is faster than real time.

However, do not use RTF alone.

Also consider:

* first-output latency
* sustained latency
* branch switching latency
* memory
* buffering

State exactly which criterion is being used.

---

# 43. Real-Time Failure Cases

Identify cases where:

```text
RTF > 1
```

or latency grows over time.

Investigate whether the cause is:

* model inference
* OSD
* buffering
* branch initialization
* GPU memory pressure
* Python overhead
* synchronization
* excessive context

Do not assume the cause.

Measure where possible.

---

# 44. Error Visualization

Create examples showing:

```text
Audio timeline
Ground-truth speakers
Ground-truth overlap
OSD
Router
WhisperRT output
Multi-Talker output
```

Use these to explain representative failures.

Do not use only qualitative examples to establish quantitative claims.

---

# 45. Final System Definition

At the end of Phase 9, define the exact configuration considered the:

```text
PRIMARY PROPOSED SYSTEM
```

Include:

```text
VAD model/config
OSD model/config
OSD threshold
routing policy
WhisperRT checkpoint
multi-talker checkpoint
fine-tuning strategy
chunk size
context
lookahead
dataset
```

This configuration should be frozen for Phase 10.

---

# 46. Final Comparison

The final comparison should answer:

```text
What does the project add compared with ordinary streaming WhisperRT?
```

Break the answer into:

### Recognition

Does it improve overlapping-speech recognition?

### Latency

Does it remain sufficiently real-time?

### Computation

Does adaptive routing reduce unnecessary computation?

### Robustness

Does it work across different overlap conditions?

### Generalization

Does it work beyond the synthetic training distribution?

### Complexity

What additional components are required?

Do not reduce the conclusion to a single metric.

---

# 47. Scientific Failure Is Acceptable

If the results show:

```text
Adaptive < Always Multi-Talker
```

or:

```text
Fine-Tuning < Base Model
```

or:

```text
OSD errors dominate the final system
```

report it.

These are scientifically useful results.

Do not modify the system until the desired result appears.

---

# 48. Final Experiment Freeze

Once the main experiments are complete:

Create:

```text
docs/final_experiment_configuration.md
```

Record the exact configuration used for the primary results.

This file should be sufficient for another researcher to understand:

```text
what was evaluated
how it was evaluated
which model versions were used
which dataset versions were used
which parameters were frozen
```

---

# 49. Notebook

Create or update:

```text
notebooks/10_evaluation.ipynb
```

The notebook should demonstrate:

1. loading evaluation configuration
2. loading result files
3. computing metrics
4. generating main tables
5. generating main plots
6. comparing systems
7. stratifying by overlap ratio
8. latency analysis
9. ablation analysis
10. exporting results

The notebook should preferably consume already-generated predictions rather than rerunning all expensive models.

Provide a separate optional section for reproducing a small evaluation run.

---

# 50. Tests

Add/update:

```text
tests/test_evaluation.py
tests/test_benchmark.py
tests/test_metrics.py
```

Test:

* WER calculation
* multi-talker metric calculation
* permutation handling
* latency calculation
* RTF calculation
* aggregation
* stratification
* invalid inputs
* missing predictions
* result serialization

Do not require large model downloads for metric tests.

---

# 51. Regression Tests

Run all previous tests:

```bash
pytest tests/
```

Verify:

* Phase 1 streaming ASR
* Phase 2 VAD
* Phase 3 dataset
* Phase 4 overlap baseline
* Phase 5 OSD
* Phase 6 adaptive routing
* Phase 7 multi-talker branch
* Phase 8 training/checkpoint loading

still work.

A regression that changes previous results must be documented.

---

# 52. Smoke Test

Run:

```bash
python scripts/smoke_test.py
```

The smoke test must remain successful.

If the smoke test becomes too large or slow, simplify it without removing coverage of the major interfaces.

---

# 53. Final Results Storage

Store final results under:

```text
outputs/
├── metrics/
├── predictions/
├── figures/
├── reports/
└── logs/
```

Do not commit large generated outputs.

Small summary files may be committed if appropriate.

---

# 54. Report Generation

Where practical, generate summary tables programmatically from the raw evaluation results.

Do not manually copy metrics into multiple files.

The same source result should generate:

```text
CSV/JSON
tables
plots
report summaries
```

This reduces transcription errors.

---

# 55. Documentation

Update:

```text
README.md
docs/architecture.md
docs/experiments.md
docs/datasets.md
docs/thesis_mapping.md
```

Create:

```text
docs/final_experiment_configuration.md
docs/phase_reports/phase_9_report.md
```

The documentation should now describe the complete evaluated system rather than merely the implementation.

---

# 56. Phase 9 Report

Create:

```text
docs/phase_reports/phase_9_report.md
```

Include:

## 1. Research Questions

List all evaluated RQs.

## 2. Frozen System

Exact models and configurations.

## 3. Dataset

Training/validation/test separation.

## 4. Evaluation Protocol

Exact methodology.

## 5. Baselines

All baseline conditions.

## 6. Main Results

Primary quantitative results.

## 7. OSD Results

Precision/recall/F1/delay.

## 8. ASR Results

WER/cpWER/etc.

## 9. Streaming Results

RTF/latency/memory.

## 10. Ablation Results

Component contribution.

## 11. Fine-Tuning Results

Base vs adapted.

## 12. Robustness

Overlap ratio, speaker dominance, chunk size, etc.

## 13. Generalization

External dataset if available.

## 14. Error Analysis

Structured failure categories.

## 15. Computational Analysis

Cost of adaptive routing.

## 16. Statistical Analysis

Confidence intervals / seed stability where available.

## 17. Limitations

Explicit limitations.

## 18. Primary System

Exact final configuration.

## 19. Answers to Research Questions

Answer each RQ using measured evidence.

## 20. Phase 10 Requirements

List exactly what remains for reproducibility and thesis finalization.

---

# 57. Required Final Result Table

The report should contain a primary table similar to:

| System                  | Clean Metric | Low Overlap | Medium Overlap | High Overlap | RTF | Latency |
| ----------------------- | -----------: | ----------: | -------------: | -----------: | --: | ------: |
| WhisperRT               |              |             |                |              |     |         |
| Multi-Talker            |              |             |                |              |     |         |
| Fine-Tuned Multi-Talker |              |             |                |              |     |         |
| Adaptive                |              |             |                |              |     |         |
| Oracle Adaptive         |              |             |                |              |     |         |

Use the correct ASR metric for each condition.

Do not force the same metric into every column if the output representations differ.

---

# 58. Required Ablation Table

Include:

| System Configuration | VAD | OSD | Routing | Multi-Talker | Fine-Tuning | ASR Metric | RTF |
| -------------------- | --: | --: | ------: | -----------: | ----------: | ---------: | --: |
| Baseline             |     |     |         |              |             |            |     |
| + VAD                |     |     |         |              |             |            |     |
| + OSD                |     |     |         |              |             |            |     |
| + Routing            |     |     |         |              |             |            |     |
| + Multi-Talker       |     |     |         |              |             |            |     |
| + Fine-Tuning        |     |     |         |              |             |            |     |

Only use configurations that were actually evaluated.

---

# 59. Required OSD Table

Include:

| Condition | Precision | Recall |  F1 | Detection Delay |
| --------- | --------: | -----: | --: | --------------: |
| OSD       |           |        |     |                 |
| Oracle    |       1.0 |    1.0 | 1.0 |               0 |

The oracle row should only be included if the metric definition actually makes this interpretation valid.

Clearly mark it as an oracle reference, not model performance.

---

# 60. Required Computational Table

Include:

| System       | Model Params | Trainable Params | GPU Memory | RTF | Avg Latency |
| ------------ | -----------: | ---------------: | ---------: | --: | ----------: |
| WhisperRT    |              |                  |            |     |             |
| Multi-Talker |              |                  |            |     |             |
| Fine-Tuned   |              |                  |            |     |             |
| Adaptive     |              |                  |            |     |             |

Only report values actually measured.

---

# 61. Research Integrity

This phase is the most important phase for avoiding misleading conclusions.

Never fabricate:

* WER
* cpWER
* OSD F1
* latency
* RTF
* memory
* statistical significance
* confidence intervals
* improvement percentages
* external-dataset results

Never infer a quantitative improvement from qualitative examples.

Never claim:

```text
real-time
```

without measuring real-time performance.

Never claim:

```text
robust
```

without testing relevant perturbations.

Never claim:

```text
generalizes
```

without an appropriate independent evaluation.

Use:

```text
NOT MEASURED
```

where necessary.

---

# 62. Improvement Reporting

When an improvement is observed, calculate it programmatically.

For example:

```text
absolute improvement
relative improvement
```

depending on the metric.

For WER:

```text
relative WER reduction =
(base_WER - new_WER) / base_WER
```

Do not report an improvement if the baseline is invalid or incomparable.

For metrics where lower is better, make the direction explicit.

For metrics where higher is better, do the same.

---

# 63. Error Bars and Confidence

Where practical, include uncertainty.

Do not display false precision.

If the experiment supports only:

```text
WER = 23.4
```

do not report:

```text
WER = 23.417293%
```

unless the measurement methodology justifies that precision.

Use sensible significant figures.

---

# 64. Phase 9 Completion Criteria

Phase 9 is complete only when:

* final experimental configuration is frozen
* primary test set is untouched during final tuning
* all major baselines are evaluated
* adaptive system is evaluated
* multi-talker model is evaluated
* fine-tuned model is evaluated
* OSD is evaluated
* ablations are completed
* streaming metrics are measured
* computational cost is measured
* overlap-ratio analysis is completed
* error analysis is completed
* external evaluation is performed if feasible
* statistical analysis is performed where justified
* all final results are reproducible
* evaluation scripts work
* final tables are generated programmatically
* final figures are generated
* all tests pass
* smoke test passes
* notebook works
* README is updated
* final experiment configuration is documented
* Phase 9 report is complete

---

# 65. Final Agent Response

After completing Phase 9, stop.

Do not automatically start Phase 10.

Report:

1. frozen final configuration
2. datasets and splits
3. baseline systems
4. ablation matrix
5. primary results
6. overlap-ratio results
7. OSD results
8. routing results
9. multi-talker results
10. fine-tuning results
11. streaming latency
12. RTF
13. memory/compute
14. robustness results
15. external evaluation results if performed
16. statistical analysis
17. error categories
18. major failure cases
19. primary proposed system
20. strongest measured improvement
21. strongest measured limitation
22. whether adaptive routing actually provides a useful trade-off
23. whether fine-tuning actually helps
24. whether the system satisfies the real-time target
25. tests
26. smoke-test status
27. notebook status
28. documentation status
29. exact limitations
30. exact requirements for Phase 10

Clearly distinguish:

```text
IMPLEMENTED
```

from:

```text
EVALUATED
```

from:

```text
MEASURED
```

from:

```text
STATISTICALLY SUPPORTED
```

from:

```text
NOT MEASURED
```

from:

```text
LIMITATION
```

from:

```text
FUTURE WORK
```

The final scientific conclusion must be based only on measured evidence.

The central output of Phase 9 is not another model.

It is a **controlled, reproducible body of evidence** showing:

```text
What works
Why it works
When it works
How much it helps
What it costs
Where it fails
```

The resulting configuration and results should be sufficiently stable that Phase 10 can focus on **reproducibility, final experiment packaging, thesis mapping, and presentation of the research**, rather than introducing new technical functionality.
