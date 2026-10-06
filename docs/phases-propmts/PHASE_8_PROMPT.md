# PHASE 8 — Training / Fine-Tuning the Overlap-Aware ASR System

## Objective

Implement the training and fine-tuning stage of the English Streaming ASR project.

Phase 7 introduced the first actual overlap-aware / multi-talker ASR branch and integrated it into the adaptive streaming architecture.

Phase 8 should investigate whether the selected overlap-aware ASR model can be improved through **task-specific training or parameter-efficient fine-tuning (PEFT)** using the project's controlled synthetic overlap data.

The main research question is:

> Can lightweight task-specific adaptation improve overlap-aware English ASR performance while preserving the model's streaming or low-latency behavior and keeping computational requirements practical?

The training pipeline must be scientifically controlled.

Do not assume that fine-tuning will improve performance.

The result may be:

```text
Fine-tuning improves ASR
```

or:

```text
Fine-tuning does not improve ASR
```

or:

```text
Fine-tuning improves overlap recognition but degrades streaming behavior
```

or:

```text
Fine-tuning is technically incompatible with the selected model
```

All of these are valid research outcomes.

---

# 1. Scope

Implement:

1. Training/fine-tuning investigation.
2. Dataset preparation for training.
3. Training data validation.
4. Appropriate preprocessing.
5. Model-specific training interface.
6. Full fine-tuning support if practical.
7. PEFT support if technically compatible.
8. LoRA/QLoRA investigation where appropriate.
9. Validation pipeline.
10. Checkpoint management.
11. Experiment configuration.
12. Evaluation before and after adaptation.
13. Streaming regression testing.
14. Adaptive pipeline regression testing.
15. Ablation experiments.
16. Reproducible training scripts.
17. Kaggle-compatible training demonstration where feasible.
18. Documentation.
19. Phase report.

Preserve all previous phases.

---

# 2. Explicitly Out of Scope

Do NOT turn Phase 8 into:

* large-scale pretraining
* training a new ASR architecture from scratch
* reinforcement learning
* RLHF
* preference optimization
* speaker identification
* speaker enrollment
* full diarization
* source separation as the primary objective
* multilingual training
* massive dataset preprocessing
* distributed multi-node training
* thesis-scale exhaustive hyperparameter search

The goal is **targeted adaptation**, not creation of a new foundation model.

---

# 3. Start With Model Compatibility Analysis

Before writing the training code, inspect the exact model selected in Phase 7.

Determine:

* architecture
* trainable parameters
* encoder/decoder structure
* tokenizer
* loss function
* training objective
* input format
* target format
* attention mechanism
* streaming state
* cache/state mechanism
* pretrained checkpoint
* official training implementation
* official fine-tuning implementation
* Hugging Face Trainer compatibility
* Transformers compatibility
* PEFT compatibility
* LoRA compatibility
* QLoRA compatibility
* gradient checkpointing support
* mixed precision support
* quantization constraints
* checkpoint saving/loading
* whether streaming inference behavior changes after training

Do not assume that a model that supports inference through Hugging Face also supports training through the same interface.

Document the findings in:

```text
docs/training_model_compatibility.md
```

---

# 4. Training Strategy Decision

Based on the compatibility investigation, select the most defensible training strategy.

Possible strategies:

### Option A — PEFT

```text
Base Model
     ↓
LoRA / QLoRA
     ↓
Overlap-Aware Fine-Tuning
```

Prefer this when technically supported and sufficient for the research objective.

### Option B — Full Fine-Tuning

Use only if:

* model size is manageable,
* training implementation is reliable,
* GPU requirements are practical,
* and full fine-tuning is scientifically justified.

### Option C — Partial Fine-Tuning

For example:

* decoder only
* selected attention layers
* projection layers
* task-specific heads

Use only if appropriate for the selected architecture.

### Option D — No Fine-Tuning

If the selected model cannot be reliably fine-tuned with available infrastructure, document:

```text
TRAINING NOT TECHNICALLY FEASIBLE
```

Do not invent a training procedure.

---

# 5. Training Decision Criteria

The selected strategy should consider:

* available Kaggle GPU
* GPU memory
* model size
* training stability
* checkpoint size
* reproducibility
* implementation complexity
* compatibility with streaming inference
* expected scientific value
* training time

The goal is not necessarily the highest possible model quality.

The goal is a **reproducible and defensible experiment**.

---

# 6. Training Data

Use the project's existing data pipeline.

Primary training source:

```text
LibriSpeech
        +
Synthetic Overlap Dataset
```

The main training objective should focus on overlap-aware recognition.

Use the Phase 3 synthetic overlap generator and manifests rather than creating an unrelated dataset pipeline.

Where appropriate, include:

```text
clean single-speaker examples
+
overlap examples
```

This should allow investigation of whether adaptation causes catastrophic degradation on ordinary speech.

---

# 7. Training Dataset Structure

Create a model-specific training representation from the existing manifests.

Do not duplicate the audio dataset unnecessarily.

The training manifest should contain enough information to reproduce each example.

At minimum:

```text
sample_id
audio_path
sample_rate
duration
split
num_speakers
speaker_ids
source_ids
source_transcripts
overlap_start
overlap_end
overlap_ratio
mixing_parameters
target_format
```

The exact target representation must match the selected multi-talker model.

---

# 8. Target Representation

This is one of the most important parts of Phase 8.

Determine how the selected model represents multiple speakers.

Possible representations include:

```text
separate transcription streams
```

or:

```text
serialized output
```

or:

```text
speaker-attributed token sequence
```

or:

```text
multi-output decoder targets
```

Do not invent a target representation.

Use the representation required by the model's training objective.

Document:

```text
Input
Target
Loss
Output
Evaluation mapping
```

in:

```text
docs/training_objective.md
```

---

# 9. Synthetic Overlap Data

Use controlled overlap generation from Phase 3.

The training pipeline must preserve the ability to configure:

```text
overlap ratio
relative timing
speaker count
relative gain/SNR
duration
```

At minimum support:

```text
2 speakers
```

Initially.

Do not expand to three or more speakers unless the selected model and computational budget justify it.

---

# 10. Training Conditions

Create at least the following training conditions where feasible.

### Condition A — Base Model

No fine-tuning.

```text
Pretrained Model
```

This is the baseline.

---

### Condition B — Clean Fine-Tuning

Fine-tune using clean single-speaker English data.

Purpose:

Determine whether the training pipeline itself can adapt the model without specifically teaching overlap handling.

---

### Condition C — Overlap Fine-Tuning

Fine-tune using synthetic overlap data.

Purpose:

Measure direct improvement on overlapping speech.

---

### Condition D — Mixed Fine-Tuning

Use:

```text
clean speech
+
overlap speech
```

with a configurable sampling ratio.

Purpose:

Investigate whether mixed training provides a better tradeoff between ordinary ASR and overlap ASR.

---

# 11. Keep the Experiment Matrix Small

Do not immediately run dozens of configurations.

Start with:

```text
Base
Clean FT
Overlap FT
Mixed FT
```

If PEFT is used, begin with one defensible LoRA configuration.

Only expand the search if the first experiments demonstrate that training is stable and meaningful.

---

# 12. Configuration

Create:

```text
configs/training.yaml
```

Configuration should include:

```yaml
model:
  name:
  checkpoint:
  strategy:
  trainable_parameters:

dataset:
  train_manifest:
  validation_manifest:
  clean_ratio:
  overlap_ratio:
  max_duration:

training:
  seed:
  epochs:
  max_steps:
  batch_size:
  gradient_accumulation:
  learning_rate:
  warmup_ratio:
  weight_decay:
  max_grad_norm:

peft:
  enabled:
  method:
  rank:
  alpha:
  dropout:
  target_modules:

precision:
  dtype:
  gradient_checkpointing:

evaluation:
  eval_steps:
  save_steps:
  metric:
```

Do not hard-code these parameters in the training implementation.

---

# 13. Training Script

Create or update:

```text
scripts/train_asr.py
```

The script should support:

```text
configuration loading
dataset loading
model loading
training
validation
checkpoint saving
metric logging
experiment logging
```

Example:

```bash
python scripts/train_asr.py \
    --config configs/training.yaml
```

The exact CLI can differ, but it must be reproducible.

---

# 14. Training Module

Implement or extend:

```text
src/streaming_asr/training/
```

Potential components:

```text
trainer.py
losses.py
peft.py
data.py
collator.py
checkpoint.py
```

Reuse existing utilities instead of duplicating:

* configuration
* logging
* paths
* dataset backends
* experiment tracking
* reproducibility

---

# 15. Data Collator

Implement a model-compatible data collator.

It must handle:

* variable-length audio
* padding
* attention masks
* target sequences
* multiple output streams if required
* serialized targets if required
* ignored/padded target positions
* sample-rate normalization

Validate that the collator produces exactly the tensors expected by the selected model.

Add unit tests for it.

---

# 16. Dataset Leakage Prevention

Maintain strict separation between:

```text
training speakers
validation speakers
test speakers
```

Do not allow the same speaker to appear across splits if the experimental design claims speaker-disjoint evaluation.

Verify this programmatically.

Create a validation utility that checks:

```text
speaker leakage
source utterance leakage
duplicate samples
manifest inconsistencies
```

A training run must fail early if leakage is detected.

---

# 17. Training Reproducibility

Every training run must record:

```text
experiment ID
timestamp
seed
model
checkpoint
training strategy
dataset
dataset version/manifest
training configuration
software environment
hardware
GPU
batch size
gradient accumulation
learning rate
epochs
steps
trainable parameter count
total parameter count
training duration
validation metrics
checkpoint path
```

The experiment must be reproducible from its configuration.

---

# 18. Parameter Count

Always report:

```text
Total parameters
Trainable parameters
Frozen parameters
Trainable percentage
```

For example:

```text
Total: ...
Trainable: ...
Frozen: ...
Trainable %: ...
```

Do not fabricate values.

Calculate them from the actual model.

This is especially important for PEFT experiments.

---

# 19. PEFT Implementation

If PEFT is compatible with the selected model, implement it cleanly.

Potential methods:

```text
LoRA
QLoRA
```

Do not implement every PEFT method.

Start with one method that is well supported by the selected architecture.

Document:

```text
rank
alpha
dropout
target modules
quantization
trainable parameters
```

If the model does not support LoRA/QLoRA correctly, do not force it.

---

# 20. Quantization

If QLoRA is investigated:

Verify:

* supported quantization library
* supported GPU
* model architecture compatibility
* training stability
* checkpoint loading
* inference compatibility

Do not use quantization merely because it reduces memory if it creates an unreliable training pipeline.

If QLoRA is used, compare the resulting model against the appropriate non-quantized baseline where feasible.

---

# 21. Training Stability

Track:

```text
training loss
validation loss
learning rate
gradient norm if available
epoch
step
```

Detect:

* NaN loss
* exploding gradients
* divergence
* unstable validation loss
* checkpoint corruption

Stop or fail clearly when training becomes invalid.

Do not report a failed run as a successful experiment.

---

# 22. Checkpoint Management

Save:

```text
best validation checkpoint
last checkpoint
training configuration
tokenizer/configuration
adapter weights if PEFT
```

Do not commit checkpoints to Git.

Use:

```text
checkpoints/
```

and/or configured external storage.

The checkpoint must be reloadable independently from the training process.

---

# 23. Resume Training

Support resuming from a checkpoint when the selected training framework allows it.

Verify:

```text
model state
optimizer state
scheduler state
training step
random state
```

where supported.

Test resume functionality on a small run.

---

# 24. Early Stopping

If supported by the training framework, implement configurable early stopping based on validation performance.

Do not select the final model based on test performance.

The test set must remain untouched until final evaluation.

---

# 25. Validation Strategy

Validation should use a speaker-disjoint validation split.

At minimum evaluate:

```text
clean validation examples
overlap validation examples
```

Report them separately.

This is important because an overlap-trained model may improve overlap recognition while degrading clean speech.

---

# 26. Test Set Protection

Do not use:

```text
test-clean
```

or the final overlap test set for:

* hyperparameter tuning
* threshold selection
* early stopping
* checkpoint selection
* model selection

The test set is for final evaluation.

---

# 27. Phase 8 Evaluation

After training, evaluate:

### Clean ASR

```text
Base Model
vs
Fine-Tuned Model
```

### Overlap ASR

```text
Base Model
vs
Fine-Tuned Model
```

### Adaptive Pipeline

```text
Phase 7 Model
vs
Phase 8 Fine-Tuned Model
```

where applicable.

---

# 28. Required Metrics

## ASR

Use valid metrics based on the output format:

```text
WER
CER if appropriate
cpWER / permutation-invariant WER
stream-wise WER
```

Do not use an invalid single-reference metric for multi-talker output.

---

## Streaming

Measure:

```text
RTF
first-output latency
average latency
end-of-utterance latency
chunk processing time
model inference time
memory
GPU memory
```

---

## Stability

Where applicable:

```text
partial hypothesis revision rate
token stability
output delay
```

---

# 29. Regression Evaluation

Fine-tuning must not be evaluated only on the data it was designed to improve.

Always check:

```text
clean speech
low overlap
medium overlap
high overlap
```

where available.

The purpose is to identify:

```text
specialization benefit
```

versus:

```text
general ASR degradation
```

---

# 30. Adaptive Pipeline Integration

After training, integrate the fine-tuned model into:

```text
Phase 6 Adaptive Pipeline
```

The final architecture should become:

```text
Audio Stream
      ↓
Streaming VAD
      ↓
Streaming OSD
      ↓
Adaptive Router
      ↓
 ┌───────────────┴──────────────────┐
 │                                  │
Single Speaker                  Overlap
 │                                  │
WhisperRT                  Fine-Tuned Multi-Talker
 │                                  │
 └───────────────┬──────────────────┘
                 ↓
        Incremental Output
                 ↓
             Evaluation
```

The base multi-talker model must remain available.

Do not overwrite it.

---

# 31. Compare Base vs Fine-Tuned Adaptive Pipeline

At minimum compare:

```text
Phase 7 Adaptive
```

against:

```text
Phase 8 Fine-Tuned Adaptive
```

Use the same evaluation data and evaluation protocol.

The objective is to determine whether the training stage actually improves the complete system rather than only an isolated model.

---

# 32. OSD Regression

Do not retrain OSD automatically in this phase.

However, verify that changes to the ASR branch do not break:

```text
VAD
OSD
routing
branch transitions
```

Run the existing Phase 5 and Phase 6 tests.

If ASR training changes pipeline timing, measure the effect.

---

# 33. Streaming Regression

This is mandatory.

The fine-tuned model must be tested using the same streaming interface established in Phase 7.

Do not evaluate the fine-tuned model only with:

```text
full audio → model → transcript
```

The project remains streaming-first.

If training changes the model's streaming behavior, document it.

---

# 34. Offline vs Streaming Evaluation

If the selected model has both offline and streaming modes, evaluate both only if useful.

Clearly distinguish:

```text
Offline inference
```

from:

```text
Streaming inference
```

The primary result must remain the streaming/low-latency configuration.

Do not use offline performance as the main thesis result for a streaming system.

---

# 35. Training Efficiency

Measure:

```text
training time
samples/second
steps/second
GPU memory
peak GPU memory
checkpoint size
trainable parameter count
```

For PEFT:

Compare:

```text
adapter size
```

against:

```text
full model size
```

where meaningful.

---

# 36. Small Development Run

Before any real experiment:

Run a tiny training job.

For example:

```text
few samples
1–2 steps
minimal batch
```

The purpose is to verify:

* dataset
* collator
* forward pass
* backward pass
* loss
* optimizer
* checkpointing
* reload
* inference

This is not a research result.

Label it:

```text
DEVELOPMENT SMOKE TEST
```

---

# 37. Kaggle Training

Create or update:

```text
notebooks/09_training.ipynb
```

The notebook should demonstrate:

1. environment setup
2. dataset access
3. model loading
4. training configuration
5. tiny development run
6. real small training run if feasible
7. checkpoint creation
8. checkpoint reload
9. streaming inference
10. evaluation

The notebook must not assume that unlimited GPU time is available.

Use a configurable subset.

---

# 38. Hugging Face Integration

Continue supporting the existing Hugging Face model/data backend.

If trained adapters can be exported to Hugging Face-compatible format, support that.

Do not automatically upload anything.

Uploading should remain an explicit user action.

---

# 39. Local Training

The same configuration should work locally where dependencies and GPU resources are available.

Avoid Kaggle-only assumptions in the core training implementation.

---

# 40. Tests

Add or update:

```text
tests/test_training.py
tests/test_data_collator.py
tests/test_peft.py
tests/test_checkpoint.py
tests/test_metrics.py
tests/test_pipeline.py
```

Tests should cover:

### Dataset

* manifest loading
* target generation
* speaker leakage detection
* invalid sample handling

### Collator

* padding
* batch construction
* target representation
* multiple streams

### Model

* forward pass
* loss
* backward pass

### PEFT

* adapter creation
* trainable parameter count
* save/load if supported

### Checkpoint

* save
* reload
* inference after reload

### Pipeline

* fine-tuned model in overlap branch
* adaptive routing
* branch transitions

Do not require downloading the full model for ordinary unit tests.

---

# 41. Training Smoke Test

Update:

```text
scripts/smoke_test.py
```

so that it can verify the training pipeline without performing a meaningful training run.

The smoke test should verify:

```text
dataset
→ collator
→ model
→ forward
→ loss
→ backward
→ checkpoint
→ reload
→ inference
```

Use the smallest practical configuration.

---

# 42. Experiment Naming

Use explicit experiment identifiers.

For example:

```text
baseline_multitalker
clean_ft
overlap_ft
mixed_ft
lora_overlap_ft
qlora_overlap_ft
adaptive_finetuned
```

Do not use ambiguous names such as:

```text
experiment1
test2
new_model
final_model
```

---

# 43. Experiment Tracking

Each training experiment should generate a machine-readable result.

For example:

```text
outputs/metrics/
outputs/logs/
outputs/reports/
```

Store:

```json
{
  "experiment_id": "...",
  "model": "...",
  "training_strategy": "...",
  "dataset": "...",
  "seed": 42,
  "trainable_parameters": 0,
  "total_parameters": 0,
  "training_time": 0,
  "validation_metrics": {},
  "test_metrics": {},
  "streaming_metrics": {},
  "hardware": {}
}
```

Use actual values.

Do not manually fabricate a JSON result.

---

# 44. Ablation Study

Keep the initial ablation compact.

At minimum investigate:

### Data ablation

```text
clean only
overlap only
clean + overlap
```

### Training strategy

Where feasible:

```text
base
PEFT
full fine-tuning
```

Do not run full fine-tuning simply for completeness if it is computationally unreasonable.

---

# 45. Overlap Difficulty Analysis

If enough data exists, report performance by:

```text
overlap ratio
```

For example:

```text
low
medium
high
```

The exact thresholds must come from the existing Phase 3 configuration.

Do not invent new overlap categories without documenting them.

---

# 46. Generalization Analysis

If computationally feasible, evaluate on a dataset not generated using exactly the same mixture configuration.

Possible future/secondary datasets may include:

* LibriCSS
* AMI
* CHiME-6

However, do not make external datasets mandatory if they introduce major preprocessing or licensing problems.

If one is used, clearly separate:

```text
synthetic test results
```

from:

```text
real-world test results
```

Do not combine them into a single score.

---

# 47. Avoid Synthetic-Data Overfitting

Investigate whether the fine-tuned model simply learns the specific synthetic mixture characteristics.

At minimum discuss:

* source speakers
* overlap ratios
* mixing strategy
* SNR/gain
* timing patterns
* silence distribution
* acoustic diversity

If the model improves only on the synthetic test set, do not claim general overlap robustness.

---

# 48. Catastrophic Forgetting

Explicitly check whether overlap adaptation harms ordinary ASR.

Compare:

```text
Base clean WER
```

against:

```text
Fine-tuned clean WER
```

and similarly compare overlap performance.

If overlap improves but clean performance degrades, document the trade-off.

---

# 49. Thresholds and Routing

Do not retune OSD thresholds on the test set.

If fine-tuning changes branch behavior and thresholds need adjustment:

```text
validation set
```

must be used.

Keep test data untouched.

---

# 50. Research Integrity

Never fabricate:

* training results
* validation loss
* WER
* cpWER
* latency
* memory usage
* training speed
* parameter counts
* model compatibility
* PEFT compatibility
* benchmark improvements

Use:

```text
NOT MEASURED
```

when appropriate.

Use:

```text
NOT TECHNICALLY SUPPORTED
```

when a model/framework combination does not work.

Use:

```text
EXPERIMENT FAILED
```

when a legitimate experiment was attempted but failed.

Do not silently replace failed experiments with another configuration.

---

# 51. Scientific Interpretation

The experiment must distinguish:

### Model adaptation effect

```text
Base Multi-Talker
vs
Fine-Tuned Multi-Talker
```

### Routing effect

```text
Always Multi-Talker
vs
Adaptive Routing
```

### OSD effect

```text
Oracle OSD
vs
Predicted OSD
```

### Data effect

```text
Clean FT
vs
Overlap FT
vs
Mixed FT
```

Do not attribute an improvement to the wrong component.

---

# 52. Documentation

Update:

```text
README.md
docs/architecture.md
docs/experiments.md
docs/thesis_mapping.md
```

Create:

```text
docs/training_model_compatibility.md
docs/training_objective.md
docs/phase_reports/phase_8_report.md
```

Document:

* training strategy
* data preparation
* target format
* loss
* PEFT configuration
* hardware
* training procedure
* checkpoint management
* evaluation
* limitations

---

# 53. Phase 8 Report

Create:

```text
docs/phase_reports/phase_8_report.md
```

Include:

## 1. Objective

What training problem was addressed.

## 2. Model Compatibility

What the Phase 7 model supports.

## 3. Training Strategy

Why PEFT/full/partial fine-tuning was selected.

## 4. Dataset

Exact training/validation data.

## 5. Target Representation

How multi-talker targets were constructed.

## 6. Training Configuration

All relevant hyperparameters.

## 7. Parameter Counts

Total vs trainable parameters.

## 8. Experiments

All executed conditions.

## 9. Training Results

Actual results.

## 10. Clean ASR Results

Base vs fine-tuned.

## 11. Overlap Results

Base vs fine-tuned.

## 12. Streaming Results

Latency/RTF comparison.

## 13. Adaptive Results

Phase 7 vs Phase 8 adaptive pipeline.

## 14. Ablations

Data/training-strategy ablations actually performed.

## 15. Resource Usage

GPU/CPU/memory/training time.

## 16. Failure Analysis

Training and inference failures.

## 17. Limitations

Especially synthetic-data limitations.

## 18. Reproducibility

Exact commands/configurations.

## 19. Phase 9 Requirements

What remains for the final evaluation/ablation stage.

---

# 54. Required Experiment Table

The report should contain a table similar to:

| Experiment | Model | Training Data   | Strategy | Clean WER | Overlap Metric | RTF | Latency | GPU Memory |
| ---------- | ----- | --------------- | -------- | --------: | -------------: | --: | ------: | ---------: |
| Base       | ...   | None            | None     |       ... |            ... | ... |     ... |        ... |
| Clean FT   | ...   | Clean           | ...      |       ... |            ... | ... |     ... |        ... |
| Overlap FT | ...   | Overlap         | ...      |       ... |            ... | ... |     ... |        ... |
| Mixed FT   | ...   | Clean + Overlap | ...      |       ... |            ... | ... |     ... |        ... |

Only populate cells with actually measured values.

Use:

```text
N/A
```

or:

```text
NOT MEASURED
```

where appropriate.

---

# 55. Completion Criteria

Phase 8 is complete only if:

* model training compatibility was investigated
* training strategy was selected based on evidence
* training dataset pipeline works
* target representation is documented
* speaker leakage checks exist
* development smoke training succeeds
* at least one meaningful fine-tuning experiment is completed, if technically feasible
* checkpoint save/reload works
* evaluation works
* clean and overlap performance are separately measured
* streaming behavior is evaluated
* adaptive integration is tested
* previous phases remain functional
* tests pass
* `scripts/smoke_test.py` passes
* Kaggle notebook is updated
* README is updated
* training documentation exists
* phase report exists

If meaningful training is technically impossible with the selected model, Phase 8 can instead conclude with a documented **training feasibility study**, but do not pretend that a fine-tuning experiment was completed.

---

# 56. Final Agent Response

After completing Phase 8, stop.

Do not automatically start Phase 9.

Report:

1. model selected in Phase 7
2. training compatibility findings
3. selected training strategy
4. why that strategy was selected
5. PEFT/full/partial decision
6. dataset used
7. target representation
8. training configuration
9. trainable parameter count
10. total parameter count
11. experiments executed
12. training duration
13. GPU/CPU/memory usage
14. validation results
15. clean test results
16. overlap test results
17. streaming metrics
18. adaptive pipeline results
19. ablation results
20. checkpoint locations
21. tests
22. smoke-test status
23. notebook status
24. documentation status
25. failures and limitations
26. whether fine-tuning actually improved the system
27. exact recommendations for Phase 9

Clearly distinguish:

```text
IMPLEMENTED
```

from:

```text
TRAINED
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
EXPERIMENT FAILED
```

from:

```text
NOT TECHNICALLY FEASIBLE
```

from:

```text
FUTURE WORK
```

The central objective of Phase 8 is **not merely to produce a trained checkpoint**.

It is to establish, through controlled experiments, whether task-specific adaptation of the overlap-aware ASR branch provides a measurable benefit while preserving the project's fundamental requirements:

```text
English
Streaming
Overlapping Speakers
Low-Latency Operation
Adaptive Routing
Reproducibility
```

Do not proceed to Phase 9 until the training/fine-tuning results and their limitations are clearly documented.
