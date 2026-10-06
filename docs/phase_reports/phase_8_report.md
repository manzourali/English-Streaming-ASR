# Phase 8 — Training / fine-tuning the overlap-aware ASR system

Phase 8 should now move from inference-only overlap-aware ASR to training/fine-tuning the selected overlap-aware model, using the controlled synthetic overlap data established in Phase 3. The important constraint is that training should be driven by the actual model selected in Phase 7 rather than assuming that LoRA/QLoRA is automatically compatible.

## 1. Objective

Determine whether the selected SURT 2.0 branch can be adapted reproducibly with Phase 3 synthetic overlap data without inventing an unsupported training interface.

## 2–3. Model compatibility and strategy

The full compatibility study is in [training_model_compatibility.md](../training_model_compatibility.md). SURT 2.0 has official Icefall training recipes, but this checkout has only an inference decoder factory—not a pinned trainable recipe, checkpoint, BPE assets, or model graph. The selected Phase 8 strategy is therefore **training feasibility study / no project-local fine-tuning**. PEFT, LoRA, and QLoRA are **NOT TECHNICALLY SUPPORTED** here.

## 4–6. Dataset, targets, configuration

Phase 3 manifests are transformed into deterministic two-channel HEAT-style transcript targets without duplicating audio. The collator pads variable-length mono audio and preserves channel text targets for the exact external recipe tokenizer. Split validation rejects speaker, source, and mixture-audio leakage. Configuration exposes manifests, condition, duration cap, seed, precision, training schedule, checkpoint/resume, and external backend factory in `configs/training.yaml`.

## 7. Parameter counts

The training backend contract reports total, trainable, frozen, and trainable percentage from a real loaded model. No model was loaded locally, so all concrete parameter values are **NOT MEASURED**.

## 8–16. Experiments, results, resource usage, and failures

| Experiment | Model | Training data | Strategy | Clean WER | Overlap metric | RTF | Latency | GPU memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | SURT 2.0 | None | None | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Clean FT | SURT 2.0 | Clean | Not supported locally | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Overlap FT | SURT 2.0 | Synthetic overlap | Not supported locally | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Mixed FT | SURT 2.0 | Clean + overlap | Not supported locally | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

No external recipe, checkpoint, data manifest, GPU, training time, loss curve, parameter count, evaluation metric, streaming metric, adaptive result, or resource value was fabricated. The implemented failure path stops with `TRAINING NOT TECHNICALLY SUPPORTED` when `training.backend_factory` is absent.

## 17. Limitations

The training lifecycle is a contract, not a substitute for official SURT training. Its fake-backend tests validate checkpoint, resume, stability, and metadata behavior; they do not constitute model fine-tuning. Synthetic mixtures also remain controlled research data rather than a claim of real-meeting generalization.

## 18. Reproducibility

```bash
python3 scripts/train_asr.py --config configs/training.yaml --validate-only
python3 scripts/train_asr.py --config configs/training.yaml --override surt_training_override.yaml
```

The override must pin the Icefall/SURT revision, checkpoint, BPE/tokenizer assets, exact `training.backend_factory`, data manifests, hardware/precision choices, and any recipe-specific options.

## 19. Phase 9 requirements

Install and verify one exact official SURT 2.0 recipe/checkpoint, implement its backend factory and tokenization/loss bridge, conduct a small development run, reload its checkpoint, and then compare base, clean, overlap, and mixed conditions on clean, overlap, streaming, and adaptive tests without retuning on test data.

---

## Implemented Phase 8 as a defensible training-feasibility layer rather than fabricating an unsupported fine-tuning procedure.

SURT 2.0 has published Icefall training recipes and model releases, but this checkout has only an inference-oriented external decoder factory—not a pinned trainable recipe, tokenizer/BPE assets, checkpoint, or optimizer/loss graph. So the project now explicitly reports local PEFT/LoRA/QLoRA training as NOT TECHNICALLY SUPPORTED until an exact external SURT training backend is configured.

Implemented:

Phase 3 overlap-manifest → deterministic two-channel HEAT-style training targets.
Strict speaker, source-utterance, duplicate-audio, and split leakage detection.
Numpy-based variable-length audio collator with padding and masks.
External ExternalSurtTrainingBackend interface for a pinned Icefall/SURT recipe.
Parameter-count reporting, finite-loss checks, validation, early stopping, best/last checkpoints, and resume support.
scripts/train_asr.py
 with --validate-only and model-free feasibility handling.
Phase 8 model-free smoke test: scripts/smoke_test.py --phase 8.
Updated training configuration, Kaggle notebook, README, architecture/experiments/thesis notes, compatibility review, target-objective documentation, and Phase 8 report.

Validation passed:

python3 -m pytest -q → 40 passed, 1 skipped
python3 scripts/smoke_test.py --phase 8 --config configs/training.yaml
 → passed with training_smoke: true
Python compilation, notebook JSON validation, and diff checks passed.
No actual fine-tuning, parameter counts, WER, PI-WER, loss curves, latency, memory, or adaptation improvement was claimed or fabricated. A real run now requires an override that supplies valid Phase 3 manifests plus a version-pinned model.backend_factory wrapping the exact official Icefall/SURT recipe.
