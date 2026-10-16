# Kaggle E2E execution steps — English Streaming ASR with overlapped voices

**Run date:** October 7, 2026  
**Repository:** `/kaggle/working/English-Streaming-ASR`  
**Kaggle hardware verified:** Tesla T4, CUDA 12.8, PyTorch `2.11.0+cu128`

This is the exact practical path from the successful setup already completed in Kaggle to a scientifically valid end-to-end evaluation. Run every command from the repository root:

It records your completed Phase 0/3/5 results, exact VAD/routing commands, artifact persistence, WhisperRT baseline setup, real overlap generation, Phase 4–7 execution, SURT integration gate, and Phase 9 evaluation commands.
It explicitly distinguishes runnable commands from required external WhisperRT/SURT compatibility work; no unsupported model substitution is recommended.

```python
%cd /kaggle/working/English-Streaming-ASR
```

> **Important:** a complete end-to-end system is not available from commands alone yet. The repository has a concrete external-model integration gate: it needs a verified WhisperRT runtime/checkpoint and a version-pinned external SURT multi-talker decoder. Do not replace either with another model and call the result the planned system.

---

## 0. What is already complete

The following commands have succeeded in this Kaggle session:

| Item | Status | Evidence |
| --- | --- | --- |
| Python/package environment | Complete | Python 3.13.15, project import is OK, PyTorch and GPU are available. |
| GPU | Complete | Tesla T4 with CUDA 12.8 is available. |
| Phase 0 infrastructure | Complete | `scripts/smoke_test.py --phase 0` succeeded. |
| Phase 3 demo data | Complete | 12 deterministic synthetic mixtures: 4 each in train, validation, and test. |
| Phase 5 heuristic OSD | Complete for demo data | Test subset: precision `1.0000`, recall `0.6364`, F1 `0.7778`, OSD RTF `0.00866`. |

The current OSD score is a **small synthetic-demo baseline**, not a final real-world overlap-detection claim.

---

## 1. Immediate next commands: validate demo artifacts and VAD

These commands do not need Hugging Face access or a model download. They are the next safe actions.

### 1.1 Validate the Phase 3 manifests for Phase 4

```bash
!python scripts/run_overlap_baseline.py \
  --config configs/overlap_baseline.yaml \
  --validate-only
```

**Expected:** JSON with `status: "validated_only"`, `num_examples: 12`, `wer: "NOT_MEASURED"`, and `model_run: "NOT_RUN"`.

This proves that all 12 generated records can be consumed by the overlap-ASR benchmark. It deliberately does not load WhisperRT.

### 1.2 Run the standalone streaming VAD demo

```bash
!python scripts/run_vad.py --config configs/streaming_vad.yaml
!python scripts/smoke_test.py \
  --phase 2 \
  --real-vad \
  --config configs/streaming_vad.yaml
```

**Expected:** both commands complete and write a timestamped VAD run/metric record. The standalone demo is not a labeled VAD accuracy experiment, so do not report VAD precision/recall/F1 from it.

### 1.3 Validate the adaptive-routing and final-evaluation configurations

```bash
!python scripts/run_adaptive.py \
  --config configs/adaptive.yaml \
  --validate-only

!python scripts/evaluate.py \
  --config configs/evaluation.yaml \
  --validate-only
```

**Expected:** configuration validation only. No ASR model is loaded and no E2E score is produced.

---

## 2. Persist the work from this Kaggle session

Kaggle's `/kaggle/working` directory is temporary. Preserve the generated data and current result files before ending the session.

### 2.1 Inspect the two output roots

Phase 3 writes relative to the repository; some other scripts select Kaggle's working output root automatically.

```bash
!find /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap -maxdepth 3 -type f | sort
!find /kaggle/working/outputs -maxdepth 3 -type f 2>/dev/null | sort
```

### 2.2 Make a portable artifact bundle

```bash
!mkdir -p /kaggle/working/asr_phase3_phase5_artifact
!cp -a /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap \
  /kaggle/working/asr_phase3_phase5_artifact/
!cp -a /kaggle/working/outputs/phase5 \
  /kaggle/working/asr_phase3_phase5_artifact/ 2>/dev/null || true
!cp -a /kaggle/working/outputs/reports \
  /kaggle/working/asr_phase3_phase5_artifact/ 2>/dev/null || true
!cp configs/overlap_dataset.yaml configs/overlap_detection.yaml \
  /kaggle/working/asr_phase3_phase5_artifact/
!git rev-parse HEAD | tee /kaggle/working/asr_phase3_phase5_artifact/git_commit.txt
!find /kaggle/working/asr_phase3_phase5_artifact -maxdepth 3 -type f | sort
```

In Kaggle, use **Save Version** / create a private Kaggle Dataset from `asr_phase3_phase5_artifact`. In a future session attach it as an input, where Kaggle exposes it read-only under `/kaggle/input/<dataset-slug>/`.

---

## 3. Phase 1 gate: make the configured WhisperRT baseline runnable

The existing project adapter imports a Python module named `whisper_rt` and expects its causal streaming API. The repository's own dependency files do **not** install this runtime. The configured model is:

```text
Repository: MLSpeech/WhisperRT-Streaming
Checkpoint: small_300.pt
```

### 3.1 Obtain Hugging Face access securely

1. On Hugging Face, request/confirm access to `MLSpeech/WhisperRT-Streaming`.
2. In Kaggle, add your Hugging Face token as a notebook **Secret** named `HF_TOKEN`. Do not paste it into a cell, YAML file, output, or Git commit.
3. With Internet enabled, authenticate in a Kaggle cell:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
```

4. Confirm access without printing the secret:

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="MLSpeech/WhisperRT-Streaming",
    filename="small_300.pt",
)
print(checkpoint)
```

**Expected:** a local path to `small_300.pt`. The first download is approximately 1 GB, so do this once and preserve the model through a private Kaggle Dataset if future sessions must work offline.

### 3.2 Required compatibility check — do not skip

Before the Phase 1 command can be considered runnable, prove that the upstream package provides the exact API expected by `src/streaming_asr/models/whisperrt.py`:

```bash
!python -c "import whisper_rt; print(whisper_rt.__file__)"
!python -c "from whisper_rt.audio import SpectrogramStream; print(SpectrogramStream)"
!python -c "from whisper_rt import StreamingWhisper; print(StreamingWhisper)"
```

**Pass condition:** all three commands succeed.

**If any command fails:** stop here. This is not a Kaggle-installation problem; it is an external-runtime compatibility gap. Create a correction branch such as `correction/whisperrt-runtime-adapter`, then adapt `src/streaming_asr/models/whisperrt.py` to the official package's actual, version-pinned API. Add one real model-load/incremental-decode test. Do not silently substitute offline Whisper or another ASR package.

### 3.3 Install the official WhisperRT runtime only from its official release instructions

The upstream model/runtime must be installed according to the pinned official release that exposes the API checked in Step 3.2. Record all of the following in an experiment note before proceeding:

```bash
!python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
!python -c "import whisper_rt; print(whisper_rt.__file__)"
!python -m pip freeze | grep -Ei 'torch|whisper|careless|huggingface' | sort
```

Do not guess a package name based on the Hugging Face model repository. A successful `hf_hub_download()` only downloads weights; it does not prove that a compatible streaming decoder is installed.

### 3.4 Run the smallest real clean-speech WhisperRT baseline

After Step 3.2 passes, use exactly one LibriSpeech example first:

```bash
!python scripts/run_streaming.py \
  --config configs/whisperrt_baseline.yaml \
  --max-samples 1
```

**Expected outputs:**

```text
/kaggle/working/outputs/predictions/<run-id>.jsonl
/kaggle/working/outputs/metrics/<run-id>.json
/kaggle/working/outputs/reports/<run-id>.md
/kaggle/working/outputs/runs/<run-id>/
```

Check the output rather than accepting a zero-exit status blindly:

```bash
!ls -lt /kaggle/working/outputs/predictions | head
!ls -lt /kaggle/working/outputs/metrics | head
!ls -lt /kaggle/working/outputs/reports | head
```

Read the newest report. It must contain a non-empty prediction and measured model timing before expanding the sample count.

### 3.5 Establish a small clean baseline

After the one-sample command succeeds, use a deliberately small development subset:

```bash
!python scripts/run_streaming.py \
  --config configs/whisperrt_baseline.yaml \
  --max-samples 20
```

Save its config, report, metrics, prediction JSONL, Git commit, exact checkpoint path/hash, installed package versions, GPU name, and seed. This is the clean baseline used to interpret later overlap/routing experiments.

---

## 4. Phase 3: create a persistent real overlap dataset

Do not use the 12-example demo as the final research dataset. It is only an integration fixture.

### 4.1 Create a small explicit development override

In a Kaggle cell, create an uncommitted override file:

```python
from pathlib import Path

Path("kaggle_overlap_dev.yaml").write_text("""\
generation:
  num_samples: 20
  max_source_samples: 100
output:
  root: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev
""")
```

### 4.2 Generate, validate, and inspect it

```bash
!python scripts/generate_overlap.py \
  --config configs/overlap_dataset.yaml \
  --override kaggle_overlap_dev.yaml

!find outputs/synthetic_overlap_dev -maxdepth 3 -type f | sort | sed -n '1,80p'
```

**Expected:** WAV files, `manifests/train.jsonl`, `manifests/validation.jsonl`, `manifests/test.jsonl`, and `statistics.json`. Stop if validation errors are non-empty.

### 4.3 Publish the dataset before larger experiments

Create a private Kaggle Dataset with this exact layout:

```text
synthetic-overlap-dev-v1/
├── audio/
├── manifests/
└── statistics.json
```

In all later sessions, attach it and point configuration overrides to `/kaggle/input/<your-dataset-slug>/` rather than regenerating it.

---

## 5. Phase 4: real single-stream WhisperRT overlap baseline

This measures what the single-speaker WhisperRT stream does on overlapped audio. It is not multi-talker transcription.

### 5.1 Create a Phase 4 Kaggle override

```python
from pathlib import Path

Path("kaggle_phase4.yaml").write_text("""\
data:
  manifest_root: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev
  conditions: [test]
""")
```

If you attached the published dataset in a later session, replace that path with `/kaggle/input/<your-dataset-slug>/`.

### 5.2 Validate, then run one mixture

```bash
!python scripts/run_overlap_baseline.py \
  --config configs/overlap_baseline.yaml \
  --override kaggle_phase4.yaml \
  --validate-only

!python scripts/run_overlap_baseline.py \
  --config configs/overlap_baseline.yaml \
  --override kaggle_phase4.yaml \
  --max-samples 1
```

Only then run the whole development test split:

```bash
!python scripts/run_overlap_baseline.py \
  --config configs/overlap_baseline.yaml \
  --override kaggle_phase4.yaml
```

Persist the Phase 4 JSONL, metrics, report, override, and source dataset version.

---

## 6. Phase 5: score OSD on the real generated dataset

### 6.1 Create an OSD override and run it

```python
from pathlib import Path

Path("kaggle_osd.yaml").write_text("""\
data:
  manifest_root: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev
  splits: [test]
""")
```

```bash
!python scripts/evaluate_osd.py \
  --config configs/overlap_detection.yaml \
  --override kaggle_osd.yaml
```

Record precision, recall, F1, detection latency, RTF, thresholds, and the test manifest version. The current OSD is heuristic; use this as a controlled baseline rather than claiming a learned overlap detector.

---

## 7. Phase 6: adaptive streaming pipeline controls

Phase 6 must run all three modes on the same test data:

```text
always_normal  = deployable baseline with no overlap routing benefit
oracle         = non-deployable upper-bound reference using known synthetic timing
predicted      = deployable candidate using the heuristic OSD
```

### 7.1 Create a routing override

```python
from pathlib import Path

Path("kaggle_adaptive.yaml").write_text("""\
data:
  manifest: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev/manifests/test.jsonl
  max_samples: null
branches:
  overlap: shared_normal
""")
```

`shared_normal` means both routes use the same WhisperRT stream. This is a routing-control experiment, **not** the final multi-talker system.

### 7.2 Validate, run one mixture, then run all test mixtures

```bash
!python scripts/run_adaptive.py \
  --config configs/adaptive.yaml \
  --override kaggle_adaptive.yaml \
  --validate-only

!python scripts/run_adaptive.py \
  --config configs/adaptive.yaml \
  --override kaggle_adaptive.yaml \
  --max-samples 1 \
  --modes always_normal oracle predicted

!python scripts/run_adaptive.py \
  --config configs/adaptive.yaml \
  --override kaggle_adaptive.yaml \
  --modes always_normal oracle predicted
```

Expected files include `*_always_normal_routing.jsonl`, `*_oracle_routing.jsonl`, `*_predicted_routing.jsonl`, route-event JSON, metrics, and an adaptive report. Persist every mode. Never present the oracle row as deployable performance.

---

## 8. Required E2E gate: integrate an external SURT multi-talker recognizer

This is the gate that turns the routing control into the intended E2E system for overlapped voices.

### 8.1 What must exist before running Phase 7

1. A version-pinned official SURT/Icefall source checkout.
2. A compatible pretrained SURT checkpoint.
3. Matching BPE/tokenizer assets and any model configuration files.
4. A tested function exposed through `model.decoder_factory`.
5. The function must return a callable with this exact project contract:

```python
callable(samples: numpy.ndarray, sample_rate: int) -> streams
```

The `streams` value must be a string, a list of strings, or a list of mappings containing stream text. Output channels are unordered recognition streams; they are not speaker identities.

### 8.2 Build and test the factory on a correction branch

Do not edit `master` directly for an external runtime integration:

```bash
!git switch -c correction/surt-runtime-adapter
```

Add a small module in the project (for example `streaming_asr.integrations.surt:build_decoder`) that loads the pinned external recipe/checkpoint once and returns the callable. Add an integration test that decodes one short overlap mixture incrementally. The repository currently does not contain this factory; it cannot be invented only from YAML.

### 8.3 Create the private Phase 7 override after the factory works

```python
from pathlib import Path

Path("kaggle_surt.yaml").write_text("""\
data:
  manifest: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev/manifests/test.jsonl
model:
  checkpoint: /kaggle/input/<your-surt-checkpoint-dataset>/<checkpoint-file>
  decoder_factory: streaming_asr.integrations.surt:build_decoder
""")
```

Replace both angle-bracket values with real attached Kaggle Dataset paths. Do not put the checkpoint in Git.

### 8.4 Validate and run Phase 7

```bash
!python scripts/run_multitalker.py \
  --config configs/multitalker.yaml \
  --override kaggle_surt.yaml \
  --validate-only

!python scripts/run_multitalker.py \
  --config configs/multitalker.yaml \
  --override kaggle_surt.yaml \
  --max-samples 1

!python scripts/run_multitalker.py \
  --config configs/multitalker.yaml \
  --override kaggle_surt.yaml
```

The current adapter is **low-latency windowed** unless the external decoder exposes verified native causal state. Report it with that label unless you have demonstrated native stateful streaming.

---

## 9. Replace Phase 6's control overlap branch with SURT

After the Phase 7 direct run is valid, route detected overlap to SURT instead of the shared WhisperRT control branch.

```python
from pathlib import Path

Path("kaggle_adaptive_surt.yaml").write_text("""\
data:
  manifest: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev/manifests/test.jsonl
branches:
  overlap: surt2
multitalker:
  decoder_factory: streaming_asr.integrations.surt:build_decoder
  checkpoint: /kaggle/input/<your-surt-checkpoint-dataset>/<checkpoint-file>
""")
```

Run the same controls:

```bash
!python scripts/run_adaptive.py \
  --config configs/adaptive.yaml \
  --override kaggle_adaptive_surt.yaml \
  --max-samples 1 \
  --modes always_normal oracle predicted

!python scripts/run_adaptive.py \
  --config configs/adaptive.yaml \
  --override kaggle_adaptive_surt.yaml \
  --modes always_normal oracle predicted
```

This is the first intended E2E system path:

```text
16 kHz audio chunks → VAD → heuristic OSD → adaptive router
                                      ├→ WhisperRT normal branch
                                      └→ SURT overlap branch
                                               ↓
                              incremental transcript streams + routing traces
```

---

## 10. Optional Phase 8: fine-tune SURT only with a real external trainer

Do **not** start this stage until the external Icefall/SURT recipe's official training command, checkpoint format, BPE assets, optimizer/scheduler resume state, and exact source revision are known.

First validate the project's manifest/target contract:

```python
from pathlib import Path

Path("kaggle_training.yaml").write_text("""\
dataset:
  train_manifest: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev/manifests/train.jsonl
  validation_manifest: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev/manifests/validation.jsonl
  test_manifest: /kaggle/working/English-Streaming-ASR/outputs/synthetic_overlap_dev/manifests/test.jsonl
""")
```

```bash
!python scripts/train_asr.py \
  --config configs/training.yaml \
  --override kaggle_training.yaml \
  --validate-only
```

For a real train/resume run, `model.backend_factory` must wrap the exact verified external recipe. Persist model weights, adapter weights (if any), optimizer, scheduler, trainer state, epoch, global step, random state, config, tokenizer/BPE assets, and manifest hashes. Without those, you can run inference but cannot reliably resume training.

---

## 11. Phase 9: freeze and evaluate comparable systems

Evaluation needs per-example `EvaluationRecord` JSONL files produced by comparable runs. Do not use a Phase 4-specific JSONL or routing trace unless it matches the `EvaluationRecord` schema expected by `scripts/evaluate.py`.

### 11.1 Freeze the actual assets before touching the final test set

Create an override only after choosing the final checkpoints and datasets:

```python
from pathlib import Path

Path("kaggle_evaluation.yaml").write_text("""\
frozen_system:
  whisperrt_checkpoint: /kaggle/input/<your-whisperrt-dataset>/small_300.pt
  surt_checkpoint: /kaggle/input/<your-surt-checkpoint-dataset>/<checkpoint-file>
  finetuned_surt_checkpoint: null
  train_manifest: /kaggle/input/<your-overlap-dataset>/manifests/train.jsonl
  validation_manifest: /kaggle/input/<your-overlap-dataset>/manifests/validation.jsonl
  test_manifest: /kaggle/input/<your-overlap-dataset>/manifests/test.jsonl
""")
```

### 11.2 Validate the freeze

```bash
!python scripts/evaluate.py \
  --config configs/evaluation.yaml \
  --override kaggle_evaluation.yaml \
  --validate-only
```

### 11.3 Aggregate final per-example records

After saving valid records for clean WhisperRT, overlap WhisperRT, base/fine-tuned SURT (if applicable), and predicted adaptive routing, run:

```bash
!python scripts/evaluate.py \
  --config configs/evaluation.yaml \
  --override kaggle_evaluation.yaml \
  --records \
    /kaggle/working/outputs/predictions/final_records.jsonl

!python scripts/benchmark.py \
  --records /kaggle/working/outputs/predictions/final_records.jsonl
```

Replace `final_records.jsonl` with the real consolidated `EvaluationRecord` file(s). The final report belongs in `outputs/reports/`; save it together with every source JSONL and frozen override.

---

## 12. Final deliverables checklist

Before claiming a completed E2E system, create a private versioned Kaggle Dataset containing:

```text
e2e-asr-overlap-experiment-v1/
├── configs/                         # Every base + override YAML used
├── manifests/                       # train/validation/test JSONL and hashes
├── checkpoints/                     # Only if license/permissions permit
├── predictions/                     # Per-example JSONL records
├── metrics/                         # Raw metric JSON
├── reports/                         # Phase 1/4/5/6/7/9 reports
├── environment/                     # pip freeze, GPU, Python, CUDA details
├── git_commit.txt
└── README.md                        # Exact command order and known limitations
```

The final scientific comparison must contain at least:

| System | Required? | Notes |
| --- | --- | --- |
| Clean WhisperRT | Yes | Establishes the clean-speech baseline. |
| WhisperRT on synthetic overlap | Yes | Shows the single-stream baseline under overlap. |
| Adaptive `always_normal` | Yes | Deployable routing-control baseline. |
| Adaptive `predicted` | Yes | Actual deployable OSD-driven candidate. |
| Adaptive `oracle` | Yes, but non-deployable | Upper-bound diagnostic only. |
| Direct external SURT | Yes for E2E multi-talker claim | Evaluate with permutation-invariant WER. |
| Adaptive WhisperRT + SURT | Yes for E2E claim | The target routed system. |
| Fine-tuned SURT | Optional | Include only after a reproducible external training run. |

## Next command to run now

Run this first:

```bash
!python scripts/run_overlap_baseline.py \
  --config configs/overlap_baseline.yaml \
  --validate-only
```

Then complete Sections 1.2 and 1.3. Do not start the WhisperRT/SURT stages until you have passed the explicit compatibility gates in Sections 3 and 8.
