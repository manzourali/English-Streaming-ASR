# Kaggle execution tutorial

This is the practical guide for running this repository from a fresh Kaggle notebook. Read it before running a model command: the checked-in code is usable, but the repository does **not** bundle the external WhisperRT runtime/checkpoint, a SURT decoder/checkpoint, or final experiment artifacts.

## The short answer

| Question | Answer |
| --- | --- |
| Should I reset to Phase 0 or Phase 1? | **No.** Use the current `master` HEAD; it contains all earlier code. |
| Should I create a correction branch now? | **No.** Create one only if a concrete run finds a regression or an incompatible external runtime. |
| Can I run this repository in Kaggle? | **Yes** for installation checks, Phase 0, demo overlap generation, VAD, heuristic OSD, and validation commands. Real WhisperRT/SURT experiments need external prerequisites below. |
| First experiment | The small, model-free Phase 0 smoke test, then the deterministic Phase 3 demo. |
| First project notebook | `notebooks/00_environment.ipynb`; run it from the Kaggle UI or run the equivalent `scripts/smoke_test.py` command. |
| What must be fixed first? | Nothing for the smoke test. Before Phase 1, install the official `whisper_rt` runtime and authenticate to Hugging Face for the WhisperRT checkpoint. |

The current HEAD is an implementation and reproducibility foundation, **not** a measured end-to-end research result. It correctly refuses to substitute a different model when required external components are absent.

## Simple project status

| Phase | What this phase does | Current code exists? | Can I run it now? | Depends on | Missing | What I need to do |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Checks project imports, configuration, chunk buffering, and dummy pipeline state. | Yes | Yes | Core Python packages | Nothing | Run the smoke test first. |
| 1 | Loads clean LibriSpeech and sends successive audio chunks to WhisperRT. | Yes | Not until external setup is complete | PyTorch, `whisper_rt`, Hub checkpoint, LibriSpeech | Official runtime/checkpoint access and a real run | Install/authorize, then run one sample. |
| 2 | Applies WebRTC VAD to successive audio frames. | Yes | Yes for demo/WAV input | `webrtcvad-wheels` | Labeled real-audio evaluation data for accuracy claims | Run demo, then add valid labels if measuring F1. |
| 3 | Mixes two source utterances and writes audio, transcripts, timing, and JSONL manifests. | Yes | Yes for deterministic demo | Demo fixtures or LibriSpeech | Persistent real generated corpus | Run demo; generate/publish real data only when ready. |
| 4 | Sends synthetic overlap audio to the single-stream WhisperRT baseline. | Yes | Validation only without WhisperRT | Phase 3 manifest + WhisperRT | Real model execution | Validate manifest, then run after Phase 1 works. |
| 5 | Detects likely overlap from audio features and scores against synthetic timing labels. | Yes | Yes after Phase 3 | Phase 3 manifest | Learned/real-data OSD validation | Run it as a heuristic baseline. |
| 6 | Routes chunks between normal/overlap logical branches using VAD/OSD. | Yes | Validation only without assets | Phase 3, WhisperRT, optionally Phase 5 | Real ASR assets; separate overlap recognizer | Run controls after Phase 1 and 3. |
| 7 | Calls a supplied multi-talker recognizer and emits unordered transcript channels. | Adapter only | No real run until external setup | Phase 3 + external SURT decoder/checkpoint | Pinned decoder factory/runtime | Integrate and verify the official external recipe. |
| 8 | Validates training manifests and defines an external training lifecycle. | Validation code only | Validate-only | Phase 3 split manifests + external training backend | Pinned Icefall/SURT trainer/checkpoint/tokenizer | Do not train until the backend is supplied. |
| 9 | Aggregates already-produced comparable per-example evaluation records. | Yes | Yes for protocol/empty plan; results need records | Saved prediction JSONL | Real system records | Freeze config only after experiments. |
| 10 | Audits provenance and reproducibility materials. | Yes | Yes | Existing project artifacts | Final measured artifacts | Run after actual experiments. |

## Do I need to go back?

| Question | Answer |
| --- | --- |
| Should I reset to Phase 0? | No. |
| Should I reset to Phase 1? | No. |
| Should I use current HEAD? | Yes. |
| Should I create a correction branch? | Not now; do so only for a confirmed defect/regression. |
| Can I run the current repository in Kaggle? | Yes for supported validations and demos; model-dependent runs need their external prerequisites. |
| What is the first experiment I should run? | Phase 0 smoke test, then deterministic Phase 3 overlap generation. |
| What is the first notebook I should run? | `notebooks/00_environment.ipynb`, or the equivalent smoke-test script. |
| What must be fixed before that? | Nothing. Before Phase 1, install/verify WhisperRT and Hub access. |

You do not need to rerun old Git history to reproduce a phase. A phase is an experiment/configuration path within the current codebase, not a command to check out an old commit.

## 1. Start a new Kaggle notebook

1. In Kaggle, create a new **Notebook**.
2. In the notebook settings, enable **Internet** before commands that download Python packages, Hugging Face data, or model files. Enable **GPU** only when running WhisperRT or an external SURT recipe; the first smoke tests do not need it.
3. Run this first cell:

```bash
!git clone https://github.com/manzourali/English-Streaming-ASR.git
```

4. Run this Python cell:

```python
%cd /kaggle/working/English-Streaming-ASR
```

5. Install the project and its declared phase extras. `requirements.txt` alone installs only the deliberately lightweight base set, so use the editable project install as well:

```bash
!python -m pip install -q -r requirements.txt
!python -m pip install -q -e ".[audio,phase1,phase2]"
```

`-e .` means “use the source code in this cloned folder,” so edits and scripts use the repository you cloned. The command installs `soundfile`, `datasets`, `huggingface_hub`, and WebRTC VAD, but it does **not** install PyTorch or the separately distributed WhisperRT runtime.

6. Check the installation:

```bash
!python scripts/check_environment.py
!python scripts/smoke_test.py --phase 0 --config configs/base.yaml
```

**Expected result:** `Project import: OK`, the required packages are installed, and the smoke test writes a small record below `outputs/`. A GPU is not required.

**If this fails:** confirm `%cd` printed `/kaggle/working/English-Streaming-ASR`, rerun the two install commands, then run `!python scripts/check_environment.py --json` to see the missing package. Do not run a notebook from outside the repository root.

## Kaggle paths and persistence

| Path | Meaning | How this project uses it |
| --- | --- | --- |
| `/kaggle/working/` | Writable temporary disk for the current session. | Clone code to `/kaggle/working/English-Streaming-ASR`; default Kaggle paths put generated data, outputs, and checkpoints here. |
| `/kaggle/input/` | Read-only files from Kaggle Datasets attached in the right-side **Add Input** panel. | Attach a previously published synthetic-overlap artifact, checkpoint, or local model file here. Point a YAML override at its path. |
| Hugging Face cache | Package-managed download cache, normally under the current user's cache directory. | `datasets` caches LibriSpeech; `hf_hub_download` caches `small_300.pt`. It is not a durable experiment record. |
| `outputs/` | Writable, Git-ignored run results. | Scripts write `runs/`, `metrics/`, `predictions/`, and `reports/`. |
| `checkpoints/` | Writable, Git-ignored model state. | Intended for an external training backend; this checkout does not create a real SURT training checkpoint itself. |

Kaggle can end or reset a session. Before ending one, publish/download any result you need later; never assume `/kaggle/working` survives.

## Git commit, notebook, experiment, dataset, and checkpoint

- **Git commit:** a saved version of code. `master` at the current HEAD is the code starting point.
- **Project notebook:** an `.ipynb` file in this repository. It is a guide/demo and does **not** run merely because you cloned the repository.
- **Kaggle notebook:** the interactive document open in the Kaggle UI. It is where you run commands and can open/copy project notebooks.
- **Experiment:** one particular execution using a chosen config, dataset subset, model revision, seed, and hardware.
- **Dataset:** the audio and metadata supplied to an experiment, such as LibriSpeech or generated overlap mixtures.
- **Checkpoint:** saved learned model state. It is needed to resume training; it is not the same thing as source code or a dataset.

## How to use repository notebooks

The files in `notebooks/` are not automatically connected to the Kaggle UI. Their code cells are mostly setup and commented shell commands, not a standalone pipeline.

**Recommended workflow:** use this tutorial's shell commands in your Kaggle notebook. They are easier to restart, log, and adapt with YAML overrides. Open a project notebook through the Kaggle file browser only when you want its explanatory text; copy/run its cells after `cd`-ing to the repository root. Do **not** use `%run` on `.ipynb` files, and do not use `jupyter nbconvert --execute` as the primary workflow: several notebooks intentionally contain commented commands and require private overrides/external assets.

## Dataset setup

### A. Download directly from Hugging Face

| Dataset | Purpose | Source / ID | How obtained | Expected location | Storage / regeneration | Needed next? |
| --- | --- | --- | --- | --- | --- | --- |
| LibriSpeech ASR | Clean speech for the Phase 1 WhisperRT baseline; source utterances for real Phase 3 mixtures. | Hugging Face `openslr/librispeech_asr`, configured as `clean`. | `datasets.load_dataset()` inside `iter_librispeech`; Internet enabled. | Hugging Face cache; generated files go to `outputs/synthetic_overlap/`. | Large if full splits are requested; cache can be recreated. | Needed for real Phase 1 or non-demo Phase 3. |

The configured loader is `streaming_asr.datasets.loaders.iter_librispeech`. It calls Hugging Face's `load_dataset()` using the IDs in `configs/whisperrt_baseline.yaml`, `configs/streaming_vad.yaml`, and `configs/overlap_dataset.yaml`.

```text
Hugging Face → datasets.load_dataset() → local HF cache → project loader → experiment
```

LibriSpeech is a public dataset. The repository does not configure a token for it; use a Hugging Face token only if the platform asks for one or your account/network requires authentication.

### B. Attach as a Kaggle Dataset

Create a private Kaggle Dataset after a successful costly generation or when carrying assets to a new session. Good candidates are:

```text
synthetic-overlap-v1/
├── audio/
│   ├── train/
│   ├── validation/
│   └── test/
├── manifests/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── statistics.json
```

This is appropriate because Phase 4–8 consume the manifests and audio repeatedly. Attach it at `/kaggle/input/<your-dataset-slug>/`, then create a small uncommitted YAML override pointing `data.manifest_root` or `data.manifest` to that read-only location. Do not expect the default `data/manifests/*.jsonl` paths to exist: this checkout intentionally contains only `.gitkeep` files.

If you have a WhisperRT `.pt` checkpoint or a verified external SURT checkpoint, attach it as a separate private Kaggle Dataset and set `model.checkpoint` in an override. Do not copy a large checkpoint into Git.

### C. Generate inside Kaggle

First prove the generator without any download:

```bash
!python scripts/generate_overlap.py --config configs/overlap_dataset.yaml --demo
!python scripts/run_overlap_baseline.py --config configs/overlap_baseline.yaml --validate-only
!python scripts/evaluate_osd.py --config configs/overlap_detection.yaml --max-samples 4
```

The demo writes deterministic WAVs, manifests, and statistics under `outputs/synthetic_overlap/`. For real mixtures, remove `--demo`; only do that after confirming LibriSpeech access and choosing an intentionally small `generation.num_samples`/`max_source_samples` override.

### D. Optional external evaluation datasets

No external conversational-overlap evaluation dataset is configured in this checkout. `configs/evaluation.yaml` leaves `evaluation.external_dataset: null`. Do not attach a dataset merely because it is mentioned in a research paper; add it only after defining its protocol and a compatible manifest/record conversion.

## Model setup

| Model | Phase | Purpose | Source / ID | Download / path | GPU | Kaggle status | Tested in this checkout? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WhisperRT | 1, 4, 6 | Incremental single-stream ASR. | Hugging Face `MLSpeech/WhisperRT-Streaming`, file `small_300.pt`. | `hf_hub_download()` unless `model.checkpoint` is set. Official package exposes module `whisper_rt`. | Recommended/required in practice for meaningful runs; exact VRAM is **not measured here**. | Requires separate runtime install, Hub access, and checkpoint download/attachment. | Adapter and tests: yes; real benchmark: no. |
| WebRTC VAD | 2, 6 | Voice-activity decisions, not ASR. | Python package `webrtcvad-wheels`. | Installed by `.[phase2]`. | No. | Ready. | Small integration path tested. |
| Heuristic OSD | 5, 6 | Causal overlap detector. | Repository code; no learned weights. | `src/streaming_asr/models/overlap_detector.py`. | No. | Ready for synthetic-manifest runs. | Demo evaluated only. |
| SURT 2.0-compatible decoder | 7, 8, 6 overlap branch | Multi-talker recognition. | External Icefall/SURT recipe and assets, not bundled. | A private version-pinned `model.decoder_factory` and checkpoint are required. | Likely yes; exact need is **not measured here**. | Not ready until you provide a verified external integration. | No real decoder run. |

### WhisperRT setup and minimal run

The adapter downloads `small_300.pt` with `huggingface_hub.hf_hub_download()` and calls `whisper_rt.load_streaming_model(...)`. It then feeds 16 kHz chunks through `SpectrogramStream` and `StreamingWhisper.decode`; it does not call offline Whisper or reload the model per chunk. This is implementation evidence of streaming-style chunk processing, not a real-time performance claim.

The project does not declare the official runtime in `requirements.txt` or `pyproject.toml`. Before Phase 1, install the exact official package/API version documented by the WhisperRT release/model card, then verify this import:

```bash
!python -c "import torch, whisper_rt; print(torch.__version__); print(whisper_rt.__file__)"
```

The WhisperRT model card currently states that Hugging Face token access is required. In Kaggle, add the token using the notebook's secrets mechanism, expose it only to the notebook, and log in with the official Hugging Face CLI/API method before the first model download. Never print the token in notebook output.

Use a one-example baseline only after this import and Hub access work:

```bash
!python scripts/run_streaming.py --config configs/whisperrt_baseline.yaml --max-samples 1
```

Expected outputs are timestamped JSONL predictions in `outputs/predictions/`, metrics in `outputs/metrics/`, a Markdown report in `outputs/reports/`, and run metadata under `outputs/runs/`. A successful run is the proof that the external runtime/model combination works; it does not by itself prove real-time throughput.

## Recommended first-day workflow

1. Create Kaggle notebook, enable Internet, clone the repository, and `%cd` into it.
2. Install `requirements.txt` and `-e ".[audio,phase1,phase2]"`.
3. Run `scripts/check_environment.py` and Phase 0 `scripts/smoke_test.py`.
4. Open/read `notebooks/00_environment.ipynb` if you want the equivalent demo.
5. Run the Phase 3 deterministic demo and validate it with the Phase 4 command above.
6. Run VAD safely without an external dataset: `!python scripts/run_vad.py --config configs/streaming_vad.yaml`, then optionally `!python scripts/smoke_test.py --phase 2 --real-vad --config configs/streaming_vad.yaml`.
7. Run Phase 5 heuristic OSD on the generated demo manifest.
8. Only then set up WhisperRT, run one clean LibriSpeech baseline example, and persist its outputs.
9. Generate/attach a real Phase 3 dataset before Phase 4, Phase 6, Phase 7, or Phase 8.
10. Do not start SURT training until an exact external Icefall/SURT backend, checkpoint, tokenizer/BPE assets, and resume contract are verified.

## Notebook 01: status and action

`notebooks/01_dataset_validation.ipynb` contains one Markdown cell: “Reserved for a later phase.” It is an actual placeholder, not an executed dataset validator.

| Question | Finding |
| --- | --- |
| Does the project depend on it? | No. Scripts load/validate data directly. |
| Is functionality elsewhere? | Yes: LibriSpeech loading is in `datasets/loaders.py`; overlap manifest/file validation is in `datasets/validators.py`; `generate_overlap.py` validates generated records. |
| Does it block Kaggle? | No. |
| Should it be run? | No; skip it. |
| Should it be implemented? | **REQUIRED FOLLOW-UP only if an interactive dataset-validation tutorial is desired.** It should load a deliberately small configured LibriSpeech subset, inspect sample rate/transcript fields, and validate a supplied overlap manifest without generating data. |

## Running the complete end-to-end system

```text
Audio chunks → VAD → heuristic OSD → routing policy → WhisperRT normal branch
                                        └──────────→ external SURT overlap branch
                                                       ↓
                                           incremental transcripts → evaluation
```

### Currently available

- Timestamped audio buffering, VAD backends, heuristic OSD, routing state machine, WhisperRT adapter, synthetic overlap manifests, record aggregation, and an external SURT adapter interface.
- Phase 6 can run its control branch only when WhisperRT and a valid Phase 3 manifest are supplied. Its default “overlap” route is the same WhisperRT stream, not multi-talker ASR.

### Required before true end-to-end multi-talker ASR

1. A tested official WhisperRT runtime/checkpoint and a real clean baseline run.
2. A persistent real Phase 3 overlap dataset/manifests.
3. A pinned external SURT 2.0 recipe/checkpoint/BPE assets plus `decoder_factory` that meets this project's contract.
4. For fine-tuning, a pinned external `training.backend_factory`; this repository's Phase 8 intentionally cannot train SURT by itself.
5. Comparable saved per-example `EvaluationRecord` JSONL files before Phase 9 aggregation.

## Checkpoints and Kaggle-session strategy

```text
Kaggle session 1 → run a small experiment → save outputs/checkpoint
                 → publish/download a versioned Kaggle Dataset
Kaggle session 2 → attach that Dataset → point an override at it → continue
```

For inference, persist the exact config override, Git commit, model revision/checkpoint, input manifest, prediction JSONL, metrics/report, and any required local model file. For training, persist all of those **plus** model weights, adapter weights (if used), optimizer state, scheduler state, trainer state, epoch, global step, random state, and tokenizer/BPE assets. Weights alone can reproduce inference but usually cannot resume optimizer training faithfully.

The current scripts create timestamped files under `outputs/runs/`, `outputs/predictions/`, `outputs/metrics/`, and `outputs/reports/`; the external training backend controls the actual checkpoint directory. The more elaborate `outputs/experiments/EXP_...` tree is a useful future convention, not a structure currently created by this code.

## Troubleshooting

| Symptom | Relevant cause and action |
| --- | --- |
| `ModuleNotFoundError: streaming_asr` | Run `%cd /kaggle/working/English-Streaming-ASR` and install `-e .`; scripts themselves add `src/` to `sys.path`. |
| `WhisperRT requires torch, huggingface_hub, and the official WhisperRT package` | Install PyTorch suitable for the Kaggle image, install the verified official WhisperRT package, and rerun the import check. The project does not vendor that runtime. |
| Hugging Face 401/403 for WhisperRT | Enable Internet and authenticate with an approved Kaggle secret/token; confirm access on the model page. Do not put the token in YAML or Git. |
| LibriSpeech download fails | Enable Internet, use the configured public dataset ID, and try a small source limit first. Attach a prepared dataset if network access is unavailable. |
| Manifest path not found | Default `data/manifests/*.jsonl` files are absent by design. Generate Phase 3 data or set an absolute `/kaggle/input/...` manifest path in an override. |
| GPU unavailable/CUDA mismatch | Select GPU in Kaggle settings before launching the session; inspect `scripts/check_environment.py`. Do not install arbitrary CUDA builds over Kaggle's matching PyTorch unless the external model instructions require it. |
| Out of memory | Lower `--max-samples`, reduce Phase 3 generation limits, use the small WhisperRT checkpoint, and run one branch at a time. No repository VRAM benchmark exists. |
| Output disappeared next session | Publish/download the relevant `outputs/` and checkpoint folders as a Kaggle Dataset before stopping the session. |
| SURT command fails before decoding | Expected until a valid pinned `decoder_factory`/checkpoint is supplied. Do not replace it with WhisperRT and call it multi-talker ASR. |

For the notebook-by-notebook plan, use [KAGGLE_NOTEBOOK_GUIDE.md](KAGGLE_NOTEBOOK_GUIDE.md). For phase dependencies and the operational checklist, use [phase_dependency_map.md](phase_dependency_map.md) and [execution_matrix.md](execution_matrix.md).
