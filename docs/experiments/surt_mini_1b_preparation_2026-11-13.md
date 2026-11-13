# Persian mini 1B — local preparation and 1A review

Date: 2026-11-13. Gate 1A: **accepted for bounded English baseline**.
Gate 1B: **prepared locally; NOT passed; real Kaggle execution/review pending**.

The user explicitly confirmed listening to both English clips and their mixture,
checking the references and the two decoded branches, in this session on 2026-11-13.
All 44 entries in `outputs/surt-base-1a-review/review-sha256.json` match their files.
Checkpoint contract: 597 state tensors, 26,675,639 parameters, no missing/unexpected
keys or shape mismatches. Official Base checkpoint SHA256:
`919bb9454526fb9ad963eb5a4168940a43c3b8331017a1c9ade477197c7072d4`.
The archived token check, finite/strict load and process completion succeeded.
Clean sentence PI-WER counts are 0/8 and 1/12; mixture 4/20 (20%, substitutions).
Whole-batch greedy decode, not incremental streaming or an official LibriCSS benchmark.
The archive's pending-human-review marker predates this confirmation and is preserved.
No 1A assets or archived files were edited.

Only small public HF metadata and the pinned README were read locally, using the
system curl certificate store. Dataset revision was actually returned by the HF API:
`5dc7ac366099292d4fbf079c99d58c1b2fba602d`, public and ungated on inspection.
The pinned README describes an **unofficial community-filtered subset** and declares
**Public Domain, CC-0** in its body; there is no structured license field.
This is uploader-declared provenance/license, not independent verification of every clip.
Counts/schema and primary-source addresses are recorded in `configs/surt_mini/pins.json`.
The previously untracked `.mini-source/` is user material and remains untouched.

No Mac audio/model/corpus download, scientific/GPU package installation or real model
execution is part of this preparation. The new path uses only already frozen packages:
urllib for bounded individual viewer assets, soundfile 0.14.0/libmpg123 for MP3,
SciPy 1.13.1 `resample_poly`, NumPy 1.26.4, SentencePiece 0.2.3 and Lhotse 1.33.0.
No datasets/torchcodec/FFmpeg installation or automatic Torch/CUDA upgrade is needed.
Runtime checks decoder support before downloads and verifies the inherited exact freeze.
Real compressed-file decoding remains a Kaggle check.

Local preparation implements a maximum of 600 metadata rows, 24 audio attempts,
300 seconds, 32MiB transport, 64MiB persistent cache and 2MiB per compressed clip.
Eight 1–10-second clips from eight distinct speakers are assigned 4/2/2 to mini
train/validation/test independently of upstream splits, before any mixtures.
No alternative source or full-corpus fallback exists. Every selected clip is used once;
full clean signals, conversion records, source IDs, original split, speaker and audio
hashes survive in the manifest. Exhaustion stops with partial selection and budget evidence.
The inspection ledger conservatively records all inspected metadata identities and audio
attempts, including failures/rejections; retain these for exclusion from final CV27 test.

Normalization is explicit in the runbook. BPE sees only the four mini-train references;
actual vocabulary and all piece IDs, unknown/round-trip/token length per split are saved.
No generated text or held-out vocabulary selection. The nine vocabulary-bearing tensors
are remapped by exact compatible piece or special role, with fresh unmatched rows;
every other tensor receives a role/shape audit and a recorded transfer decision.

Pinned `compute_fbank_lsmix.py`, `add_source_feats.py`, model and beam-search hashes
are preserved alongside the two unchanged 1A patches. Feature extraction uses the
same Lhotse KaldifeatFbank configuration, then the actual K2SurtDataset with sources.
Gain-adjusted clean features and full 0/0.5s timing are stored explicitly, using NumPy
storage instead of lilcom compression (no feature-algorithm change). The recipe's
HEAT target LOG_EPSILON addition and padding convention are retained and documented.
The sole update calls actual `train.compute_loss`: at batch index 0 the coefficients
are simple RNN-T 1, pruned RNN-T 0.1, CTC 0.2, HEAT auxiliary MSE 0.001.
Fresh FP32 Adam, one train mixture, exactly one optimizer step; no old optimizer state.
Raw component checks reject nonfinite losses before the recipe can filter samples.
Gradient, parameter delta, full state finite checks, strict save/reload, eval/inference
decode of validation/test mixtures and PI-WER counts are required evidence.

This temporary subset validates only the phase-zero pipeline. Main training is planned
from Common Voice 27 at Mozilla Data Collective. Only its train split may train the
model/tokenizer; validation/test remain protected. At migration, compare all exposed mini
speakers, source IDs and original/canonical audio hashes; cross-version re-encoding also
requires robust audio fingerprint review. Exclude collisions from final test. The mini BPE
is not automatically the final tokenizer: coverage review and renewed weight audit are
required for any tokenizer change. No long training or incremental-streaming claim here.

Local validation: 47 offline tests passed across `test_surt_mini.py`,
`test_surt_base_english.py`, `test_kaggle_surt_fixture.py`, `test_kaggle_workflow.py`
and `test_kaggle_environment.py`. All six runbook cells and `env.sh` pass bash
syntax checks; new Python files parse, JSON pins load, and `git diff --check` passes.
Tests exercise normalization, speaker/clip/audio separation, semantic remapping with
equal vocabulary sizes but changed IDs, all nine vocabulary tensors, fresh unmatched
rows, train-only tokenizer input, bounded network/cache/deadline behavior, Mac refusal,
append-only evidence and archives excluding weights/audio/secrets. Tokenizer/tensor
API doubles verify policies only; actual SentencePiece, audio decoder, feature kernels,
K2SurtDataset and RNN-T/CTC/HEAT runtime remain unexecuted locally and require Kaggle.
The installed Lhotse 1.33.0 source was inspected: per-cut feature storage takes a writer
via `storage=`, not the CutSet-level `storage_path` interface. Kaldifeat v1.25.4 primary
source confirms CPU waveform inputs are copied to the configured compute device and
features returned to the input device, matching the Lhotse NumPy extraction path.
