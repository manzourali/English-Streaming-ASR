# Review plan: pinned Icefall SURT smoke test

Status: **review only; no fixture or training has been run.** No Persian changes,
full corpus preparation, model changes, or Kaggle execution are authorized by
this plan. The existing uncommitted project changes are separate from this
smoke-test work.

## Pins and environments

- Icefall: `f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1`.
- Project HEAD at preparation: `e67d785b276b590df94349945dfba31e7fed831a`.
  This is **not** a runnable pin for the new files: they are uncommitted. The
  project revision to clone on Kaggle must be the new commit containing the
  reviewed artifacts, chosen after review and push. Record its full hash.
- macOS: review code and, if authorized, run lightweight tests only. Do not
  install the full training stack or download models, datasets or checkpoints.
- Kaggle: after separate approval, use a GPU runtime and choose mutually
  compatible CUDA/PyTorch/k2/kaldifeat/Lhotse/SentencePiece builds, recording
  exact package versions, GPU, VRAM, CUDA, and both repository hashes. The Mac
  versions previously imported are not a tested Kaggle compatibility matrix.

## Actual pinned recipe contracts

The observations below come from `prepare.sh`, `dprnn_zipformer/train.py`,
`dprnn_zipformer/asr_datamodule.py`, `dprnn_zipformer/decode.py`,
`local/add_source_feats.py`, and the inspected Lhotse `K2SurtDataset`.
Exact behavior must be rechecked against the Lhotse version selected on Kaggle.

- The standard data module loads `cuts_train_comb_full_sources.jsonl.gz` and
  `cuts_train_comb_ov40_sources.jsonl.gz`; training also loads
  `cuts_dev_libricss-sdm.jsonl.gz`. The standard decoder loads both
  `cuts_dev_libricss-ihm-mix.jsonl.gz` and
  `cuts_test_libricss-ihm-mix.jsonl.gz`.
- Each cut must have loadable precomputed mono 16 kHz mixture audio and
  80-dimensional frame features; each supervision needs a unique ID,
  `recording_id`, channel 0, start, positive duration, nonempty English text,
  and speaker. Supervision intervals are relative to the cut and bounded by
  it. The train loader drops cuts below 1 or above 30 seconds.
- Each training cut must expose `source_feats`, a loadable two-dimensional
  concatenation of clean-source feature matrices ordered by supervision
  `(start, speaker)`, and `source_feat_offsets`, one offset per supervision.
  For two sources these are `[0, len(source_1)]`; splitting at the second offset
  yields the individual `[source_frames, 80]` matrices. For each supervision,
  its start/end times become mixture-frame boundaries, and its source matrix
  length must match that slice within Lhotse's two-frame adjustment tolerance.
  The HEAT mask loss adds each source matrix to its assigned branch's target
  slice. Training needs both nonempty branches for this two-speaker fixture.
- `K2SurtDataset` batches features as `inputs` `[B,T,80]`, `input_lens` `[B]`,
  per-cut two-branch `text`, `source_feats`, and `source_boundaries`. It assigns
  ordered supervisions to available branches using their start/end times.
  Decode needs `inputs`, `input_lens`, and `cuts` containing reference
  supervisions; it groups cut IDs by `0L`, `0S`, `OV10`, `OV20`, `OV30`, and
  `OV40` substrings.
- Stage 4's LibriSpeech feature-extraction command is commented out. Stage 5
  hard-codes 100,000 meetings; stages 7–8 also expect RIR/noise and auxiliary
  source features. Stage 1 can exit zero without alignments or manifests. None
  is a validated three-sample route; do not execute full `prepare.sh` for this
  smoke test.

## Review artifact

`scripts/kaggle_surt_fixture.py` proposes six deterministic tone mixtures:
one per decode label. Each contains two one-second source signals, 80-bin
Kaldifeat features with dither disabled, two English placeholder transcripts,
and concatenated source features. It writes one `OV40` training cut into each
training manifest and all six cuts into each dev/decode manifest. These splits
intentionally reuse the same synthetic data and are **not** evaluation splits.
Six cuts, rather than three, ensure each official decoder label is populated.

The validator checks the Icefall commit, manifests, labels, timing, feature
dimensions, source offsets and slices, source/mixture audio-feature consistency,
and actual `K2SurtDataset` batch construction. It probes the pinned decoder
with an empty iterable and records the observed result, then requires every
decode label to have a cut. Static inspection suggests `decode_dataset` returns
an uninitialized `masks_dict` for an empty iterable; the probe has not been run.
Passing validation would verify these data contracts only, not model execution.

The proposed future Kaggle-only invocations, **not executed**, are:

```bash
python scripts/kaggle_surt_fixture.py build /kaggle/working/surt-fixture
python scripts/kaggle_surt_fixture.py validate /kaggle/working/surt-fixture \
  --recipe-dir /kaggle/working/icefall/egs/libricss/SURT
```

The builder refuses a nonempty output directory. Manifests contain absolute
paths and must remain in the same Kaggle session/location. `NumpyFilesWriter`
is used instead of the recipe's Lilcom storage; verify the selected Lhotse
version accepts it. Feature extraction uses the recipe's Kaldifeat defaults on
CPU with dither disabled; GPU training remains the subject of the future test.
Floating-point output and generated storage keys should not be assumed bitwise
identical across environments; save hashes of actual inputs and outputs.

## Future Kaggle gate

1. Record exact project and Icefall commits, hardware and dependencies; download
   the existing English BPE only on Kaggle and record its hash.
2. Build and strictly validate the fixture. Inspect the actual empty-group probe
   and all six group counts before any training.
3. With one actual batch, disable optional MUSAN and SpecAugment and configure
   checkpoint/averaging intervals so batch 1 can be saved. Confirm a finite
   forward loss, finite gradients, an optimizer update, and a loadable checkpoint
   through logs and checkpoint inspection; do not rely only on exit status.
4. Decode with the same model configuration and a single non-averaged checkpoint,
   with bounded cut count. Verify outputs and scoring; a WER against invented
   labels for tones is only a plumbing value, not recognition quality.
5. Report whether decoding is whole-batch with chunk-constrained attention or
   genuinely incremental with persisted state. The pinned `decode.py` calls the
   mask and encoder on whole input batches; it does not by itself prove an
   incremental stateful streaming API or real-time performance.

Capture commands, logs, package versions, GPU and peak VRAM, input/BPE/model
hashes, optimizer-step evidence, checkpoint reload and decode artifacts. Do not
claim success for any step whose evidence is missing.
