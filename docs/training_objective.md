# Phase 8 — Training objective and target representation

## Input

The input is the existing Phase 3 mono synthetic mixture at its recorded sample rate (normally 16 kHz). The training pipeline does not create a duplicate audio corpus; it references the existing mixture path in a model-specific JSONL manifest.

## Target

The selected SURT-family target is represented as:

```text
heat_two_channel_transcripts
channel 0: transcript of the source with earlier start time
channel 1: transcript of the other source
```

This is a deterministic HEAT-style source-onset assignment appropriate to the project’s current two-source mixtures. It creates *model output channels*, not real speaker labels or identities. The exact external Icefall recipe owns BPE/tokenization and may validate or adapt the assignment to its documented format.

## Loss

The selected SURT 2.0 recipe—not this project—must compute the exact transducer/HEAT loss and any documented mask-estimation or encoder-CTC auxiliary losses. The project refuses to substitute a generic cross-entropy loss because that would not train the selected architecture correctly.

## Output and evaluation mapping

The model produces two unordered transcript channels. Evaluation uses permutation-invariant WER with empty-channel padding; no speaker-attributed WER or diarization claim is made. The implementation stores enough provenance to rebuild a target example:

- mixture/sample ID, path, sample rate, duration, and split;
- source/speaker IDs, source transcripts, source timing, overlap interval/ratio;
- source-order channel targets and target format;
- mixing seed, regime, relative gain, and overlap duration.

## Conditions

`base` creates no training batch. `clean` selects zero-overlap examples, `overlap` selects positive-overlap examples, and `mixed` retains both. These are intentionally small first conditions; test manifests are loaded only for leakage protection and are not passed to the training backend.
