# Persian streaming SURT: feasibility and decision gates

## Phase 0: audit

This repository is an inference and experiment scaffold, not an Icefall SURT
2.0 recipe. It does not contain the upstream Icefall checkout. Its Kaggle
notebooks cannot by themselves train, decode, or establish WER for SURT.

| Component | Present | Gap |
| --- | --- | --- |
| Data | Two-source mixture generator, manifests, validation and statistics | No Persian corpus preparation, silence-aligned transcripts, RIR or MUSAN augmentation |
| Features | 80-bin streaming log-Mel frontend | Equivalence to the selected Icefall recipe's feature extraction has not been demonstrated |
| Tokenizer | Inference can load an external SentencePiece model | No supplied English model or end-to-end Persian model integration |
| Model | `TorchSURTBackend` composes externally supplied mask, encoder, prediction network and joiner; optional external branch-tying layer | No trained DP-LSTM, Zipformer, stateless predictor, joiner, branch-tying implementation or compatible export supplied |
| Training | Dataset/collator and external-backend interface | No model backend, pruned RNN-T, CTC or mask objective, HEAT implementation, optimizer schedule or checkpoint averaging |
| Decoding | Greedy streaming wrapper for a compatible export | No supplied compatible export, native Icefall decoding integration or validated streaming-equivalence test |
| Metrics | Short-mixture permutation-invariant WER | No ORC-WER/cpWER integration for continuous conversations |

`scripts/train_asr.py --smoke` only validates manifests; it performs no
optimization or decoding. The local checkout has no speech data or usable
checkpoint, and the required Icefall/k2 stack is not installed. Therefore a
real one-step training/decode smoke test, Kaggle GPU measurements, and WER are
**not available**. Fabricated labels on synthetic tones are infrastructure
fixtures, not speech recognition results. Kaggle execution has not been tested.

## Available data increments

The existing `scripts/generate_overlap.py` generates timed two-source mixtures
and split statistics. Its `--demo` input is tones, not speech. It does not
implement VAD, RIR or noise. The adapter accepts this repository's overlap
JSONL schema, not arbitrary FastMSS metadata; FastMSS must first be mapped to
that schema. It emits Lhotse-style recordings, supervisions and cuts, each
mixture with two timed speaker-labeled transcripts:

```bash
PYTHONPATH=src python3 -m streaming_asr.datasets.icefall_adapter \
  outputs/synthetic_overlap/manifests/test.jsonl data/persian/manifests --split test
```

Paths in the generated manifests are absolute; regenerate or rewrite them when
moving the data. The adapter does not extract 80-bin features or preserve
aligned clean-source audio as mask targets. Loading in Lhotse and acceptance by
a particular Icefall datamodule must be tested with the pinned environment;
two supervisions on one mono cut do not themselves prove SURT compatibility.

Optional Persian dependencies support normalization, BPE and OOV reporting:

```bash
python3 -m pip install -e '.[persian]'
python3 scripts/persian_tokenizer.py normalize train.raw.txt train.txt
python3 scripts/persian_tokenizer.py normalize valid.raw.txt valid.txt
python3 scripts/persian_tokenizer.py train train.txt data/persian/bpe --vocab-size 500
python3 scripts/persian_tokenizer.py coverage data/persian/bpe.model valid.txt
```

Create the model parent directory first. Expand dates and decimal numbers with
the intended spoken form before normalization; ambiguous forms are rejected.
The generated model is not wired into an Icefall recipe; verify special token
IDs and generate any additional vocabulary files expected by the pinned recipe.

## Completion checklist

- Pin an upstream Icefall commit and a matching PyTorch, CUDA, k2 and Lhotse
  environment; test imports and record the Kaggle image/GPU/session limits.
- Inspect that revision's actual SURT datamodule, CutSet/mixture and clean-source
  requirements before claiming any generated manifests are recipe-compatible.
- Produce speaker-disjoint Persian splits with timed, nonempty transcripts;
  retain clean source signals and timing for mask supervision.
- Train Persian BPE on training text only, verify unknown-token coverage, and
  ensure blank ID and output dimensions match the exact Icefall recipe.
- Reuse upstream data preparation, model, loss, training and decoding code;
  run one forward/backward/update and actual decoding on a tiny speech subset.
- Verify checkpoint compatibility before warm-starting: a changed vocabulary
  requires resetting incompatible decoder/joiner/CTC parameters and optimizer
  state. A single-speaker checkpoint is not automatically SURT-compatible.
- Confirm HEAT ordering and mask targets against the pinned implementation;
  test weighted pruned RNN-T, CTC and mask losses, optimizer/scheduler and
  averaging semantics rather than assuming their CLI flags.
- Check chunked against reference decoding, bounded state, right context and
  end-to-end latency; report WER with a declared permutation-aware protocol.

## Staged experiment design

1. Normalize Persian consistently across train and evaluation. Retain a raw
   text copy and version the normalization and BPE model. Split by speaker
   before augmentation to prevent leakage.
2. Warm up a small streaming single-talker transducer, or transfer only
   shape-compatible weights from a known checkpoint. Validate on held-out
   clean Persian before multi-talker training.
3. Use the pinned upstream SURT implementation for two-branch training. The
   requested objective is pruned RNN-T + 0.2 CTC + 0.2 mask loss; verify its
   scaling and supervision semantics against that implementation. Start with
   a tiny model, short cuts, low max-duration and one GPU. `ScaledAdam` and a
   0.004 peak/base learning rate with 5,000 warmup steps are experimental
   targets, not validated settings. Average the last nine compatible epoch
   checkpoints only when at least nine exist; otherwise explicitly report the
   smaller window. Save optimizer/scheduler and random states for resume.
4. Evaluate held-out 0%, 20%, and 40% overlap, separately and combined. Define
   overlap as intersection divided by shorter source duration if using this
   repository's generator. Report ORC-WER or cpWER with the exact segmentation,
   normalization, speaker and assignment protocol; do not equate them with
   short-utterance PI-WER. A real-conversation set is optional and must be
   consented, annotated independently, and disjoint from training speakers.

For every run record revision, dependency versions, data hashes, split rules,
seed, tokenizer, chunk/right-context configuration, hardware, GPU peak memory,
epoch duration, total GPU time, real-time factor and wall-clock latency. Free
Kaggle session duration and available GPU are not guaranteed; a paper-scale
training run cannot be assumed feasible. A tiny proof establishes plumbing,
not accuracy or thesis-scale resource feasibility.

## Results template (not measured)

| Set | Overlap | Metric | WER | GPU / peak VRAM | Seconds / epoch |
| --- | --- | --- | --- | --- | --- |
| Synthetic | 0% | ORC-WER or cpWER | Not run | Not measured | Not measured |
| Synthetic | 20% | ORC-WER or cpWER | Not run | Not measured | Not measured |
| Synthetic | 40% | ORC-WER or cpWER | Not run | Not measured | Not measured |
| Real, optional | measured | ORC-WER or cpWER | Not run | Not measured | Not measured |

## Next steps and decision gate

Obtain licensed Persian speech with speaker IDs and time-aligned text; acquire
noise and RIR data with appropriate licenses. Pin and run the upstream recipe
on the actual Kaggle GPU. The decision gate is a reproducible real-speech
miniature run that converts data, computes valid mask targets, trains one
step, saves/reloads a checkpoint, decodes in streaming mode and scores the
outputs. Only after measuring memory, throughput and quality on that run can
the full thesis schedule and free-tier feasibility be estimated responsibly.
