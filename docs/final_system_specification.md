# Current system specification

The target is continuous English multi-talker SURT 2.0. The previous routed
WhisperRT/SURT system is retired. Source modules, fixed-frame scheduling,
portable mixture data, endpoint logic and evaluation records are implemented.
Actual trained-model GPU inference, fine-tuning and final scientific results
remain unverified.

- Input: contiguous 16 kHz mono PCM.
- Features: 80-bin log-Mel; compatibility with the exact checkpoint must be validated.
- Model: trained DPRNN masks, streaming Zipformer, optional channel tying and
  stateless RNN-T. Recipe-specific native modules/export are required.
- Model advance: 32 frames, 320 ms. Separate acquisition, delivery, endpoint
  and compute delays are described in the stage2 report.
- Primary comparison: native SURT base versus optional adapted SURT on an
  identical protected test split. WhisperRT is an independent optional baseline.
- Current metric: PI-WER on short two-source mixtures. LibriCSS ORC-WER and
  long-recording GPU latency/memory validation remain separate integration tasks.
- Training: external backend not implemented; validate-only produces no weights.
- Reproducibility: record source/model revisions, feature profile, dataset
  hashes, tokenizer, configs and environment alongside real predictions.

Start with the [Persian Kaggle guide](KAGGLE_WORKFLOW_FA.md),
[execution matrix](execution_matrix.md) and [architecture report](stages/stage2_report.md).
