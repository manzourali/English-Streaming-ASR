# SURT synthetic infrastructure smoke — 2026-11-12

Recorded 2026-11-13. Evidence: user-supplied decode console log and the user's
confirmation of the exact checkpoint-inspection success message. The complete
saved Kaggle artifact bundle has not been inspected locally.

| Item | Observed / reported evidence |
| --- | --- |
| Recipe | Icefall LibriCSS SURT DPRNN/Zipformer Base, 26,675,639 parameters in decode log |
| Icefall | `f8c8578c-dirty` in log; runbook specifies `f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1` plus a local evaluation-sampler patch |
| Project commit | Runbook specifies `4c426d12dd18548656c4830210e80dee099a41f8`; actual `commits.txt` not supplied for independent comparison |
| Runtime | Python 3.12, Torch 2.3.1+cu121, CUDA 12.1, k2 1.24.4, Lhotse 1.33.0; decode device `cuda:0` |
| Input | Deterministic tones and invented English labels; same synthetic cuts reused across dev/test |
| Update | User confirms `batch_idx_train=1; optimizer step=1; saved tensors finite` after checkpoint reload |
| Decode | `checkpoint-1.pt` loaded; 12 distinct scored dev/test groups; final `Done!`; no traceback in supplied log |
| Scores | 1250–2250% WER; plumbing outputs, not speech recognition accuracy |
| Sampler | User's 10-buckets/6-cuts failure resolved by SimpleCutSampler in validation/test; preserve actual diff |
| Missing archival metadata | Archive/checkpoint/BPE/manifest SHA-256 values and Kaggle version ID not independently supplied; add actual values later |
| Not measured | Persian loss/transfer, genuine chunk-state inference, per-step training time, process peak VRAM, thesis-scale budget |

**Decision:** infrastructure smoke passed on the provided evidence. Persian
language adaptation and streaming feasibility remain open. Next gate: official
pretrained SURT Base on a tiny real English speech set; then mini Persian
vocabulary/transfer/update and stateful streaming/resource checks.

No real-speech WER, Persian recognition quality, finite intermediate gradients,
full-training reproducibility, or full dataset evaluation is inferred from
this smoke run. Whole-batch decoding with chunk-constrained settings is not an
incremental streaming demonstration. See [roadmap](../PERSIAN_SURT_ROADMAP_FA.md)
and [artifact retention](../KAGGLE_SURT_ARTIFACTS_FA.md).
