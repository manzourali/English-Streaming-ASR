# Future work

## Model

- Verify a pinned native-streaming SURT 2.0 runtime and checkpoint; compare it against the existing low-latency wrapper.
- Evaluate stronger causal multi-talker decoders without changing the reported Phase 10 artifact retroactively.

## Data

- Run the frozen synthetic Phase 3 test protocol with source-disjoint speakers.
- Add separately reported real-world English overlap evaluation after license, preprocessing, and annotation checks.
- Investigate more speakers and acoustic conditions only as a new experiment protocol.

## OSD and routing

- Replace the heuristic OSD only in a separately versioned experiment.
- Evaluate uncertainty-aware or cost-aware routing against the Phase 6 baseline.

## Training

- Install a version-pinned Icefall/SURT recipe and verify its tokenizer, loss, checkpoint resume, and native streaming behavior.
- Explore PEFT only after exact model modules and a stable full-training baseline are verified.

## Deployment

- Measure CPU and GPU resource use, microphone ingestion, long-session state behavior, and failure recovery under controlled deployment conditions.
