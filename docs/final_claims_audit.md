# Final claims, validity, and contribution audit

## Claim audit

| Candidate claim | Status | Reason |
| --- | --- | --- |
| A streaming-first architecture is implemented | **IMPLEMENTED** | Timestamped chunks, component state, partial/final outputs, and routing state are code-tested |
| Streaming OSD works in general | **NOT VERIFIED** | Only a causal heuristic and limited synthetic evidence exist |
| Adaptive routing improves recognition | **NOT MEASURED** | No comparable final records/checkpoints |
| Multi-talker ASR improves overlap recognition | **NOT MEASURED** | External SURT decoder/checkpoint absent |
| Fine-tuning improves multi-talker ASR | **NOT TECHNICALLY SUPPORTED** locally | No exact trainable external recipe |
| The final system is real-time | **NOT MEASURED** | No final RTF/latency/memory measurement |
| The system generalizes to real meetings | **NOT MEASURED** | No external evaluation |

## Contribution classification

| Contribution | Classification | Status |
| --- | --- | --- |
| Stateful audio/VAD/OSD/routing contracts | ENGINEERING CONTRIBUTION | IMPLEMENTED |
| Synthetic controlled-overlap generation and provenance | ENGINEERING CONTRIBUTION | IMPLEMENTED |
| Multi-talker adapter and PI-WER evaluation contract | ENGINEERING CONTRIBUTION | IMPLEMENTED; external model not reproduced |
| Training feasibility, leakage, collation, checkpoint contract | REPRODUCIBILITY CONTRIBUTION | IMPLEMENTED |
| Frozen per-example evaluation/reporting protocol | REPRODUCIBILITY CONTRIBUTION | IMPLEMENTED |
| Measured architectural improvement | EXPERIMENTAL CONTRIBUTION | NOT MEASURED |
| New ASR/OSD model | RESEARCH CONTRIBUTION | NOT CLAIMED |

## Threats to validity

- **Internal:** unverified external checkpoint/runtime and missing final records prevent comparable-system conclusions.
- **External:** synthetic two-speaker mixtures do not establish robustness on real conversations or more speakers.
- **Construct:** channel permutation metrics do not provide speaker identity, diarization, or attribution quality.
- **Reproducibility:** external model recipes, dependencies, hardware, and data assets must be pinned before full reproduction.

## Responsible reporting

This artifact contains no personally identifying speaker-enrollment workflow. It must not be represented as a surveillance, diarization, identity, or production-ready system. Results must retain oracle/non-deployable labels and report failures rather than replacing them silently.
