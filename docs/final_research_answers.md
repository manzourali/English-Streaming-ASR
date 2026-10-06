# Final research questions and answers

The table distinguishes what is implemented from what was measured. No question below has a quantitative final answer until verified checkpoints, final records, and the frozen test set are executed.

| Research question | Experiment / metric | Observed result | Answer | Evidence | Limitation |
| --- | --- | --- | --- | --- | --- |
| Does streaming WhisperRT handle clean English speech? | Clean baseline; WER/RTF | NOT MEASURED | No final conclusion | Phase 1 adapter and runner | Model/checkpoint unavailable |
| How much does overlap degrade single-stream ASR? | Overlap baseline; valid overlap metric | NOT MEASURED | No final conclusion | Phase 4 protocol | No executed model/data run |
| Does streaming OSD identify overlap? | Precision/recall/F1/delay | Limited Phase 5 synthetic heuristic demo only; no final frozen result | Not established for final system | Phase 5 implementation | Heuristic, synthetic condition |
| Does adaptive routing help? | Predicted vs oracle vs control | NOT MEASURED | No final conclusion | Phase 6 controller/protocol | Missing final branches/checkpoints |
| Does multi-talker ASR help? | Permutation-invariant WER | NOT MEASURED | No final conclusion | Phase 7 adapter and metric | External decoder/checkpoint unavailable |
| Does fine-tuning help? | Base vs adapted multi-talker | NOT TECHNICALLY SUPPORTED locally | No claim | Phase 8 feasibility study | No pinned trainable recipe |
| Is the system real-time? | RTF and latency | NOT MEASURED | No claim | Phase 9 benchmark schema | No hardware execution |
| Does it generalize / remain robust? | External data and perturbation tests | NOT MEASURED | No claim | Phase 9 protocol | No independent evaluation |

The valid final conclusion is an engineering/reproducibility conclusion: the project provides a traceable implementation and a guarded evaluation protocol, but not verified end-to-end empirical evidence for the proposed system.
