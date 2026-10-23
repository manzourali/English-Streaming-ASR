# Active artifact dependencies

```mermaid
flowchart TD
    A[Main notebook: real LibriSpeech] --> B[Portable mixtures and split manifests]
    C[Notebook 12: official weights and vocabulary] --> D[Missing engineering: native model connection and feature parity]
    D --> E[Validated native export]
    B --> F[Notebook 12: base inference]
    E --> F
    F --> G[EvaluationRecord JSONL]
    B --> H[Notebook 13: optional training]
    C --> H
    I[Missing engineering: actual training backend] --> H
    H --> J[Trained checkpoint]
    J --> K[Missing engineering: native export]
    K --> L[Notebook 12: fine-tuned inference]
    B --> L
    L --> M[EvaluationRecord JSONL]
    G --> N[Notebook 14: identical test sets and reports]
    M --> N
```

WhisperRT, VAD and OSD outputs are not dependencies of SURT inference.
Data preparation and result-schema handoffs are executable. The native model
connection, feature/checkpoint parity, training backend and exporter remain
explicit implementation gaps. For a base-only study, omit training and report
only the actual base model results. See [the Persian guide](KAGGLE_WORKFLOW_FA.md).
