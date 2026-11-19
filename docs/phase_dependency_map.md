# Active dependencies — Persian SURT adaptation

Updated 2026-11-19: real English 1A and the technical mini Persian 1B gate are
accepted. Native streaming remains open. Start with [1C-A](KAGGLE_SURT_STREAMING_RUNBOOK_FA.md)
for exact API/source and component/resource evidence, then complete native state
integration and end-to-end parity in 1C-B. Restore saved 1B inputs after a session
reset; rerunning the completed phases is unnecessary.

```mermaid
flowchart TD
    A[Completed synthetic GPU smoke] --> B[Official SURT Base: real English baseline]
    B --> C[Mini Persian: text/tokenizer and vocabulary transfer]
    C --> D[Real Persian update and checkpoint evidence]
    D --> E[Chunk-state proof and measured resource budget]
    E --> F[Licensed Persian data and frozen splits]
    F --> G[Simple mixtures then aligned FastMSS]
    G --> H[Controlled Persian adaptation]
    H --> I[ORC-WER and streaming evaluation]
```

The current native-export wrappers are an optional integration path. Missing
local exports do not block the upstream real-speech baseline. Actual chunk-state
engineering remains necessary before claiming incremental streaming. Generic
VAD/OSD diagnostics are not prerequisites. See [roadmap](PERSIAN_SURT_ROADMAP_FA.md).
