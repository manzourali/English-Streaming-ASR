# Active dependencies — Persian SURT adaptation

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
