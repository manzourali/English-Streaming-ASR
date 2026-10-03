# Phase 7 candidate review

This review distinguishes architecture-level streaming claims from what this repository can reproduce. “Checkpoint available” means a public release describes model artifacts; it does **not** mean this checkout has downloaded or verified one.

| Candidate | Evidence / implementation | Overlap capability | Streaming classification | Public inference readiness | Decision |
| --- | --- | --- | --- | --- | --- |
| SURT (Lu et al., 2021) | [paper](https://arxiv.org/abs/2011.13148) | Two-channel end-to-end unmixing plus RNN-T recognition | **TRUE STREAMING** architecture; the paper reports a 150 ms algorithmic-latency condition | Research implementation details, not a stable project-local runtime | Foundation for selected family |
| SURT 2.0 (Raj, Povey, Khudanpur, 2023) | [paper](https://arxiv.org/abs/2306.10559), [release page](https://sites.google.com/view/surt2/home), [icefall](https://github.com/k2-fsa/icefall) | Continuous multi-talker ASR with multi-channel outputs | **TRUE STREAMING** architecture: streaming Zipformer + stateless transducer; project adapter remains **LOW-LATENCY WINDOWED** until the external native-state API is verified | Public code/model release is described, but the exact runtime API/checkpoint was not installed in this checkout | **Selected** |
| t-SOT | [paper](https://arxiv.org/abs/2202.00842), [Microsoft publication](https://www.microsoft.com/en-us/research/publication/streaming-multi-talker-asr-with-token-level-serialized-output-training/) | Serialized multi-talker tokens | **TRUE STREAMING** research method | No verified, directly usable public checkpoint/runtime was established here | Not selected: reproduction risk |
| Whisper-Sidecar | [repository](https://github.com/LingweiMeng/Whisper-Sidecar) | Joint multi-talker / target-talker research code | Not established as a causal streaming implementation for this project | Code is available, but target-ASR mode uses enrolled prompt audio and its checkpoint/runtime suitability was not verified | Not selected: enrollment and streaming fit are poor |

## Selection

**Selected model:** SURT 2.0 (Streaming Unmixing and Recognition Transducer 2.0) through the public release/icefall ecosystem.

- **Repository/runtime:** external SURT 2.0 release material and icefall-compatible recipes; no dependency is vendored.
- **Checkpoint:** external release artifact, **NOT DOWNLOADED OR VERIFIED** in this checkout.
- **Input:** 16 kHz mono mixture audio.
- **Output:** unordered recognition streams/channels, not real-world speaker identities.
- **Overlap capability:** designed specifically for continuous multi-talker ASR.
- **Streaming classification:** the published architecture is true streaming; the project adapter is labelled **LOW-LATENCY WINDOWED INFERENCE** until a configured external decoder exposes and verifies native encoder/decoder state.
- **GPU / license / installation:** **RESEARCH DECISION PENDING** for a concrete checkpoint. They must be captured from the exact selected external release before a measured run. Icefall itself is publicly released under Apache-2.0; that does not automatically determine the license of every model artifact.

SURT 2.0 was chosen because it is the only reviewed option that directly targets continuous streaming multi-talker recognition while retaining public release material. The limitation is intentional and explicit: this repository will not guess an external decoder API, download weights automatically, or substitute ordinary WhisperRT when SURT is unavailable.
