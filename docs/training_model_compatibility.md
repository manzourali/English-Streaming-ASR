# Phase 8 — SURT 2.0 training compatibility

## Evidence reviewed

The selected SURT 2.0 release describes a continuous, streaming, speaker-agnostic multi-talker recognizer; it publishes Icefall recipes for LibriCSS, AMI, and ICSI, plus pre-trained models. The release page directs training through `egs/libricss/SURT` or `egs/ami/SURT`, not through Hugging Face `Trainer`. The paper describes a dual-path mask estimator, streaming Zipformer encoder, stateless transducer decoder, mask-estimation auxiliary loss, and encoder CTC auxiliary loss. See the [SURT 2.0 release](https://sites.google.com/view/surt2/home), [paper](https://arxiv.org/abs/2306.10559), and [Icefall streaming documentation](https://k2-fsa.github.io/icefall/recipes/Streaming-ASR/introduction.html).

| Compatibility item | Finding | Project status |
| --- | --- | --- |
| Architecture | End-to-end mask estimator + streaming Zipformer + stateless transducer | Compatible in principle; exact recipe not installed |
| Training objective | HEAT-style channel assignment for multi-talker transducer training; SURT 2.0 adds masking/encoder-CTC auxiliaries | Target manifest preparation implemented; exact recipe loss is external |
| Input / outputs | Mixture features/audio in; two unordered transcript channels out | Implemented as `heat_two_channel_transcripts` metadata |
| Tokenizer | Icefall recipe BPE assets are model/checkpoint-specific | **NOT AVAILABLE** locally; never inferred or recreated |
| Native streaming state | Recipe architecture is streaming with limited right context | Phase 7 adapter remains **LOW-LATENCY WINDOWED** until exact runtime is verified |
| Official fine-tuning | Recipes include training/adaptation paths | Requires a pinned Icefall checkout, recipe, checkpoint, BPE assets, and dependency versions |
| Hugging Face Trainer / Transformers | No verified official SURT 2.0 Trainer path | **NOT TECHNICALLY SUPPORTED** for this project |
| PEFT / LoRA / QLoRA | No verified target-module mapping or official procedure for this exact model/runtime | **NOT TECHNICALLY SUPPORTED** in this checkout |
| Mixed precision / gradient checkpointing | Recipe-specific capability; not exposed by Phase 7 decoder factory | **RESEARCH DECISION PENDING** after exact recipe installation |
| Checkpoint resume | Icefall recipes document checkpoint/optimizer state management | Project lifecycle supports an external backend checkpoint contract |

## Training strategy decision

**Selected strategy: no project-local fine-tuning (training-feasibility study).**

The repository has a generic external inference `decoder_factory`, not a trainable PyTorch/Icefall model object. It cannot expose parameters, tokenizer IDs, trainable target modules, optimizer state, or the exact multi-objective loss. Pretending that generic PyTorch/Transformers/PEFT code could adapt it would create an unreproducible and scientifically invalid procedure.

The Phase 8 implementation therefore provides validated manifests, target construction, batching, leakage checks, run/checkpoint lifecycle, and an `ExternalSurtTrainingBackend` protocol. A real run is enabled only by a version-pinned `training.backend_factory` that wraps one verified Icefall SURT recipe. Until then:

```text
TRAINING NOT TECHNICALLY SUPPORTED
PEFT / LoRA / QLoRA NOT TECHNICALLY SUPPORTED
```

This is a technical compatibility conclusion, not a claim that SURT 2.0 itself cannot be trained.
