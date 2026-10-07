# Phase dependency map

This map describes code/data dependencies, not commit history. Current `master` contains all phases; running an earlier phase does not require checking out an earlier commit.

```text
Phase 0 environment
├── Phase 1 WhisperRT clean baseline ──────────┐
├── Phase 2 VAD ───────────────────────────────┤
└── Phase 3 synthetic overlap data ─→ Phase 4 overlap baseline ─┐
                                    └→ Phase 5 OSD ───────────────┼→ Phase 6 adaptive routing
                                                                   │       │
                                                          Phase 7 SURT branch ◀┘
                                                                   │
                                                    Phase 8 external training (optional)
                                                                   │
                                                    Phase 9 evaluation / Phase 10 reproducibility
```

| Phase | Inputs → phase → outputs → consumers |
| --- | --- |
| 0 | Core Python/configs → infrastructure smoke path → importable package and small run record → all phases. |
| 1 | LibriSpeech + WhisperRT runtime/checkpoint → clean incremental ASR → predictions, WER/RTF/latency → Phase 4/6 comparison and Phase 9. |
| 2 | Audio/WAV or synthetic demo + WebRTC VAD → stateful VAD → activity decisions/metrics → Phase 6 VAD gate. |
| 3 | LibriSpeech or demo fixtures → deterministic two-source mixtures → WAVs, JSONL manifests, timing/transcript metadata → Phases 4–8. |
| 4 | Phase 3 manifests + WhisperRT → single-stream overlap baseline → records/metrics with limited single-stream reference policy → Phases 6, 7, 9. |
| 5 | Phase 3 manifests/timing labels → causal heuristic OSD → per-frame overlap decisions and OSD metrics → predicted Phase 6 routing. |
| 6 | Phase 3 manifest + WhisperRT + optional VAD/OSD → routing controls (`always_normal`, `oracle`, `predicted`) → routing traces/metrics → Phase 7 integration and Phase 9. |
| 7 | Phase 3 manifest + external SURT decoder/checkpoint → unordered multi-talker streams/PI-WER records → Phase 9; can be used as Phase 6 overlap branch. |
| 8 | Phase 3 train/validation/test manifests + external trainable Icefall/SURT backend → external checkpoints/training state → optional fine-tuned Phase 7 system. |
| 9 | Comparable saved per-example records and frozen configs → aggregate metrics, strata, bootstrap summaries → Phase 10 artifact. |
| 10 | Code, configs, records, checkpoints, provenance → reproducibility/audit package → publication/thesis reporting. |

## What is missing?

| Component | Status | Depends on | Why incomplete | What to do |
| --- | --- | --- | --- | --- |
| Phase 0 smoke test | Ready | Core dependencies | None. | Run it first. |
| WhisperRT baseline | Implemented, unmeasured | External `whisper_rt`, PyTorch, Hub checkpoint, LibriSpeech | Runtime/checkpoint were not present in the audit. | Install/authorize and run one sample. |
| VAD | Ready for integration | `webrtcvad-wheels` | No labeled real-data score. | Run demo; obtain labels for accuracy claims. |
| Synthetic overlap data | Ready | Demo fixtures or LibriSpeech | Final real corpus not persisted. | Generate a controlled small version and publish it. |
| OSD | Partial baseline | Phase 3 manifest | Heuristic only; no learned OSD validation. | Use only as stated baseline, then validate/change experimentally. |
| Adaptive routing | Implemented, unmeasured | WhisperRT + Phase 3 + OSD | Default overlap branch is shared normal ASR. | Run controls after Phase 1/3 assets exist. |
| Multi-talker ASR | External dependency | SURT runtime/decoder/checkpoint | No bundled verified decoder. | Build a pinned `decoder_factory`. |
| Fine-tuning | Not supported locally | External Icefall backend/assets | Generic adapter cannot train SURT. | Provide pinned `backend_factory`; otherwise do not claim training. |
| Final evaluation | Protocol ready, result absent | Comparable prediction records | No final records/checkpoints. | Run systems, freeze config, aggregate. |
