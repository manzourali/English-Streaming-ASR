# Current execution matrix — Persian SURT

| Entry point / gate | Output | Evidence / limitation |
| --- | --- | --- |
| `docs/KAGGLE_SURT_RUNBOOK.md` | Synthetic one-step checkpoint and 12-cut decode | Passed on user-supplied Kaggle evidence; no real speech or Persian |
| `docs/prompts/PERSIAN_SURT_NEXT_STEP.md` | Request for a pinned real-English pretrained baseline runbook | Next work; not yet executed |
| Mini Persian transfer/update and stateful streaming | Persian feasibility/resource evidence | Planned; not established |
| `scripts/persian_tokenizer.py` | Normalization, train-only BPE and coverage | Utilities exist; not connected to a validated Persian model recipe |
| `generate_overlap.py`, manifests and `icefall_adapter` | Small two-source data / Lhotse-style metadata | Does not alone implement FastMSS or aligned mask-target recipe features |
| `check_surt_assets.py`, `run_multitalker.py`, `run_streaming.py` | Native export checks / transcripts / records | Real compatible export still required; upstream recipe is a separate execution path |
| `train_asr.py --validate-only` | Manifest/target checks | No optimizer update |
| `train_asr.py` | External training-backend wrapper | Real training integration missing; use actual pinned Icefall for model work |
| `evaluate.py`, `benchmark.py` | Existing record aggregation | Short-mixture PI-WER available; ORC-WER continuous protocol integration pending |

Current decisions and prompts: [roadmap](PERSIAN_SURT_ROADMAP_FA.md).
