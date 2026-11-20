# Current execution matrix — Persian SURT

| Entry point / gate | Output | Evidence / limitation |
| --- | --- | --- |
| `docs/KAGGLE_SURT_RUNBOOK.md` | Synthetic one-step checkpoint and 12-cut decode | Passed on user-supplied Kaggle evidence; no real speech or Persian |
| `docs/KAGGLE_SURT_BASE_ENGLISH_RUNBOOK_FA.md` | Real-English official Base baseline | 1A accepted, including strict loading and auditory review |
| `docs/KAGGLE_SURT_MINI_PERSIAN_RUNBOOK_FA.md` | Real Persian transfer, feature/source/HEAT, one update and reload | 1B technical gate accepted for mini-20261119-d; mini PI-WER 100%, no useful-quality claim |
| `docs/KAGGLE_SURT_STREAMING_RUNBOOK_FA.md` | Restart, exact API/source packet, incremental feature/LSTM component probes, warm resource measurements | 1C-A executed in stream-20261119-b; component/resource evidence reviewed; historical runbook |
| `docs/KAGGLE_SURT_STREAMING_1CB_RUNBOOK_FA.md` | Full-model bounded stateful inference and same-context parity | Technical gate accepted for streamb-20261120-a; useful Persian quality and live delivery latency remain unproven |
| `docs/prompts/PERSIAN_SURT_FOLLOWUPS.md`, stage 2 | Bounded pilot data, frozen splits and text/tokenizer policy | Prompt prepared; stage 2 awaits user instruction |
| `scripts/persian_tokenizer.py` | Normalization, train-only BPE and coverage | Utilities exist; not connected to a validated Persian model recipe |
| `generate_overlap.py`, manifests and `icefall_adapter` | Small two-source data / Lhotse-style metadata | Does not alone implement FastMSS or aligned mask-target recipe features |
| `check_surt_assets.py`, `run_multitalker.py`, `run_streaming.py` | Native export checks / transcripts / records | Real compatible export still required; upstream recipe is a separate execution path |
| `train_asr.py --validate-only` | Manifest/target checks | No optimizer update |
| `train_asr.py` | External training-backend wrapper | Real training integration missing; use actual pinned Icefall for model work |
| `evaluate.py`, `benchmark.py` | Existing record aggregation | Short-mixture PI-WER available; ORC-WER continuous protocol integration pending |

Current decisions and prompts: [roadmap](PERSIAN_SURT_ROADMAP_FA.md).
