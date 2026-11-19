# Persian Streaming ASR with Overlapped Speech — SURT 2.0

The research goal is Persian language adaptation of SURT 2.0 for continuous
recognition of two overlapping voices. Two output channels do not guarantee
persistent speaker identities. The existing repository/package names are kept
for installation compatibility.

Start with the [current Persian roadmap](docs/PERSIAN_SURT_ROADMAP_FA.md).
The synthetic infrastructure smoke and real English stage 1A passed. The real
Persian mini run `mini-20261119-d` passed the stage 1B technical gate, including
one update, strict reload and the user's auditory confirmation. Its mini
validation/test PI-WER is 100%; useful Persian recognition is not established.
**Native incremental streaming remains an open stage 1C gate.**

- [Current runbook: stage 1C components/resources and session restoration](docs/KAGGLE_SURT_STREAMING_RUNBOOK_FA.md)
- [Reviewed stage 1B evidence and limitations](docs/experiments/surt_mini_1b_review_2026-11-19.md)
- [Later gated prompts: mini Persian → data → adaptation → evaluation](docs/prompts/PERSIAN_SURT_FOLLOWUPS.md)
- [Download all logs as one archive; what belongs in Git](docs/KAGGLE_SURT_ARTIFACTS_FA.md)
- [Recorded smoke evidence and remaining measurements](docs/experiments/surt_smoke_2026-11-12.md)
- [Successful synthetic-smoke procedure](docs/KAGGLE_SURT_RUNBOOK.md)

## Implementation boundaries

Audio handling, controlled two-source mixtures, split validation, Persian text
utilities, scheduler/state contracts and short-mixture PI-WER are available.
The actual pinned Icefall recipe is the current model-execution path. A compatible
trained native export and the external training backend for this repository's
wrappers are not supplied. Those missing wrappers do not block testing the
upstream pretrained recipe. Mini Persian vocabulary transfer has actual Kaggle
evidence; ORC-WER, continuous streaming and useful adaptation quality still need
their own implementation and validation.

The old single-stream comparison model, baseline notebook and heuristic-router
artifacts have been removed. Generic VAD/OSD diagnostics are retained outside
the SURT recognition path. Shared device selection lives in `utils/device.py`.

## Existing diagnostic notebooks

These are tools from the English-data/native-export scaffold, not a completed
Persian adaptation pipeline or the mandatory order after the smoke run.

| Notebook | Purpose / prerequisite |
| --- | --- |
| `english-streaming-asr.ipynb` | Real English source data and controlled mixtures |
| `notebooks/12_surt_streaming_inference.ipynb` | Asset checks; inference requires a validated compatible native export |
| `notebooks/13_surt_training_evaluation.ipynb` | Data validation; training needs a real external backend |
| `notebooks/14_surt_evaluation.ipynb` | Compare complete identical test sets from valid records |

Notebook setup preserves the installed ML stack and records constraints,
environment and pip freeze. The legacy Icefall smoke uses an isolated Python
3.12/Torch cu121 environment, separate from the reported Python 3.13/Torch
cu128 notebook image. See [environment compatibility](docs/KAGGLE_ENVIRONMENT.md).

## Local checks

Heavy model/data processing belongs on Kaggle. The following are reference
commands for small fixture checks, not instructions to run them automatically;
the current standing instruction requires explicit authorization before tests:

```bash
python3 -m pip install -e '.[dev,audio]'
python3 -m pytest -q
python3 scripts/validate_notebooks.py
python3 -m compileall -q src scripts
```

Historical phase reports and previous environment-change reports are records
of earlier development, not current implementation instructions. Use the
[execution matrix](docs/execution_matrix.md),
[dependency map](docs/phase_dependency_map.md), and
[feasibility audit](docs/PERSIAN_SURT_FEASIBILITY.md) for current boundaries.
