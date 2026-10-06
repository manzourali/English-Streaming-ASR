# Reproducibility guide

## Reproduction levels

| Level | Command | Meaning | Current status |
| --- | --- | --- | --- |
| Code verification | `python3 scripts/reproduce.py --level code` | Core environment import and unit tests | **VERIFIED** in this checkout |
| Small protocol verification | `python3 scripts/reproduce.py --level small` | Code checks plus Phase 8/9 model-free smoke tests, frozen-evaluation validation, and artifact audit | **VERIFIED** in this checkout |
| Full research result reproduction | `python3 scripts/reproduce.py --level full` | Runs evaluation from final test records and verified checkpoints | **NOT REPRODUCED**; required assets are absent |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/check_environment.py
```

For Conda, use `conda env create -f environment.yml`. Optional model/data dependencies are intentionally not installed by the minimal environment because the exact external SURT and WhisperRT runtimes must be version-pinned by the researcher who supplies them.
See [dependency_freeze.md](dependency_freeze.md) for the declared environment boundaries and executed-environment capture procedure.

## Inputs required for full reproduction

1. A clean Git revision and an audit manifest.
2. Phase 3 train/validation/test manifests and referenced audio.
3. Verified WhisperRT model package/checkpoint.
4. Verified external SURT decoder/checkpoint plus its exact runtime revision.
5. If applicable, verified fine-tuning checkpoint and external training recipe.
6. Per-example final records at `outputs/predictions/final_records.jsonl` or an explicitly supplied path.

## Commands

```bash
# Environment and artifact audit
python3 scripts/check_environment.py --json
python3 scripts/audit_artifact.py --write

# Tests and model-free smoke paths
python3 -m pytest -q
python3 scripts/validate_notebooks.py
python3 scripts/smoke_test.py --phase 8 --config configs/training.yaml
python3 scripts/smoke_test.py --phase 9 --config configs/evaluation.yaml

# Validate final protocol, then aggregate actual records
python3 scripts/evaluate.py --config configs/evaluation.yaml --validate-only
python3 scripts/evaluate.py --config configs/evaluation.yaml --records outputs/predictions/final_records.jsonl
python3 scripts/benchmark.py --records outputs/predictions/final_records.jsonl
```

Notebook files are JSON-validated by the final test suite. Their model/data cells require the same external assets and must not be described as reproduced when those assets are unavailable.
