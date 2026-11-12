# Dependency and environment freeze

## Declared environments

The lightweight core is intentionally consistent across the project manifests:

| Purpose | File | Core specification |
| --- | --- | --- |
| Installable package | `pyproject.toml` | Python `>=3.10`, NumPy `>=1.23`, PyYAML `>=6.0` |
| Developer/test install | `requirements.txt` | Core plus pytest `>=7.0` |
| Local approximation of supplied Kaggle image | `environment.yml` | Python 3.13, NumPy 2.1.3, PyYAML, pytest; no GPU stack |

Optional packages are deliberately separated: `soundfile` for audio, `webrtcvad-wheels` for Phase 2, and `datasets`/`huggingface_hub` for model/data paths. PyTorch, Transformers, PEFT, and exact SURT/Icefall dependencies are not pinned by the minimal artifact because no verified final model recipe was executed. Pinning an arbitrary version would be less reproducible than recording the exact external recipe after it is selected.

## Capturing an executed environment

`configs/kaggle_environment.json` transcribes the user-supplied
`kaggle_env_info.md`; it is a reference snapshot, not an executed lockfile.
The active notebook setup constrains the versions of installed ML/audio
packages dynamically, rather than forcing a different Kaggle image to install
that snapshot. Resolver conflicts stop setup. Each notebook saves these actual
constraints, a complete pip freeze and a timestamped environment report.
See [Kaggle environment compatibility](KAGGLE_ENVIRONMENT.md).

Run:

```bash
python3 scripts/check_environment.py --json
python3 scripts/audit_artifact.py --write
```

The audit report records Python/platform, Git revision/worktree state, and SHA-256 hashes of the core configuration and dependency files. A full empirical report must archive that generated report together with the exact external model/runtime package versions, CUDA/GPU details, checkpoint hashes, and data-manifest hashes.
