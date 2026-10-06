# Dependency and environment freeze

## Declared environments

The lightweight core is intentionally consistent across the project manifests:

| Purpose | File | Core specification |
| --- | --- | --- |
| Installable package | `pyproject.toml` | Python `>=3.10`, NumPy `>=1.23`, PyYAML `>=6.0` |
| Developer/test install | `requirements.txt` | Core plus pytest `>=7.0` |
| Conda baseline | `environment.yml` | Python 3.10, NumPy, PyYAML, pytest |

Optional packages are deliberately separated: `soundfile` for audio, `webrtcvad-wheels` for Phase 2, and `datasets`/`huggingface_hub` for model/data paths. PyTorch, Transformers, PEFT, and exact SURT/Icefall dependencies are not pinned by the minimal artifact because no verified final model recipe was executed. Pinning an arbitrary version would be less reproducible than recording the exact external recipe after it is selected.

## Capturing an executed environment

Run:

```bash
python3 scripts/check_environment.py --json
python3 scripts/audit_artifact.py --write
```

The audit report records Python/platform, Git revision/worktree state, and SHA-256 hashes of the core configuration and dependency files. A full empirical report must archive that generated report together with the exact external model/runtime package versions, CUDA/GPU details, checkpoint hashes, and data-manifest hashes.
