Implemented Phase 0 only.
Repository changes
Added:

- pyproject.toml, requirements.txt, environment.yml, .gitignore, LICENSE
- Full src/streaming_asr/ package structure
- YAML configuration and merging
- Local/Kaggle path handling
- Audio chunks, buffers, streams, and preprocessing
- Streaming state and engine lifecycle
- Dataset/model backend abstractions
- Dummy VAD and overlap detector
- Pipeline skeleton
- Metric interfaces and basic deterministic metrics
- Logging and reproducibility utilities
- Environment checker and smoke test
- Unit tests
- Phase notebooks and documentation
- Initial [README.md](../README.md)
Validation
- Python: 3.12.0
- OS: macOS ARM64
- GPU/CUDA: unavailable
- pytest: PASS — 9 tests passed
- Local smoke test: PASS
- Kaggle configuration smoke test: PASS
- Configuration merging/path validation: PASS
- Notebook JSON validation: PASS
- No models or datasets downloaded
- No Phase 1 functionality implemented
PyYAML and pytest were installed as lightweight Phase 0 dependencies.
Known issues
- Optional packages such as PyTorch, Transformers, and SoundFile are not installed.
- The notebook was structurally validated but not executed interactively.
The repository is ready for:
Phase 1 — WhisperRT Streaming Baseline

Start with:
python3 scripts/run_baseline.py
