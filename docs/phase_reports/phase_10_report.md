# Phase 10 — Final reproducibility, thesis mapping, and research artifact

Phase 10 should be the final reproducibility, packaging, thesis-mapping, and research-artifact phase. Unlike earlier phases, it should not introduce new experimental ideas. Its job is to turn the completed system and Phase 9 evidence into a clean, reproducible research artifact suitable for thesis work, review, and future reproduction.

## 1. Objective

Finalize the repository as a traceable research artifact without adding a model, dataset, routing strategy, or training method.

## 2. Final system and configuration

The implemented candidate system is specified in [final_system_specification.md](../final_system_specification.md). It combines timestamped streaming audio, VAD, causal heuristic OSD, stabilized routing, normal WhisperRT, and an optional external SURT-compatible overlap branch. It is **IMPLEMENTED**, but no primary final experiment has selected verified external checkpoints; its final empirical status is **NOT REPRODUCED**.

`configs/final_reproducibility.yaml` links the final configs. `scripts/evaluate.py` captures Git revision/worktree state in its freeze metadata, while `scripts/audit_artifact.py` writes hashes for core configurations and dependency files.

## 3. Repository, dependencies, and environment

The repository was audited non-destructively. Large datasets, checkpoints, and outputs remain Git-ignored. Local ignored artifacts are preserved rather than deleted because they may be user data. Core dependencies remain intentionally lightweight and consistent across `pyproject.toml`, `requirements.txt`, and `environment.yml`; external ASR/SURT runtimes are optional and must be version-pinned by the reproducing researcher.

`scripts/check_environment.py` now reports project importability, package status/version, audio/model/data package availability, GPU/CUDA state, and writable runtime directories. It supports JSON and strict modes.

## 4. Reproducibility and tests

| Level | Status | Meaning |
| --- | --- | --- |
| Code | VERIFIED | Core dependencies/import, unit tests |
| Small protocol | VERIFIED | Code plus Phase 8/9 model-free smoke, evaluation validation, artifact audit |
| Full empirical | NOT REPRODUCED | Requires external checkpoints, Phase 3 final manifests/audio, and per-example final records |

The final entry point is `scripts/reproduce.py`. Notebook JSON validity and optional integration-test skip behavior remain part of the test suite.

## 5. Experiments, results, and provenance

No final results were fabricated. Final WhisperRT/SURT/fine-tuned checkpoints and per-example test records are absent, so WER, PI-WER, OSD F1, routing outcomes, ablations, latency, RTF, memory, robustness, external evaluation, and significance are all **NOT MEASURED**. The Phase 9 evaluator will only calculate/report values from supplied per-example records and leaves missing rows as `NOT MEASURED`.

## 6. Thesis mapping, contributions, limitations, and future work

The final thesis traceability table is in [thesis_mapping.md](../thesis_mapping.md). Research-question answers are explicitly bounded in [final_research_answers.md](../final_research_answers.md). Contribution classifications, validity threats, responsible-reporting constraints, and claim status are in [final_claims_audit.md](../final_claims_audit.md). Future work is in [future_work.md](../future_work.md).

## 7. Final research status

This is a coherent **engineering and reproducibility artifact**, not a completed empirical thesis result package. It is ready for a competent researcher to supply the documented external assets, execute the frozen protocol, and generate the final evidence without silently changing the system.

---

Finalization is complete as a code and reproducibility artifact. It is not presented as a completed empirical result package.

Implemented:

Final reproducibility entry point:
reproduce.py
Environment verifier with package/GPU/version reporting:
scripts/check_environment.py
Non-destructive security, large-file, license, citation, Git-state, and SHA-256 provenance audit:
audit_artifact.py
All-notebook JSON validation:
validate_notebooks.py
Final reproducibility config:
final_reproducibility.yaml
Citation metadata:
CITATION.cff
Final system, dependency, claims, research-answer, future-work, artifact-checklist, and Phase 10 documentation.

Verification status:
Area / Status
Unit tests / 44 passed, 1 skipped
Phase 10 smoke / test Passed
Notebook JSON validation / 11/11 valid
Environment/core import / Verified
Artifact/security/large-file audit / No findings
Code reproduction level / Verified
Small protocol reproduction level / Verified
Full empirical reproduction / Not reproduced

The verified local environment used Python 3.12.0 with NumPy 1.26.1, PyYAML 6.0.3, pytest 9.1.1, and WebRTC VAD installed. PyTorch, Transformers, datasets, PEFT, Hugging Face Hub, and a GPU were unavailable.

The final artifact audit records a dirty worktree and audits both tracked and untracked artifact files. It preserves local ignored outputs, datasets, checkpoints, and temporary artifacts rather than deleting potentially user-owned data.

The final system architecture is documented, but the primary empirical system is still not evaluated because final WhisperRT, SURT, and fine-tuned SURT checkpoints are unset, and final Phase 3 test manifests/per-example records are unavailable. Therefore the project makes no verified claims about final WER, PI-WER, OSD F1, adaptive-routing gains, latency, RTF, memory, robustness, generalization, or fine-tuning improvement.

Use:
``` bash
python3 scripts/reproduce.py --level code
python3 scripts/reproduce.py --level small
python3 scripts/reproduce.py --level full
```

--level full was deliberately tested and returned NOT REPRODUCED because it requires the missing WhisperRT checkpoint, SURT checkpoint, fine-tuned SURT checkpoint, and final Phase 3 test manifest.
