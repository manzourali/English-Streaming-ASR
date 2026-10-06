# Final research artifact checklist

| Area | Status | Evidence / limitation |
| --- | --- | --- |
| Source code | IMPLEMENTED | `src/`, scripts, configuration, and unit tests are present |
| Secrets / large-file audit | VERIFIED | `scripts/audit_artifact.py` found no sensitive/oversized file among audited tracked and untracked artifact files on the finalization run |
| Model IDs / checkpoints | PARTIALLY DOCUMENTED | IDs/interfaces are documented; final external checkpoints are absent |
| Data / split provenance | IMPLEMENTED | Phase 3 manifests and leakage/evaluation contracts exist; final assets absent |
| Evaluation protocol | IMPLEMENTED | Phase 9 records, tables, benchmark, and protected test configuration |
| Final empirical results | NOT REPRODUCED | No verified models/final test records/hardware run |
| Unit tests | VERIFIED | Run `python3 -m pytest -q` |
| Optional integration test | NOT RUN | It is skipped without an external SURT override |
| Notebook syntax | VERIFIED | `scripts/validate_notebooks.py` validates all notebook JSON; model/data execution pending |
| Kaggle / Hugging Face | NOT REPRODUCED | Paths/configuration exist; no final cloud run |
| License / citation | VERIFIED | `LICENSE` and `CITATION.cff` present |
| Thesis mapping / claims / future work | IMPLEMENTED | Final documentation links implementation status to evidence boundaries |
