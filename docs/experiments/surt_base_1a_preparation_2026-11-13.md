# SURT Base English 1A — local preparation

Date: 2026-11-13. Status: **prepared locally; real Kaggle execution pending**.
No model weights, real audio corpus, or GPU stack were downloaded/installed on the Mac.
Initial Git status was clean; no branch, commit, push, or existing experiment changes were made.

The independent execution guide is `docs/KAGGLE_SURT_BASE_ENGLISH_RUNBOOK_FA.md`.
Cell 1 updates the existing Kaggle Git checkout to an explicitly supplied pushed project SHA,
records that SHA, and refuses tracked edits or file collisions. It preserves the separate
model/data directories and the existing isolated environment. The optional small workspace
bundle remains available via `scripts/package_surt_base.py`, but the runbook now uses Git.
No local commit or push has been performed by the agent.

Official small HF metadata and pinned Icefall source were inspected. The model revision is
`5413ea7aac8d667d36127260ce5fa54253d74270`; the selected weight is
`exp/surt_base/pretrained.pt`, 106944796 bytes, LFS SHA256
`919bb9454526fb9ad963eb5a4168940a43c3b8331017a1c9ade477197c7072d4`.
The companion BPE/tokens and original/patched recipe hashes are pinned in
`configs/surt_base/pins.json`. Metadata evidence and primary source addresses are saved separately.
Actual weight tensor compatibility remains a Kaggle check, not a local success claim.

Icefall stays at `f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1`.
The exact successful smoke SimpleCutSampler diff and observed package freeze were extracted
from `checkpoints/surt-smoke-logs-20261113T073310Z.tar.gz` and preserved in ordinary small files.
The planned package deviations are NumPy 2.5.4 → 1.26.4 for the NumPy/Torch bridge and
SciPy 1.18.1 → 1.13.1 for compatibility with that NumPy pin; before/after freeze and a real bridge operation are mandatory.
Core CUDA wheels, Python isolation and recipe versions are unchanged.

The official pretrained entry point gets a recorded patch for strict tensor loading, correct
channel-major output grouping, inference_mode, one warmup pass and synchronized measurements.
The native exporter is not involved. Two real English utterances (different speakers), then
their full-duration mixture at a real 0.5-second offset, produce three inspectable records.
Two-stream PI-WER includes empty-reference branch insertions on clean inputs; it is not an
official LibriCSS ORC-WER evaluation, held-out generalization claim, or Persian evaluation.

Local validation:

```text
pytest -q tests/test_surt_base_english.py tests/test_kaggle_surt_fixture.py tests/test_kaggle_workflow.py tests/test_kaggle_environment.py
26 passed
Python AST / JSON configs / bash -n for all five notebook cells: OK
```

Tests cover both pinned patches, branch grouping, one-load/two-pass structure, error counts,
permutation/empty-stream scoring, distinct speakers, mixture duration/gains/true overlap,
the Mac downloader guard, and a failed-run archive retaining logs/hashes while excluding audio
and weights. Array fixtures test arithmetic only. Local soundfile/GPU import, audio decoding,
BPE runtime parity, tensor strict load and recognition quality were not executed.

Run Kaggle cells 1–5 in order; on a failure stop model work and run cell 5 for diagnostics.
Return `/kaggle/working/surt-base-1a-review.tar.gz`. Inspect both branches, references, score
counts, contract and resource measurements before accepting gate 1A. After acceptance the
next task is real mini Persian transfer/update (1B); no long training starts automatically.

Follow-up: the local Git-update fixture covers a fresh clone and an existing checkout,
repeat updates, preservation of an ignored downloaded weight, and refusal of tracked edits.
The Kaggle notebook should run only its five 1A cells; the previous smoke training cells
are excluded. Cell 1 alone refreshes source without touching the environment or assets.

Kaggle follow-up, 2026-11-13: cell 2 failed at `pip check`, before preflight/model loading:
`scipy 1.18.1 has requirement numpy<2.8,>=2.0.0, but you have numpy 1.26.4.`
The earlier NumPy-only repair missed this SciPy constraint. Preserve the observed freeze;
change the executable requirements to SciPy 1.13.1 and repair only NumPy/SciPy in an existing
environment, using binary wheels without dependencies. Verify both exact versions and a tiny
compiled SciPy operation in preflight, alongside the existing NumPy/Torch round trip.
Official compatibility/wheel sources: https://docs.scipy.org/doc/scipy/dev/toolchain.html
and https://pypi.org/project/scipy/1.13.1/. Real Kaggle rerun remains pending; no weights,
audio, scientific packages or GPU stack were downloaded/installed locally for this fix.

Fix validation: 26 CPU contract tests passed; all five Kaggle cells passed bash -n,
runtime.py passed AST parsing, and git diff --check passed. The new regression check
keeps fresh-install, existing-environment repair and preflight pins consistent.

Second Kaggle follow-up, 2026-11-13: attachment reports `START 04-decode` (not cell 2).
Preflight passed on Python 3.12.10, NumPy 1.26.4, SciPy 1.13.1 and one Tesla T4;
the launcher then refused an existing `results/result.json` before invoking the recipe.
The attachment does not contain the existing result or prove model loading/recognition.
The recovery is to run cell 5 now and inspect its small archive, preserving this session,
assets and prior output. The guard now explains that action instead of suggesting a fresh
session. GPU monitor logs get unique names so another attempt preserves earlier samples.
An offline regression verifies the guard launches no recipe, retains all existing bytes,
and directs incomplete prior results to review without calling them successful.
Validation: 27 CPU tests passed, all five cells passed bash syntax checks, the launcher
passed AST parsing, and git diff --check passed. Real decode evidence remains pending.
