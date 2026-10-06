# Phase 9 — Comprehensive evaluation, ablation, and robustness study

Phase 9 should be the main scientific evaluation and ablation phase. At this point the implementation should largely be complete; the focus shifts from adding major functionality to establishing which components actually contribute, measuring trade-offs, testing generalization, and producing thesis-quality results.

## 1. Research questions and frozen system

Phase 9 asks whether streaming WhisperRT degrades under overlap, whether OSD/routing selects the useful branch, whether the selected multi-talker model improves overlap recognition, and what latency/compute/robustness trade-offs result. The architecture is frozen: WhisperRT, the heuristic OSD, Phase 6 router, the external SURT 2.0 adapter, and the Phase 8 training-feasibility decision. No model, OSD, router, or data-generation redesign occurred in this phase.

[`configs/evaluation.yaml`](../../configs/evaluation.yaml) is the freeze manifest. The evaluation command resolves its Git revision and records whether the worktree was dirty. Exact checkpoints are currently null because none was installed or evaluated; this means it is a validated protocol freeze, **not** a final measured-system freeze.

## 2. Dataset and protocol

The primary protocol references separate Phase 3 train, validation, and test manifests. Test protection is required by configuration. Synthetic results can be stratified by the existing overlap-regime and temporal-condition metadata; relative-gain values remain stored per example. External evaluation is **NOT MEASURED** because no independently prepared external corpus was available.

## 3. Baselines and ablations

The planned primary systems are clean WhisperRT, overlap WhisperRT, base multi-talker, fine-tuned multi-talker, and adaptive predicted OSD. Oracle adaptive is stored separately and excluded from the deployable-system table. The frozen six-row ablation plan covers baseline, VAD-only, OSD-only, routed WhisperRT, adaptive multi-talker, and adaptive fine-tuned multi-talker.

## 4. Implemented evaluation protocol

Phase 9 adds versioned per-example JSONL records, weighted corpus aggregation when error/reference counts are supplied, deterministic bootstrap intervals, paired-comparison guards, OSD/routing/error-tag storage, overlap/temporal strata, and a separate compute benchmark path. The report and markdown tables are generated only from those records; absent rows render as `NOT MEASURED`.

The protocol deliberately rejects an improvement comparison unless the baseline and candidate have identical sample IDs plus the same metric direction. Stream/channel recognition remains evaluated using the appropriate ordinary or permutation-invariant metric; it does not imply speaker identity.

## 5. Results, robustness, and statistical analysis

No final model checkpoint, Phase 3 test manifest, external corpus, or execution hardware was available in this checkout. Consequently, the following are **NOT MEASURED**: all main ASR metrics, overlap-ratio effects, OSD/routing metrics, ablations, fine-tuning effect, latency, RTF, GPU memory, robustness, unseen-speaker generalization, external evaluation, and confidence intervals for a real system.

The bootstrap implementation is tested on synthetic unit records only. That verifies arithmetic and reproducibility, not scientific significance.

## 6. Reproducibility

```bash
python3 scripts/evaluate.py --config configs/evaluation.yaml --validate-only
python3 scripts/evaluate.py --config configs/evaluation.yaml --records outputs/predictions/final_records.jsonl
python3 scripts/benchmark.py --records outputs/predictions/final_records.jsonl
python3 scripts/smoke_test.py --phase 9 --config configs/evaluation.yaml
```

Real evaluation records must identify system, sample, split, dataset, metric/direction, error/reference counts where applicable, overlap condition, oracle status, timing, memory, and error tags. The final evaluator writes JSON metadata plus a report table; numbers must never be typed manually.

## 7. Phase 10 requirements

Before thesis claims can be finalized, install verified checkpoints and data, lock a clean Git revision, run all comparable primary systems on the untouched test set, perform the planned ablations/strata, measure resource costs, and generate final tables and figures from those records. Do not claim accuracy, real-time performance, robustness, or generalization until those measurements exist.

---

Implemented the Phase 9 evaluation, ablation, robustness, and reporting protocol without changing the ASR, OSD, routing, or training architecture.

What’s now in place:

Frozen evaluation configuration in
configs/evaluation.yaml
, including system/checkpoint references, manifests, seeds, test protection, primary/oracle systems, and the six-row ablation plan.
Versioned per-example EvaluationRecord JSONL schema in
records.py
.
Aggregate evaluation with:
weighted corpus metrics when error/reference counts exist;
deterministic bootstrap confidence intervals;
overlap-regime and temporal-condition stratification;
error taxonomy;
guarded paired improvement calculations that reject incomparable samples or metrics.
Separate compute benchmarking for RTF, first-output/finalization/chunk latency, and GPU-memory summaries.
Programmatic main-result, OSD, ablation, and computational Markdown tables. Missing data remains NOT MEASURED.
Record-driven figure helpers for overlap metrics, latency–accuracy comparisons, and computational cost.
Updated
scripts/evaluate.py
,
benchmark.py
, Phase 9 smoke coverage, notebook, README, architecture/experiment/thesis docs, and
MD
Phase 9 report.
