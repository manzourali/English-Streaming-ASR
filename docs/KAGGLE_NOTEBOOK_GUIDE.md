# Kaggle entry points

The current task is Persian language adaptation of SURT 2.0. Follow the
[roadmap](PERSIAN_SURT_ROADMAP_FA.md) and [immediate next-step prompt](prompts/PERSIAN_SURT_NEXT_STEP.md),
not the old English scaffold's notebook order.

The completed synthetic smoke is documented in `KAGGLE_SURT_RUNBOOK.md`.
The next real-speech baseline should run the pinned upstream recipe directly.

Four diagnostic notebooks remain: the root English data notebook, 12 (native
export checks/inference), 13 (external-backend training scaffold) and 14
(record evaluation). They do not implement Persian adaptation. Their setup
uses `scripts/setup_kaggle.py` with installed-stack constraints and saves
`environment.json` and `pip-freeze.txt`.

See [environment compatibility](KAGGLE_ENVIRONMENT.md) and
[log archive/retention](KAGGLE_SURT_ARTIFACTS_FA.md). The old independent
single-stream baseline notebook and retired router code have been removed.
