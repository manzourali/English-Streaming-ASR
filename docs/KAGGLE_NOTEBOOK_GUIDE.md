# Active Kaggle notebook guide

The previous routed WhisperRT/SURT notebook sequence is retired. Use the
complete [Persian workflow](KAGGLE_WORKFLOW_FA.md).

All five notebooks use `scripts/setup_kaggle.py`: existing ML package versions
are written as pip constraints before installing lightweight extras. The setup
requires binary wheels for dependencies and stops on dependency conflicts.
Each session records all visible GPUs and package versions in `environment.json`.
See [environment compatibility](KAGGLE_ENVIRONMENT.md) for the supplied Python
3.13/Torch cu128 image and the separate legacy Icefall environment.

1. `english-streaming-asr.ipynb`: environment and real speech-mixture preparation.
2. `notebooks/12_surt_streaming_inference.ipynb`: base weights, explicit native
   readiness check, actual inference only when a compatible export exists.
3. `notebooks/13_surt_training_evaluation.ipynb`: optional data validation and
   training through an externally implemented training backend.
4. Repeat notebook 12 for a validated fine-tuned export, then use
   `notebooks/14_surt_evaluation.ipynb` on complete identical test sets.

Notebook 11 is an independent optional WhisperRT comparison. Notebooks 00–10
and `12_adaptive_e2e_surt.ipynb` were removed because they duplicated or
contradicted this execution path. Model exports and actual training integration
remain engineering prerequisites; downloading a checkpoint does not supply them.
