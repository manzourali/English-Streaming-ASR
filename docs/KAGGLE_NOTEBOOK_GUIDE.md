# Active Kaggle notebook guide

The previous routed WhisperRT/SURT notebook sequence is retired. Use the
complete [Persian workflow](KAGGLE_WORKFLOW_FA.md).

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
