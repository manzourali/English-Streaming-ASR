# Current execution steps

The old VAD/OSD → router → WhisperRT/SURT runbook has been retired.
The complete current instructions are in [the Persian Kaggle workflow](../docs/KAGGLE_WORKFLOW_FA.md).

1. Run `english-streaming-asr.ipynb` to prepare and persist real data.
2. Run `notebooks/12_surt_streaming_inference.ipynb` for official asset download
   and explicit native readiness checks. Actual inference requires the missing
   native connection/export and feature/checkpoint validation to be completed.
3. Optionally use `notebooks/13_surt_training_evaluation.ipynb` after an actual
   training backend exists; validate-only does not train or produce weights.
4. Export and validate any fine-tuned checkpoint, repeat inference, then run
   `notebooks/14_surt_evaluation.ipynb` on identical full test sets.

`notebooks/11_whisperrt_baseline.ipynb` is an optional independent comparison.
Training is not required to use a compatible pretrained model. GPU access does
not replace missing model/training integration code.
