# English Streaming ASR with Overlapped Speech

The active research path is continuous SURT 2.0: mono 16 kHz PCM, 80-bin
log-Mel features, dual-path masks, shared streaming Zipformer, and stateless
RNN-T decoding. The heuristic WhisperRT/SURT router is retired.

**Read the [Persian Kaggle workflow](docs/KAGGLE_WORKFLOW_FA.md) first.** It
explains every component, notebook, input/output artifact and remaining gap.

## Current capability

Data preparation, controlled two-source speech mixtures, bounded streaming
scheduling, endpoint logic and PI-WER evaluation are implemented and tested.
Actual recognition still requires a validated native streaming model export.
The official pretrained checkpoint is downloadable, but the recipe-specific
native connection/exporter and actual training backend are not supplied.
Downloading weights or enabling a GPU does not implement those missing pieces.
No trained-model accuracy, GPU latency or full training reproduction is claimed.

## Kaggle notebook order

| Order | Notebook | Purpose |
| --- | --- | --- |
| 1 | [english-streaming-asr.ipynb](english-streaming-asr.ipynb) | Clone source, verify environment, download/attach real LibriSpeech, create and save mixtures |
| 2 | [12_surt_streaming_inference.ipynb](notebooks/12_surt_streaming_inference.ipynb) | Download official base weights, report native-export readiness, run actual inference when a compatible export exists |
| 3, optional | [13_surt_training_evaluation.ipynb](notebooks/13_surt_training_evaluation.ipynb) | Validate training data; train only after a real external backend is integrated |
| 2 again | Notebook 12 | Run a validated export of the fine-tuned checkpoint with a distinct system label |
| 4 | [14_surt_evaluation.ipynb](notebooks/14_surt_evaluation.ipynb) | Compare complete, identical test sets using generated EvaluationRecord files |
| Optional comparison | [11_whisperrt_baseline.ipynb](notebooks/11_whisperrt_baseline.ipynb) | Independent WhisperRT baseline; not a SURT prerequisite |

Every notebook includes its own setup for a fresh Kaggle session. Save outputs
under `/kaggle/working/asr_artifacts`; attach a previous version through Add
Input and set the exact input paths shown by Kaggle. Source changes must be
published to your GitHub repository before a Kaggle clone can see them.
Notebook 00–10 duplicates and the old adaptive notebook have been removed.

## Datasets and metrics

The default pilot downloads real `dev-clean` and partitions its speakers into
three development groups. This is not an official test-clean benchmark.
The standard layout uses train-clean-100, dev-clean and test-clean. Source
limits and mixture counts are explicit; a small pilot is not paper-scale
training. The `--demo` generator uses tones with fabricated labels and is
reserved for infrastructure tests.

Mixture audio paths are stored relative to their dataset bundle, so saved
Kaggle outputs can be attached under a new input directory. Native inference
writes both transcript JSONL and `*_evaluation_records.jsonl`; no manual
schema conversion is needed for `evaluate.py`.

PI-WER is implemented for short two-source mixtures. The official LibriCSS
continuous-conversation protocol uses ORC-WER; a LibriCSS/meeteval integration
is a separate remaining task. File-processing RTF is not live microphone
latency. VAD/OSD scripts remain independent diagnostic baselines, outside the
continuous SURT recognition path.

## Local verification

Heavy processing belongs on Kaggle. The following only checks code and small
fixtures and does not need a GPU:

```bash
python3 -m pip install -e ".[dev,audio]"
pytest -q
python3 scripts/validate_notebooks.py
python3 -m compileall -q src scripts
```

Kaggle setup uses `.[kaggle,dev]` and preserves the platform's PyTorch/CUDA
installation. Do not install arbitrary k2/CUDA wheels without matching the
actual pinned recipe and environment.

## Reference documents

- [Current Persian execution guide](docs/KAGGLE_WORKFLOW_FA.md)
- [Stage2 architecture and B1–B8 report](docs/stages/stage2_report.md)
- [Current dependency map](docs/phase_dependency_map.md)
- [Current execution matrix](docs/execution_matrix.md)
- [Historical phase reports](docs/phase_reports/)

Earlier phase reports and prompt files are historical evidence, not runnable
instructions for the current pipeline. See [CITATION.cff](CITATION.cff) when
citing the artifact; cite measured results only when actual model/data runs
and their provenance are archived.
