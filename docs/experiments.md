# Experiments

An experiment is defined by YAML, a seed, dataset/model identifiers, and an output root. Phase 1 times model loading separately from per-utterance streaming processing using `time.perf_counter()`. WER lowercases text, removes punctuation other than apostrophes, and collapses whitespace; raw and normalized reference/prediction text are saved. RTF is wall-clock processing time divided by audio duration, under the tested hardware/software/chunk configuration. Run directories contain resolved config, metadata, logs, JSONL predictions, JSON metrics, and a Markdown report. No value is populated unless measured.

Phase 2 additionally records VAD frame size, aggressiveness, decision timing, and VAD RTF separately from ASR RTF. The current primary design forwards continuous audio to ASR, so VAD is observed rather than used to discard samples. Frame-level VAD accuracy remains unmeasured unless valid frame-level references are available.

Phase 3 reports dataset properties only: mixture count, duration, overlap duration/ratio, regime distribution, speaker distribution, gain distribution, validation errors, and generation outputs. It does not report ASR WER. Generation is seeded with NumPy and generated data remains outside Git.
