# Datasets

Phase 1 uses LibriSpeech `test-clean` as the clean single-speaker baseline. The Hugging Face adapter maps this to `openslr/librispeech_asr`, configuration `clean`, split `test`; no speaker identity is used by ASR. A real sample count and duration are recorded only after a benchmark run. Synthetic overlap, LibriSpeechMix, LibriCSS, and optional AMI/CHiME-6 remain planned later-phase datasets.

Phase 2 continues with clean LibriSpeech. Utterance-level LibriSpeech boundaries do not automatically provide frame-level speech/non-speech ground truth, so frame-level VAD precision/recall/F1 are not claimed without a valid reference annotation. The demo runner reports backend timing only unless scored reference labels are supplied.
