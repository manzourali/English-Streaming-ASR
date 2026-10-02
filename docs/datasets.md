# Datasets

Phase 1 uses LibriSpeech `test-clean` as the clean single-speaker baseline. The Hugging Face adapter maps this to `openslr/librispeech_asr`, configuration `clean`, split `test`; no speaker identity is used by ASR. A real sample count and duration are recorded only after a benchmark run. Synthetic overlap, LibriSpeechMix, LibriCSS, and optional AMI/CHiME-6 remain planned later-phase datasets.
