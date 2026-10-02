# Datasets

Phase 1 uses LibriSpeech `test-clean` as the clean single-speaker baseline. The Hugging Face adapter maps this to `openslr/librispeech_asr`, configuration `clean`, split `test`; no speaker identity is used by ASR. A real sample count and duration are recorded only after a benchmark run. Synthetic overlap, LibriSpeechMix, LibriCSS, and optional AMI/CHiME-6 remain planned later-phase datasets.

Phase 2 continues with clean LibriSpeech. Utterance-level LibriSpeech boundaries do not automatically provide frame-level speech/non-speech ground truth, so frame-level VAD precision/recall/F1 are not claimed without a valid reference annotation. The demo runner reports backend timing only unless scored reference labels are supplied.

Phase 3 uses clean LibriSpeech sources, or tiny local fixtures for development, to create two-speaker mixtures. Splits are generated separately and speaker leakage is checked when speaker IDs are available. JSONL manifests preserve source paths, transcripts, speaker IDs, timing, overlap interval, overlap ratio, and relative gain. Synthetic mixtures are controlled data, not evidence of real conversational-overlap performance.

Phase 4 consumes those JSONL manifests directly. Control mixtures may use a time-ordered concatenated reference for ordinary WER; overlapping mixtures retain both source references but do not receive ordinary single-reference WER.

Phase 5 derives frame labels independently from each manifest's source timing: zero active sources is `NO_SPEECH`, one is `SINGLE_SPEAKER`, and two or more is `OVERLAP`. A frame is positive when its open time interval intersects at least two source intervals. This timing oracle is evaluation ground truth only; it is not passed to the detector. The synthetic source set remains a controlled OSD development benchmark rather than a real-world overlap corpus.
