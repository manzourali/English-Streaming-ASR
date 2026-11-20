# Project objective and standing execution rules

The project's only ASR model is SURT 2.0. The objective is Persian language adaptation for continuous two-speaker transcription, including overlapping speech. Output branches are not persistent speaker identities unless demonstrated. Legacy model comparisons, routing and VibeVoice are outside the current scope.

## Current stage and sources of truth

- Roadmap and acceptance criteria: `docs/PERSIAN_SURT_ROADMAP_FA.md`.
- Stage prompts: `docs/prompts/PERSIAN_SURT_FOLLOWUPS.md`.
- Stage 1A is accepted. Stage 1B's real Kaggle run `mini-20261119-d` completed a genuine update and strict reload; the user confirmed all eight clean recordings and four complete mixtures on November 19, 2026. This is a pipeline feasibility result, not successful Persian recognition: mini validation/test PI-WER is 100%, and the tiny tokenizer has held-out coverage gaps.
- Stages 1A, 1B and the technical 1C gate are accepted. Run `streamb-20261120-a` in `outputs/streamb-20261120-a/` demonstrates full-model bounded stateful streaming, same-context partition parity, reset/EOF/tail correctness and pre-EOF tokens. Review accepted on 2026-11-20; `result.json` records `ACCEPTED_TECHNICAL`, with the original `REVIEW_REQUIRED` result preserved in `result.before-review.json`. See `docs/PERSIAN_SURT_FEASIBILITY.md` for evidence and limits.
- 1C acceptance is scoped to the pinned B=1/two-branch FP32 protocol and 120s repeated-audio storage stress, not useful Persian quality, real long-conversation quality, measured live delivery/endpoint latency or fixed speaker identities. The saved checkpoint has one genuine Persian update, not sufficient language adaptation; the seven 1C-A resource updates were disposable and were not saved to it.
- Stage 2 was explicitly authorized on 2026-11-20 for bounded pilot data/text preparation only. Local implementation is prepared in `scripts/surt_pilot/`, with `configs/surt_pilot/`, `docs/SURT_PILOT_DATA_CONTRACT_FA.md` and the independent `docs/KAGGLE_SURT_PILOT_2_RUNBOOK_FA.md`. The user supplied Kaggle stdout demonstrating official MDC API HTTP 200, ranged GET HTTP 206 and a 5000-row train.tsv prefix (four speakers); see `configs/surt_pilot/user-access-evidence-v1.json`. This proves limited account/metadata access, not bounded selected-audio access; reported metadata-directed transfer exceeded 8MiB and 4997 exposure identifiers are missing. A Kaggle Secrets-based control-plane probe is prepared; it never follows asset URLs. Gate 2 remains OPEN: a usable bounded official subset, CV27 data, cross-release speaker/duplicate evidence and Kaggle execution are pending. Never put API credentials or presigned URLs into project files or logs. No dataset/audio/checkpoint processing or tests were run locally; no stage 3, FastMSS or training is authorized. The 1C-B runbook `docs/KAGGLE_SURT_STREAMING_1CB_RUNBOOK_FA.md` and state contract `docs/SURT_1CB_STATE_CONTRACT.md` remain reproducibility references; 1C-A evidence remains in `outputs/surt-stream-1c/`.
- Do not start corpus-scale downloads, FastMSS production or full adaptation before the applicable feasibility evidence has been reviewed.

## Where work runs

- The local machine is for editing code, documentation and inspecting small saved reports. GPU work, model/checkpoint loading, audio processing, dependency installation, experimental execution and dataset downloads run only on Kaggle.
- Do not run tests locally or on Kaggle unless the user explicitly authorizes them. Writing tests and static source inspection are allowed. Never claim an unexecuted test or Kaggle experiment passed.
- Small upstream source inspection is allowed; never download model weights or datasets locally. Preserve existing local outputs and caches.
- The user commits and pushes changes. Do not create branches, commits, pushes, pull requests or remote changes without explicit authorization.
- Every Kaggle runbook starts by fetching and checking out the user's exact pushed commit, preserving local edits and previous outputs. Each shell cell starts with `%%bash`, sources its environment independently, preserves exit codes and writes full logs to files. Notebook output stays short.
- Each experiment attempt has a unique run ID. Never remove an old run, overwrite evidence or delete a download cache to bypass a failed run.
- Support both an existing successful session and a fresh Kaggle session. Restore the saved mini run to its original canonical path so absolute manifest paths remain valid. Rebuild the pinned environment only when absent. A small review archive excludes weights/audio/features and is not a restart bundle. Do not rerun phase 0 or stage 1A just to reconstruct stage 1B outputs.

## Model and evidence contracts

Use the pinned Icefall recipe and compatible official weights. Derive feature, token, source-target, HEAT, loss and state contracts from that exact revision. Keep Torch, CUDA, k2 and kaldifeat pinned. Do not silently relax runtime validation or reinterpret an unrelated native scaffold as the complete upstream recipe.

Record raw and normalized references, train-only vocabulary, speaker/source splits, hashes, precision, timing definitions and resource limits. Never use synthetic text, tones or repeated speech to demonstrate real recognition quality. Repeated real audio is acceptable only for explicitly labelled bounded-state stress tests.

For streaming, inspect all feature/mask/encoder/branch-tying/decoder states and right context. Require segmentation equivalence at the same context, reset, EOF/tail handling and bounded state on longer input. Whole-file decoding with a chunk-size flag is not proof of incremental inference. Keep unknown latency components null rather than reporting invented zeroes.

The unofficial `aliyzd95/common_voice_22_0_fa` mini subset is temporary pipeline data. The planned main source is Common Voice 27 from Mozilla Data Collective, subject to access and terms verification. Exclude all exposed mini speakers/sources/duplicates from the eventual final test. FastMSS requires an explicitly pinned alignment contract; its benefit must be measured against a comparable simple-mixture baseline.

Do not put audio, large features, checkpoints, environments or private data in Git. Keep restart artifacts private. State actual validation performed, material limitations and the next action clearly; make claims proportional to the available evidence.
