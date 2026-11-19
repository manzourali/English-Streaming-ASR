# Project objective and standing execution rules

The project's only ASR model is SURT 2.0. The objective is Persian language adaptation for continuous two-speaker transcription, including overlapping speech. Output branches are not persistent speaker identities unless demonstrated. Legacy model comparisons, routing and VibeVoice are outside the current scope.

## Current stage and sources of truth

- Roadmap and acceptance criteria: `docs/PERSIAN_SURT_ROADMAP_FA.md`.
- Stage prompts: `docs/prompts/PERSIAN_SURT_FOLLOWUPS.md`.
- Stage 1A is accepted. Stage 1B's real Kaggle run `mini-20261119-d` completed a genuine update and strict reload; the user confirmed all eight clean recordings and four complete mixtures on November 19, 2026. This is a pipeline feasibility result, not successful Persian recognition: mini validation/test PI-WER is 100%, and the tiny tokenizer has held-out coverage gaps.
- Current work is stage 1C. Use `docs/KAGGLE_SURT_STREAMING_RUNBOOK_FA.md`. Native end-to-end streaming remains unproven until actual stateful inference passes the stated acceptance tests. API inventory, component tests and whole-utterance timings cannot close that gate.
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
