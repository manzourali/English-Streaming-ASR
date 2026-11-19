# Persian SURT feasibility — current evidence and open gates

Updated 2026-11-19. The current research target is Persian language adaptation
of SURT 2.0 for overlapping voices. The [roadmap](PERSIAN_SURT_ROADMAP_FA.md)
and [stage 1C runbook](KAGGLE_SURT_STREAMING_RUNBOOK_FA.md) define current work.

## What passed

The user ran the pinned Kaggle synthetic fixture procedure. The supplied
decode log shows CUDA checkpoint loading, all 12 scored groups and `Done!`;
the user separately confirmed checkpoint reload, one optimizer step and finite
saved tensors. This establishes infrastructure execution on synthetic tones.
The complete artifact archive and its hashes have not been reviewed locally.
See [recorded evidence](experiments/surt_smoke_2026-11-12.md).

Gate 1A is now accepted: all 44 review-package hashes match, the official Base
checkpoint loaded strictly, and both English clips plus their mixture decoded.
The user confirmed auditory/reference/two-branch review on 2026-11-13.
Bounded clean PI-WER counts: 0/8, 1/12; mixture: 4/20. See the
[1A review and 1B preparation record](experiments/surt_mini_1b_preparation_2026-11-13.md).

Stage 1B's technical gate is accepted for `mini-20261119-d`: genuine recipe
loss/backward/update, semantic vocabulary transfer, changed weights and strict
reload have saved evidence. The user confirmed eight clean references and four
complete mixtures on 2026-11-19. See the [review](experiments/surt_mini_1b_review_2026-11-19.md).
Mini validation/test PI-WER is 100%; held-out tokenizer coverage is incomplete.
This is structural feasibility, not learned Persian accuracy.

## What has not passed

| Gate | Remaining evidence |
| --- | --- |
| Useful Persian recognition | Controlled adaptation beyond the accepted one-step mini; coverage and independent quality evidence |
| Incremental streaming | Persistent bounded state, chunk/EOF behavior, context-compatible reference, real latency measurements |
| Resource feasibility | Peak device/process memory distinction, per-step time, data lengths and budget estimates on the actual GPU |
| Thesis quality | Controlled data simulation/adaptation and held-out ORC-WER, not tone WER |

**Decision: the Persian project's feasibility phase is not complete.** A tiny
successful Persian update is a structural gate, not proof of learned accuracy.
Whole-batch decode with constrained attention is not an incremental-state test.

## Existing scaffold and limits

The package contains audio streaming, features, two-source mixture manifests,
validation, a generic training/backend interface, optional Persian text/BPE
utilities, and native TorchScript composition for externally supplied exports.
It does not itself supply the trained model, full pruned-RNN-T training recipe,
Persian language adaptation or compatible native exporter. Use the pinned
upstream Icefall recipe for the next baseline/model work; lack of a local
native export must not block the real-English gate.

The existing `icefall_adapter` exports metadata, not validated recipe features
and clean-source mask targets. `train_asr.py --validate-only` validates records
without optimization. The local NumPy streaming feature profile has not been
shown equivalent to the checkpoint's recipe features. Short-mixture PI-WER is
available; continuous-conversation ORC-WER integration remains separate.

Optional Persian utilities:

```bash
python3 -m pip install -e '.[persian]'
python3 scripts/persian_tokenizer.py normalize train.raw.txt train.txt
python3 scripts/persian_tokenizer.py normalize valid.raw.txt valid.txt
mkdir -p data/persian
python3 scripts/persian_tokenizer.py train train.txt data/persian/bpe --vocab-size 500
python3 scripts/persian_tokenizer.py coverage data/persian/bpe.model valid.txt
```

These commands do not integrate a Persian model. Train vocabulary only on
training text; inspect blank/unknown IDs, lexical preservation, OOV and the
selected recipe's required vocabulary files. Preserve Latin script in mixed
text. Current normalization needs optional Hazm/num2words; installation in the
selected Python environment must be checked rather than assumed. Ambiguous
numbers/dates need a declared spoken-form policy.

Reusing an English checkpoint requires vocabulary-aware inspection, even when
matrix sizes happen to match. Report every reused, remapped and reset tensor;
reset incompatible optimizer state. Verify HEAT assignment, source timing,
mask loss, auxiliary CTC and their actual CLI semantics against the exact
pinned source. Do not replace them with dummy losses or silently skip weights.

For full data, enforce speaker/clip-disjoint splits before augmentation.
FastMSS currently documents word-aligned source inputs, so alignment or an
explicit validated input adaptation is needed beyond simple clip mixtures.
Source selection, augmentation budget and test data must be controlled when
comparing simulators. See the [roadmap sources](PERSIAN_SURT_ROADMAP_FA.md).

## Next action

Run the bounded [stage 1C component/resource procedure](KAGGLE_SURT_STREAMING_RUNBOOK_FA.md)
in Kaggle, after pushing the prepared code yourself. Restore the complete saved
1B directory in a fresh session; phase 0 and 1A do not need to be rerun. Its first
cell updates the checkout to the explicit pushed SHA. The first 1C pass gathers
source/API and component/resource evidence; native end-to-end streaming remains
open until exact state adapters and full reset/EOF/context tests pass.
Then use [the gated follow-up prompts](prompts/PERSIAN_SURT_FOLLOWUPS.md) for
stateful execution/resources, corpus preparation, adaptation and evaluation.
Keep all unexecuted measurements explicitly unmeasured until actual logs
exist. Archive current logs using [this single-file procedure](KAGGLE_SURT_ARTIFACTS_FA.md).
