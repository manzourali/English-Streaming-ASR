# Persian SURT feasibility — current evidence and open gates

Updated 2026-11-20. The current research target is Persian language adaptation
of SURT 2.0 for overlapping voices. The [roadmap](PERSIAN_SURT_ROADMAP_FA.md)
and [stage prompts](prompts/PERSIAN_SURT_FOLLOWUPS.md) define the next gated work.
Stages 1A, 1B and technical 1C are accepted. Stage 2 local implementation was authorized and prepared on 2026-11-20; its gate remains OPEN.

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

Stage 1C is technically accepted after review of `streamb-20261120-a` at
`outputs/streamb-20261120-a/`, project commit
`ca2c3be77e214f383e5fef841a01f008642b2cc1`. Eight saved upstream source hashes
match 1C-A, dependencies retain their pins, and 1B inputs remained unchanged.
Three native GPU tests passed (38.477s), including 21 boundary prefixes and a
120s repeated-real-audio storage stress; three real kaldifeat tests also passed.

- Same-context full-mixture parity: max mask error 9.536743e-6; final tensor
  error 7.688999e-6, within the unchanged atol=2e-4, rtol=1e-4 contract.
- Ordered branch IDs and token frame indices match the reference and packet
  partitions. Reset, repeated EOF and EOF on the last packet passed.
- For 444 features, 96 encoder frames preceded EOF and 13 were flushed at EOF,
  totaling the expected 109. Branch tokens appeared at received audio times
  0.7s and 2.0s, before the 4.436s file ended.
- Acoustic storage at aligned stress checkpoints stayed at 2,515,808 bytes;
  measured peak was 2,534,112 bytes (2.42 MiB). Repetition tests storage, not
  recognition of a real long conversation.
- One Tesla T4, FP32, two warmups/five measurements, CUDA synchronization:
  median compute RTF 0.230645, allocated peak 289.02 MiB, reserved 432 MiB.
  These are inference measurements, not full-training memory/cost estimates.
- First nominal acoustic context is 647.5ms. With 100ms packets, first output
  follows 700ms received audio; simulated readiness is about 764ms. Block-anchor
  simulated delays span 723–804ms. Live delivery and endpoint remain unmeasured.

`result.json` now records the authorized post-run review as `ACCEPTED_TECHNICAL`;
`result.before-review.json` preserves the original `REVIEW_REQUIRED` result and
its hash is included in the review. Raw logs and measurements are unchanged.
The initial restore attempt lacked `features/train.jsonl.gz`; a later restore
succeeded. Three nonblocking Git messages also remain in the native cell log;
explicit revision/hash checks and the experiment completed successfully.

The checkpoint still contains just **one saved Persian training update**.
Seven 1C-A resource-probe updates were disposable and did not change that
checkpoint. Most new vocabulary rows were newly initialized in 1B. Outputs
«ست» and «ویی» are compatible with this minimal training; they establish neither
useful Persian recognition nor the impossibility of successful adaptation.

## What has not passed

| Gate | Remaining evidence |
| --- | --- |
| Useful Persian recognition | Controlled adaptation beyond the accepted one-step mini; coverage and independent quality evidence |
| Live deployment and long-conversation quality | Technical bounded streaming passed; measured microphone/network/endpoint latency and real long-conversation evaluation remain |
| Training-scale resource feasibility | 1C-A short-batch resource probe and 1C-B inference measured; pilot-training and larger-scale costs still need evidence |
| Thesis quality | Controlled data simulation/adaptation and held-out ORC-WER, not tone WER |

**Decision: the Persian project's feasibility phase is not complete.** A tiny
successful Persian update is a structural gate, not proof of learned accuracy.
The technical streaming gate is closed as accepted under its same-context B=1
protocol; the overall Persian quality/cost feasibility decision remains open.

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

The prepared stage 2 prompt in [the follow-up prompts](prompts/PERSIAN_SURT_FOLLOWUPS.md)
limits the next work to source/access verification, bounded pilot-data selection,
frozen speaker/source-disjoint splits, text policy and tokenizer readiness.
**Stage 2 is now authorized only for bounded data/text preparation.** See the
[independent Kaggle runbook](KAGGLE_SURT_PILOT_2_RUNBOOK_FA.md) and
[source/access and data contract](SURT_PILOT_DATA_CONTRACT_FA.md). Local tooling,
versioned schema/config/text policy, exposure inventory and Kaggle-only tests
are written; no repository tests or pilot audio/tokenizer processing have run.
User-provided Kaggle access/metadata probes are now reviewed as limited evidence. The official
10.46GB archive is not an authorized acquisition route under the 256MiB pilot
cap; a suitable official subset route and its metadata remain unverified.
The user supplied Kaggle stdout showing HTTP 200 link issuance, HTTP 206 for a
one-byte Range, and 5000 train.tsv rows after 9,699,328 reported archive bytes,
with four speakers. The sanitized evidence summary and attachment hash are in
`configs/surt_pilot/user-access-evidence-v1.json`; this was not reproduced locally.
HEAD returned 403 while ranged GET succeeded, so HEAD success is not required.
Account and prefix-metadata access are evidenced; selected audio access is not.
The reported metadata-directed transfer exceeds the unchanged 8MiB metadata cap.
The full TSV is 9,403,215 bytes, but the size of the parsed 5000-row portion was
not reported. Sequential gzip traversal does not establish random clip access.
Only three exposure identifiers were supplied; 4997 inspected rows are missing
from the available ledger. Known hashes are excluded by plan; history remains
incomplete and finalize keeps the gate open. No automatic prefix downloader is
enabled and no sample endpoint result was supplied. The bounded-access probe
now sanitizes the observed U+200D credential artifact and compares API archive
identity with the reviewed metadata, without claiming a computed archive hash.
Saved mini ledgers contain 600 source IDs, 183 speaker hashes and 9 audio hashes;
actual CV27 overlap cannot be checked without data. Cross-release speaker
continuity and reencoded duplicate exclusion also need independent evidence.
No stage 3, FastMSS or training has started. Gate 2 stays OPEN.
Do not rerun accepted stages, start full training or download a whole corpus.
Later simple-mixture pilot adaptation and held-out evaluation must establish
learning beyond the one-update baseline. FastMSS comparison is a separate,
alignment-dependent research extension, not a prerequisite for the first basic
Persian-learning decision. Training budgets and success criteria must be set
before those experiments; no final WER target is inferred from current results.
