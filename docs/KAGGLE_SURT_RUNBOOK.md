# Kaggle SURT smoke test runbook

**Review only: not executed or verified on Kaggle.** Run in a fresh Kaggle
session with GPU and Internet enabled, only after the exact project commit has
been pushed and the user approves the run. This run generates synthetic tones;
any WER is a plumbing value, not speech-recognition quality. It does not prove
incremental stateful streaming, Persian feasibility, or real-time performance.

The commands below target Linux x86-64, **Python 3.12**, a CUDA 12.1-capable
NVIDIA driver, Torch 2.3.1 and the corresponding k2/kaldifeat wheels. Kaggle's
actual image and driver must satisfy those prerequisites. The matching wheel
names exist in the official k2 and kaldifeat indexes, but this combination and
Lhotse version have not been integration-tested. Stop on any failure and save
the logs; do not silently change versions. Core packages are pinned, but the
full transitive environment is recorded rather than locked in advance.

## 1. Clone pinned sources

Set `PROJECT_COMMIT` to the full committed hash supplied in the handoff, and
push that commit to the remote before opening Kaggle. A commit cannot contain
its own hash, so it is deliberately supplied externally here.

```bash
set -euo pipefail
export PROJECT_COMMIT="REPLACE_WITH_FULL_PROJECT_COMMIT"
export ICEFALL_COMMIT="f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1"
export WORK=/kaggle/working
export PROJECT="$WORK/English-Streaming-ASR"
export ICEFALL="$WORK/icefall"
export RECIPE="$ICEFALL/egs/libricss/SURT"
export FIXTURE="$WORK/surt-fixture"
export EXP="$WORK/surt-smoke-exp"
export LOG="$WORK/surt-smoke-logs"
mkdir -p "$LOG"
nvidia-smi | tee "$LOG/gpu-before.txt"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
git clone https://github.com/manzourali/English-Streaming-ASR.git "$PROJECT"
git -C "$PROJECT" checkout --detach "$PROJECT_COMMIT"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$PROJECT_COMMIT"
git clone https://github.com/k2-fsa/icefall.git "$ICEFALL"
git -C "$ICEFALL" checkout --detach "$ICEFALL_COMMIT"
test "$(git -C "$ICEFALL" rev-parse HEAD)" = "$ICEFALL_COMMIT"
printf 'project=%s\nicefall=%s\n' "$PROJECT_COMMIT" "$ICEFALL_COMMIT" | tee "$LOG/commits.txt"
```

## 2. Install the selected GPU stack

Use an isolated virtual environment in Kaggle; its Torch replacement does not
modify the base image. These downloads occur in Kaggle, never on the Mac.

```bash
python3 -m venv "$WORK/surt-venv"
source "$WORK/surt-venv/bin/activate"
python -m pip install 'numpy==1.26.4'
python -m pip install 'torch==2.3.1+cu121' 'torchaudio==2.3.1+cu121' \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'k2==1.24.4.dev20241030+cuda12.1.torch2.3.1' \
  -f https://k2-fsa.github.io/k2/cuda.html
python -m pip install 'kaldifeat==1.25.4.dev20240606+cuda12.1.torch2.3.1' \
  -f https://csukuangfj.github.io/kaldifeat/cuda.html
python -m pip install 'lhotse==1.33.0' 'sentencepiece==0.2.3' \
  'meeteval==0.4.3' 'pypinyin==0.50.0' einops tensorboard \
  kaldialign typeguard soundfile urllib3
python -m pip freeze | tee "$LOG/pip-freeze.txt"
export PYTHONPATH="$ICEFALL:$ICEFALL/egs/libricss/SURT/dprnn_zipformer"
python - <<'PY' | tee "$LOG/environment.txt"
import platform
import torch, torchaudio, k2.version, kaldifeat, lhotse, sentencepiece
from meeteval.wer import wer
print('platform', platform.platform())
print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)
print('torchaudio', torchaudio.__version__, 'k2', k2.version.__version__)
print('lhotse', lhotse.__version__, 'sentencepiece', sentencepiece.__version__)
assert torch.__version__.split('+')[0] == '2.3.1'
assert torch.version.cuda == '12.1' and torch.cuda.is_available()
print('GPU', torch.cuda.get_device_name(0))
print('VRAM bytes', torch.cuda.get_device_properties(0).total_memory)
assert callable(wer.orc_word_error_rate)
PY
```

## 3. BPE and bounded fixture

The English BPE is the existing model referenced by Icefall's stage 10, not a
new tokenizer. Assert its previously observed SHA-256. Do not call `prepare.sh`.

```bash
mkdir -p "$RECIPE/data/lang_bpe_500"
curl --fail --location --retry 3 \
  'https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/resolve/main/data/lang_bpe_500/bpe.model' \
  -o "$RECIPE/data/lang_bpe_500/bpe.model"
echo 'c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5  '"$RECIPE/data/lang_bpe_500/bpe.model" | sha256sum -c -
python "$PROJECT/scripts/kaggle_surt_fixture.py" build "$FIXTURE" | tee "$LOG/build.txt"
python "$PROJECT/scripts/kaggle_surt_fixture.py" validate "$FIXTURE" \
  --recipe-dir "$RECIPE" | tee "$LOG/validation.txt"
find "$FIXTURE/audio" -type f -print0 | sort -z | xargs -0 sha256sum | tee "$LOG/input-sha256.txt"
find "$FIXTURE/manifests" "$FIXTURE/features" -type f -print0 | sort -z | xargs -0 sha256sum | tee "$LOG/fixture-sha256.txt"
```

Validation must report `status: validated`, one training cut, all six nonempty
decode groups, and the observed empty-group probe. Check these rather than
assuming that the shell status proves every desired property.

## 4. One training batch and checkpoint inspection

The full and warm-up manifests each contain one cut; the `SimpleCutSampler`
and one epoch are intended to produce one update. The recipe consumes the
warm-up batch during its warm-up period. The assertions below verify the saved
batch counter, optimizer-state step and finite saved model/optimizer tensors.
They do not independently prove that every intermediate gradient was finite.
Record that limitation unless separate gradient instrumentation is approved.

```bash
cd "$RECIPE"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > "$LOG/gpu-memory.csv" &
GPU_MONITOR=$!
trap 'kill "$GPU_MONITOR" 2>/dev/null || true' EXIT
python dprnn_zipformer/train.py \
  --world-size 1 --num-epochs 1 --tensorboard false --use-fp16 false \
  --manifest-dir "$FIXTURE/manifests" --bpe-model data/lang_bpe_500/bpe.model \
  --exp-dir "$EXP" --max-duration 3 --max-duration-valid 3 --max-cuts 1 \
  --num-workers 0 --bucketing-sampler false --shuffle false --drop-last false \
  --enable-musan false --enable-spec-aug false \
  --save-every-n 1 --average-period 1 2>&1 | tee "$LOG/train.txt"
test -s "$EXP/checkpoint-1.pt"
sha256sum "$EXP/checkpoint-1.pt" | tee "$LOG/checkpoint-sha256.txt"
python - "$EXP/checkpoint-1.pt" <<'PY' | tee "$LOG/checkpoint-check.txt"
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert checkpoint['batch_idx_train'] == 1, checkpoint.get('batch_idx_train')
states = checkpoint['optimizer']['state']
assert states and any(int(state.get('step', 0)) == 1 for state in states.values())
def check_finite(value):
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        assert torch.isfinite(value).all(), 'non-finite checkpoint tensor'
    elif isinstance(value, dict):
        for item in value.values(): check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: check_finite(item)
check_finite(checkpoint['model'])
check_finite(states)
print('checkpoint reloaded; batch_idx_train=1; optimizer step=1; saved tensors finite')
PY
```

`gpu-memory.csv` is periodic whole-device sampling, not an exact per-process
peak. Preserve its maximum alongside the reported GPU capacity.

## 5. Decode and score

The official entry point processes the six groups for both dev and test, so
this is a bounded **12-cut** decode, not one cut. It uses the saved batch
checkpoint without averaged-model checkpoint averaging. Its model receives a
whole feature batch despite chunk-size settings; report it only as whole-batch
decode with chunk-constrained configuration, not true incremental decoding.

```bash
python dprnn_zipformer/decode.py \
  --iter 1 --avg 1 --use-averaged-model false --exp-dir "$EXP" \
  --manifest-dir "$FIXTURE/manifests" --bpe-model data/lang_bpe_500/bpe.model \
  --max-duration-valid 3 --max-cuts 1 --num-workers 0 \
  --decoding-method greedy_search 2>&1 | tee "$LOG/decode.txt"
find "$EXP/greedy_search" -type f \( -name 'recogs-*.txt' -o -name 'errs-*.txt' -o -name 'wer-summary-*.txt' \) -print | sort | tee "$LOG/decode-files.txt"
find "$EXP/greedy_search" -name 'wer-summary-*.txt' -type f -exec cat {} \; | tee "$LOG/wer.txt"
```

Inspect the recognition and error files for all expected nonempty groups.
`write_surt_error_stats` uses MeetEval's ORC assignment and reports WER against
the invented text. A missing score, import error, missing checkpoint, out-of-
memory, or lack of real chunk-state support is a failed or partial test—not a
zero WER or a verified recipe. Preserve the logs, BPE hash, full package freeze,
input/fixture/checkpoint hashes, GPU sampling and actual command lines.
