# Kaggle SURT smoke test runbook

**Execution evidence:** a user-supplied Kaggle log dated 2026-11-12 records
checkpoint-1 loading on CUDA, scores for all 12 decode groups, and `Done!`.
The training/checkpoint-inspection logs have not been independently reviewed.
The compact-output wrappers below were checked locally, not rerun on Kaggle.
Use a GPU/Internet session with the required project commit available remotely.
This run generates synthetic tones;
any WER is a plumbing value, not speech-recognition quality. It does not prove
incremental stateful streaming, Persian feasibility, or real-time performance.

The commands below target Linux x86-64, **Python 3.12**, a CUDA 12.1-capable
NVIDIA driver, Torch 2.3.1 and the corresponding k2/kaldifeat wheels. Kaggle's
notebook kernel may use a different Python version: the second cell installs
an isolated Python 3.12 environment without changing the notebook kernel. The
actual image and driver must satisfy the other prerequisites. The matching wheel
names exist in the official k2 and kaldifeat indexes. The supplied decode log
reports Torch 2.3.1+cu121 and Lhotse 1.33.0, but does not independently verify a
clean installation or all training checks. Stop on any failure and save
the logs; do not silently change versions. Core packages are pinned, but the
full transitive environment is recorded rather than locked in advance.


## Compact notebook output

Each cell below keeps its complete stdout/stderr in
`/kaggle/working/surt-smoke-logs/cell-*.log`. The notebook shows a start message
and one success line, or the last 40 log lines on failure. Existing per-command
logs, hashes and checks remain in place. Long installation/training steps may
have no further notebook output until completion; open the full log to inspect
progress. Rerunning a cell overwrites its cell log, so save failed logs first.
To see complete output afterward, open the corresponding file in Kaggle.

## 1. Clone pinned sources

**Supplied image:** `kaggle_env_info.md` reports a Python 3.13.15 notebook
kernel with Torch/torchaudio 2.11.0+cu128 and two T4 GPUs. The stack below is
an isolated legacy recipe, not a replacement for that kernel. Never install
its k2/kaldifeat wheels in the base environment. See
[environment compatibility](KAGGLE_ENVIRONMENT.md). Existing cells below
already recreate/activate `surt-venv-py312` in each shell.

Each code block is a separate Kaggle notebook cell: copy it **including**
`%%bash`, which must be the first line. Each `%%bash` starts a new shell, so
variables, the current directory, and virtual-environment activation do not
persist between cells. Each block below sets what it needs explicitly. Run the
cells in order and stop if any cell fails. Enable GPU and Internet in Kaggle.

The project commit below contains the fixture code; it must be pushed to the
remote before cloning. The local runbook edits do not change that commit until
they too are committed and pushed. If the repositories were already cloned in
the same Kaggle session, this cell reuses them and checks out the pinned commits.

```bash
%%bash

set -euo pipefail
CELL_LOG="/kaggle/working/surt-smoke-logs/cell-01-source.log"
mkdir -p "$(dirname "$CELL_LOG")"
printf 'Running 01-source; full output: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 01-source. Full log: %s\n" "$CELL_LOG"; else printf "FAILED: 01-source (exit %s). Last 40 log lines:\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
export PROJECT_COMMIT="4c426d12dd18548656c4830210e80dee099a41f8"
export ICEFALL_COMMIT="f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1"
export WORK=/kaggle/working
export PROJECT="$WORK/English-Streaming-ASR"
export ICEFALL="$WORK/icefall"
export RECIPE="$ICEFALL/egs/libricss/SURT"
export FIXTURE="$WORK/surt-fixture-v2"
export EXP="$WORK/surt-smoke-exp"
export LOG="$WORK/surt-smoke-logs"
mkdir -p "$LOG"
nvidia-smi | tee "$LOG/gpu-before.txt"
python3 --version
if [ ! -d "$PROJECT/.git" ]; then
  git clone https://github.com/manzourali/English-Streaming-ASR.git "$PROJECT"
fi
git -C "$PROJECT" checkout --detach "$PROJECT_COMMIT"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$PROJECT_COMMIT"
if [ ! -d "$ICEFALL/.git" ]; then
  git clone https://github.com/k2-fsa/icefall.git "$ICEFALL"
fi
git -C "$ICEFALL" checkout --detach "$ICEFALL_COMMIT"
test "$(git -C "$ICEFALL" rev-parse HEAD)" = "$ICEFALL_COMMIT"
printf 'project=%s\nicefall=%s\n' "$PROJECT_COMMIT" "$ICEFALL_COMMIT" | tee "$LOG/commits.txt"
) > "$CELL_LOG" 2>&1
```

## 2. Install the selected GPU stack

Use a Python 3.12 environment isolated from Kaggle's notebook kernel. `uv`
downloads Python 3.12 if it is not available and creates the environment with
pip; its Torch installation does not replace Kaggle's base packages. These
downloads occur in Kaggle, never on the Mac. The previous error showing
`/surt-venv` occurred because `$WORK` was not defined in that cell; the
earlier attempted installs may have affected Kaggle's base Python. A fresh
Kaggle session is recommended before retrying. No files are deleted here.

`k2` and `kaldifeat` require the CUDA 12 NVRTC shared library. Even when the
Python package is installed, Linux may not search its `site-packages/nvidia`
library directory automatically. The cell below installs the CUDA 12.1 NVRTC
package and adds the virtual environment's NVIDIA library directories to
`LD_LIBRARY_PATH` **before** importing either extension. Each subsequent
`%%bash` cell repeats this setup because the shell environment does not persist.
If the import check fails, stop: later cells depend on it.

```bash
%%bash

set -euo pipefail
CELL_LOG="/kaggle/working/surt-smoke-logs/cell-02-environment.log"
mkdir -p "$(dirname "$CELL_LOG")"
printf 'Running 02-environment; full output: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 02-environment. Full log: %s\n" "$CELL_LOG"; else printf "FAILED: 02-environment (exit %s). Last 40 log lines:\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
export WORK=/kaggle/working
export ICEFALL="$WORK/icefall"
export LOG="$WORK/surt-smoke-logs"
export UV_LINK_MODE=copy
unset UV_SYSTEM_PYTHON
mkdir -p "$LOG" "$WORK/uv-bin"
curl -LsSf https://astral.sh/uv/0.8.22/install.sh | \
  env UV_INSTALL_DIR="$WORK/uv-bin" UV_NO_MODIFY_PATH=1 sh
"$WORK/uv-bin/uv" --version
if [ ! -e "$WORK/surt-venv-py312" ]; then
  "$WORK/uv-bin/uv" venv --seed --python 3.12.10 "$WORK/surt-venv-py312"
fi
test -x "$WORK/surt-venv-py312/bin/python"
source "$WORK/surt-venv-py312/bin/activate"
python -c 'import sys; print("environment:", sys.executable, sys.version); assert sys.version_info[:2] == (3, 12)'
python -m pip --version
python -m pip install 'numpy==1.26.4'
python -m pip install 'torch==2.3.1+cu121' 'torchaudio==2.3.1+cu121' \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'nvidia-cuda-nvrtc-cu12==12.1.105'
python -m pip install 'k2==1.24.4.dev20241030+cuda12.1.torch2.3.1' \
  -f https://k2-fsa.github.io/k2/cuda.html
python -m pip install 'kaldifeat==1.25.4.dev20240606+cuda12.1.torch2.3.1' \
  -f https://csukuangfj.github.io/kaldifeat/cuda.html
python -m pip install 'lhotse==1.33.0' 'sentencepiece==0.2.3' \
  'meeteval==0.4.3' 'pypinyin==0.50.0' einops 'tensorboard==2.21.0' 'grpcio==1.85.1' \
  kaldialign typeguard soundfile urllib3
NVIDIA_LIB_DIRS="$(python -c 'import pathlib, sysconfig; p = pathlib.Path(sysconfig.get_path("purelib")); d = sorted((p / "nvidia").glob("*/lib")); nvrtc = p / "nvidia/cuda_nvrtc/lib/libnvrtc.so.12"; assert nvrtc.is_file(), f"Missing {nvrtc}"; print(":".join(map(str, d)))')"
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip freeze | tee "$LOG/pip-freeze.txt"
export PYTHONPATH="$ICEFALL:$ICEFALL/egs/libricss/SURT/dprnn_zipformer"
python - <<'PY' | tee "$LOG/environment.txt"
import platform
import ctypes
print('NVRTC:', ctypes.CDLL('libnvrtc.so.12')._name)
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
) > "$CELL_LOG" 2>&1
```

## 3. BPE and bounded fixture

The English BPE is the existing model referenced by Icefall's stage 10, not a
new tokenizer. Assert its previously observed SHA-256. Do not call `prepare.sh`.
Retry section 2 successfully before this section. The previous failed build may
have created a nonempty `surt-fixture` directory, which the builder refuses to
overwrite. This retry uses `surt-fixture-v2` and leaves the old files untouched.
If the new directory already exists and is nonempty, choose another fresh path
and use it consistently in sections 3, 4, and 5; do not delete it implicitly.

```bash
%%bash

set -euo pipefail
CELL_LOG="/kaggle/working/surt-smoke-logs/cell-03-fixture.log"
mkdir -p "$(dirname "$CELL_LOG")"
printf 'Running 03-fixture; full output: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 03-fixture. Full log: %s\n" "$CELL_LOG"; else printf "FAILED: 03-fixture (exit %s). Last 40 log lines:\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
export WORK=/kaggle/working
export PROJECT="$WORK/English-Streaming-ASR"
export ICEFALL="$WORK/icefall"
export RECIPE="$ICEFALL/egs/libricss/SURT"
export FIXTURE="$WORK/surt-fixture-v2"
export LOG="$WORK/surt-smoke-logs"
source "$WORK/surt-venv-py312/bin/activate"
export PYTHONPATH="$ICEFALL:$ICEFALL/egs/libricss/SURT/dprnn_zipformer"
NVIDIA_LIB_DIRS="$(python -c 'import pathlib, sysconfig; p = pathlib.Path(sysconfig.get_path("purelib")); d = sorted((p / "nvidia").glob("*/lib")); nvrtc = p / "nvidia/cuda_nvrtc/lib/libnvrtc.so.12"; assert nvrtc.is_file(), f"Missing {nvrtc}; rerun section 2"; print(":".join(map(str, d)))')"
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import ctypes; ctypes.CDLL("libnvrtc.so.12"); import torch, k2, kaldifeat; assert torch.cuda.is_available(); print("CUDA imports and GPU preflight OK")'
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
) > "$CELL_LOG" 2>&1
```

Validation must report `status: validated`, one training cut, all six nonempty
decode groups, and the observed empty-group probe. Check these rather than
assuming that the shell status proves every desired property.

## 4. One training batch and checkpoint inspection

### 4a. Use simple samplers for the tiny evaluation fixture

Run this cell **once before the training cell below** (it is safe to rerun).
The recipe's `--bucketing-sampler false` controls training only; validation
and test construct `DynamicBucketingSampler` independently. Validation's ten
buckets cannot be estimated from our six cuts. Decode groups are even smaller.
This smoke-only patch replaces the sampler constructor in `valid_dataloaders`
and `test_dataloaders` with `SimpleCutSampler`, retaining their existing limits
and other arguments. It does not enlarge or regenerate the fixture.

The patch checks the source structure before writing and records the local
Icefall diff. The Icefall commit plus this saved patch describe the executed
source; it is no longer an unmodified upstream checkout. If a structure check
fails, stop and inspect the actual source rather than removing the assertion.

```bash
%%bash

set -euo pipefail
CELL_LOG="/kaggle/working/surt-smoke-logs/cell-04a-sampler.log"
mkdir -p "$(dirname "$CELL_LOG")"
printf 'Running 04a-sampler; full output: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 04a-sampler. Full log: %s\n" "$CELL_LOG"; else printf "FAILED: 04a-sampler (exit %s). Last 40 log lines:\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
export WORK=/kaggle/working
export ICEFALL="$WORK/icefall"
export LOG="$WORK/surt-smoke-logs"
mkdir -p "$LOG"
"$WORK/surt-venv-py312/bin/python" - "$ICEFALL/egs/libricss/SURT/dprnn_zipformer/asr_datamodule.py" <<'PY'
import ast
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
tree = ast.parse(source)
assert any(isinstance(n, ast.ImportFrom) and n.module == 'lhotse.dataset'
           and any(a.name == 'SimpleCutSampler' and a.asname is None for a in n.names)
           for n in ast.walk(tree)), 'Expected SimpleCutSampler import missing'
lines = source.splitlines(keepends=True)
for name in ('valid_dataloaders', 'test_dataloaders'):
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(methods) == 1, f'Unexpected method structure: {name}'
    calls = [n for n in ast.walk(methods[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id in ('DynamicBucketingSampler', 'SimpleCutSampler')]
    assert len(calls) == 1, f'Unexpected sampler structure: {name}'
    call = calls[0]
    assert len(call.args) == 1 and all(k.arg in ('max_duration', 'max_cuts', 'shuffle')
                                      for k in call.keywords), f'Unexpected sampler arguments: {name}'
    node = call.func
    if node.id == 'DynamicBucketingSampler':
        index = node.lineno - 1
        lines[index] = lines[index][:node.col_offset] + 'SimpleCutSampler' + lines[index][node.end_col_offset:]
    print(f'{name}: SimpleCutSampler')
patched = ''.join(lines)
ast.parse(patched)
if patched != source:
    path.write_text(patched)
PY
git -C "$ICEFALL" diff -- egs/libricss/SURT/dprnn_zipformer/asr_datamodule.py \
  | tee "$LOG/smoke-sampler.patch"
) > "$CELL_LOG" 2>&1
```

If training previously stopped while creating the dev loader with
`The number of buckets (10) ... number of cuts (6)`, run cell 4a and retry
the training cell below in the same session. Sections 1–3 need not be rerun.
That traceback occurs before the training loop; it does not show a completed
optimizer update. Keep the existing fixture, environment, and failure log.

### 4b. Run training and inspect the checkpoint

The full and warm-up manifests each contain one cut; the `SimpleCutSampler`
and one epoch are intended to produce one update. The recipe consumes the
warm-up batch during its warm-up period. The assertions below verify the saved
batch counter, optimizer-state step and finite saved model/optimizer tensors.
They do not independently prove that every intermediate gradient was finite.
Record that limitation unless separate gradient instrumentation is approved.

```bash
%%bash

set -euo pipefail
CELL_LOG="/kaggle/working/surt-smoke-logs/cell-04b-training.log"
mkdir -p "$(dirname "$CELL_LOG")"
printf 'Running 04b-training; full output: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 04b-training. Full log: %s\n" "$CELL_LOG"; else printf "FAILED: 04b-training (exit %s). Last 40 log lines:\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
export WORK=/kaggle/working
export ICEFALL="$WORK/icefall"
export RECIPE="$ICEFALL/egs/libricss/SURT"
export FIXTURE="$WORK/surt-fixture-v2"
export EXP="$WORK/surt-smoke-exp"
export LOG="$WORK/surt-smoke-logs"
source "$WORK/surt-venv-py312/bin/activate"
export PYTHONPATH="$ICEFALL:$ICEFALL/egs/libricss/SURT/dprnn_zipformer"
NVIDIA_LIB_DIRS="$(python -c 'import pathlib, sysconfig; p = pathlib.Path(sysconfig.get_path("purelib")); d = sorted((p / "nvidia").glob("*/lib")); nvrtc = p / "nvidia/cuda_nvrtc/lib/libnvrtc.so.12"; assert nvrtc.is_file(), f"Missing {nvrtc}; rerun section 2"; print(":".join(map(str, d)))')"
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
) > "$CELL_LOG" 2>&1
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
%%bash

set -euo pipefail
CELL_LOG="/kaggle/working/surt-smoke-logs/cell-05-decode.log"
mkdir -p "$(dirname "$CELL_LOG")"
printf 'Running 05-decode; full output: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 05-decode. Full log: %s\n" "$CELL_LOG"; else printf "FAILED: 05-decode (exit %s). Last 40 log lines:\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
export WORK=/kaggle/working
export ICEFALL="$WORK/icefall"
export RECIPE="$ICEFALL/egs/libricss/SURT"
export FIXTURE="$WORK/surt-fixture-v2"
export EXP="$WORK/surt-smoke-exp"
export LOG="$WORK/surt-smoke-logs"
source "$WORK/surt-venv-py312/bin/activate"
export PYTHONPATH="$ICEFALL:$ICEFALL/egs/libricss/SURT/dprnn_zipformer"
NVIDIA_LIB_DIRS="$(python -c 'import pathlib, sysconfig; p = pathlib.Path(sysconfig.get_path("purelib")); d = sorted((p / "nvidia").glob("*/lib")); nvrtc = p / "nvidia/cuda_nvrtc/lib/libnvrtc.so.12"; assert nvrtc.is_file(), f"Missing {nvrtc}; rerun section 2"; print(":".join(map(str, d)))')"
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$RECIPE"
python dprnn_zipformer/decode.py \
  --iter 1 --avg 1 --use-averaged-model false --exp-dir "$EXP" \
  --manifest-dir "$FIXTURE/manifests" --bpe-model data/lang_bpe_500/bpe.model \
  --max-duration-valid 3 --max-cuts 1 --num-workers 0 \
  --decoding-method greedy_search 2>&1 | tee "$LOG/decode.txt"
find "$EXP/greedy_search" -type f \( -name 'recogs-*.txt' -o -name 'errs-*.txt' -o -name 'wer-summary-*.txt' \) -print | sort | tee "$LOG/decode-files.txt"
find "$EXP/greedy_search" -name 'wer-summary-*.txt' -type f -exec cat {} \; | tee "$LOG/wer.txt"
) > "$CELL_LOG" 2>&1
```

Inspect the recognition and error files for all expected nonempty groups.
`write_surt_error_stats` uses MeetEval's ORC assignment and reports WER against
the invented text. A missing score, import error, missing checkpoint, out-of-
memory, or lack of real chunk-state support is a failed or partial test—not a
zero WER or a verified recipe. Preserve the logs, BPE hash, full package freeze,
input/fixture/checkpoint hashes, GPU sampling and actual command lines.


## 6. Archive and next gate

[Archive all logs as a single downloadable file](KAGGLE_SURT_ARTIFACTS_FA.md).
Keep a small experiment summary, pins, patches and hashes in Git; keep model
weights, data and environments outside ordinary Git history. The observed
synthetic-smoke evidence is recorded in
[the experiment note](experiments/surt_smoke_2026-11-12.md).

The next gate is [official pretrained SURT Base on real English speech](prompts/PERSIAN_SURT_NEXT_STEP.md),
followed by mini Persian transfer/update and genuine chunk-state/resource
checks. This smoke alone does not complete [Persian feasibility](PERSIAN_SURT_ROADMAP_FA.md).
