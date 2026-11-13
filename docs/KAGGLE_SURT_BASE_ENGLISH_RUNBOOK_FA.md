# گام ۱A — baseline واقعی انگلیسی با SURT Base در Kaggle

وضعیت ۲۰۲۶-۱۱-۱۳: **آماده‌سازی و تست محلی؛ اجرای واقعی Kaggle هنوز انجام نشده است.**
این runbook مستقل است؛ سلول‌های smoke قبلی لازم نیست اجرا شوند. GPU و Internet را فعال کنید.
فقط دو کلیپ واقعی clean و یک مخلوط دوگوینده decode می‌شوند. هیچ آموزش، corpus کامل،
`prepare.sh` یا exporter اختصاصی پروژه لازم نیست. خروجی **pretrained whole-batch greedy decode** است؛
chunk length به معنی تغذیهٔ افزایشی با state نیست و این اجرا gate استریم را نمی‌بندد.

## به‌روزرسانی کد از Git و حفظ دانلودهای موجود

ابتدا تغییرات محلی موردنظر را خودتان commit و push کنید؛ سپس روی Mac با `git rev-parse HEAD`
SHA کامل همان commit را بگیرید و در `PROJECT_COMMIT` سلول ۱ قرار دهید. مقدار نمونه را اجرا نکنید.
سلول ۱ checkout موجود `/kaggle/working/English-Streaming-ASR` را با fetch و checkout همان SHA
به‌روز می‌کند؛ اگر checkout نباشد clone می‌کند. شاخهٔ remote جابه‌جا نمی‌شود و commit محلی ساخته نمی‌شود.
تغییر tracked محلی، remote نامنتظر یا برخورد فایل‌ها باعث توقف می‌شود؛ هیچ `reset --hard`،
`git clean` یا پاک‌کردن پوشه‌ای در کار نیست. SHA اجراشده در `logs/project-commit.txt` ثبت می‌شود.

این به‌روزرسانی را در همان session فعلی انجام دهید. محیط `/kaggle/working/surt-venv-py312`،
دانلودهای smoke و وزن‌های ۱A در `/kaggle/working/surt-base-1a/assets/model` حفظ می‌شوند؛
Icefall این baseline در `/kaggle/working/icefall-surt-base` جداست و revision قبلی را دارد.
این تضمین مربوط به عملیات سلول ۱ است؛ ذخیرهٔ دائمی فایل‌های working یا بازیابی پس از تعویض session
جزء عملیات Git نیست. ورودی‌های متصل در `/kaggle/input` فقط خوانده می‌شوند.

پنج سلول این راهنما را در بخش مستقلی از notebook قرار دهید یا جای سلول‌های smoke قبلی بگذارید؛
برای اولین اجرای ۱A فقط همین سلول‌های **۱ تا ۵** را به ترتیب اجرا کنید. سلول‌های قدیمی smoke،
optimizer step، train و prepare کامل را دوباره اجرا نکنید. «Run All» فقط برای notebook شامل همین
پنج سلول و بدون سلول اجرایی قدیمی مناسب است؛ اجرای دستی به ترتیب، توقف در خطا را روشن‌تر می‌کند.
برای دریافت صرفاً تغییر کد، اجرای سلول ۱ کافی است. سلول ۲ محیط موجود را بازاستفاده و بررسی می‌کند.
سلول ۳ روی assets موجود و سلول ۴ روی نتیجهٔ قبلی متوقف می‌شوند تا شواهد overwrite نشود؛
وجود این پوشه‌ها دلیل حذف وزن‌ها یا شروع session تازه نیست. بعد از خطا سلول ۵ را برای حفظ شواهد اجرا کنید.

## منابع، وزن و قرارداد مدل

منبع‌های اصلی و snapshot metadata در `configs/surt_base/source-evidence.json` و pinهای اجرایی در
`configs/surt_base/pins.json` ثبت شده‌اند. metadata کوچک از منبع اصلی در تاریخ بالا خوانده شد؛
وزن ۱۰۶٬۹۴۴٬۷۹۶ بایتی فقط در Kaggle گرفته می‌شود و SHA آن با LFS metadata رسمی مقایسه می‌شود.

| مورد | مقدار ثابت |
| --- | --- |
| Icefall | `f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1`، همان smoke موفق |
| مخزن رسمی HF | `desh2608/icefall-surt-libricss-dprnn-zipformer` |
| revision واقعی HF | `5413ea7aac8d667d36127260ce5fa54253d74270` |
| وزن این اجرا | `exp/surt_base/pretrained.pt`؛ SHA256 `919bb9454526fb9ad963eb5a4168940a43c3b8331017a1c9ade477197c7072d4` |
| وزن‌های متفاوت | `exp/zipformer_base.pt` آغازین تک‌گوینده؛ `exp/surt_base_adapt/pretrained.pt` تطبیق LibriCSS؛ `exp/surt_large/pretrained.pt` مدل Large؛ استفاده نمی‌شوند |
| tokenizer همراه وزن | `data/lang_bpe_500/{bpe.model,tokens.txt}` از همان revision؛ واژگان ۵۰۰، blank=۰، unk=۲؛ `#0/#1` disambiguation خارج vocabulary |
| mask encoder | DPRNN، ۴ بلوک، dim=۲۵۶، segment=۳۲، randomization خاموش (default recipe) |
| recognition | Zipformer layers=`2,2,2,2,2`، dim=۲۵۶، FF=۷۶۸، attention/unmasked=۱۹۲، heads=۸، factors=`1,2,4,8,2`، kernels=۳۱ |
| branch tying | `--use-joint-encoder-layer lstm`، LSTM مشترک دو شاخه طبق `get_joint_encoder_layer`؛ decoder/joiner=۵۱۲، context=۲ |
| context | decode chunk=۳۲ frame پیش از subsampling، left chunks=۴؛ دو شاخه بدون تضمین هویت ثابت گوینده |

BPE با تک‌تک ۵۰۰ token تطبیق داده می‌شود. قبل از model load، import واقعی recipe، NVRTC،
k2/kaldifeat و CUDA و عملیات کوچک GPU و پل `torch.from_numpy(...).numpy()` آزموده می‌شود.
مدل ساخته‌شده با تمام کلیدها و shapeهای checkpoint مقایسه و با `strict=True` بارگذاری می‌شود؛
تطبیق واقعی checkpoint تا دریافت نتیجهٔ Kaggle **تأیید نشده است**. در ناسازگاری،
`checkpoint-contract.json` missing/unexpected/shapes را ثبت می‌کند؛ strict را خاموش نکنید.

`pretrained.py` upstream در این revision، `strict=False` دارد و تابع `_group_channels` را در
گزارش نهایی صدا نمی‌زند؛ `zip(sound_files, hyps)` رشته‌های channel-major را اشتباه گزارش می‌کند.
`docs/patches/surt-base-pretrained.patch` فقط strict، diagnostics، جداسازی warmup، inference_mode و
برگشت گروه‌های `[file][branch]` را اصلاح می‌کند؛ محاسبهٔ mask/encoder/search recipe حفظ می‌شود.
patch قبلی SimpleCutSampler عیناً از آرشیو موفق استخراج و در `docs/patches/surt-simple-sampler.patch`
حفظ شده است؛ این entry point اصلاً dataloader ندارد، ولی همان patch با diff/hash ثبت می‌شود.
هش سورس اصلی، patch و سورس نهایی بررسی/آرشیو می‌شود.

## صوت واقعی و شرایط استفاده

اولویت: یک `surt-english-input.json` در ورودی متصل‌شده، شامل دقیقاً دو رکورد با کلیدهای
`id`, `speaker_id`, `text`, `source`, `license`, `audio` (مسیر نسبی به manifest).
دو `speaker_id` متفاوت، متن کامل واقعی، منبع و مجوز مجاز لازم است. صوت هر رکورد باید mono،
۱۶kHz و ۱ تا ۱۲ ثانیه باشد؛ تبدیل sample rate یا بریدن متن/صوت پنهانی انجام نمی‌شود.
ورودی واقعی را گوش کنید و تطبیق مرجع را انسانی کنترل کنید؛ metadata به‌تنهایی واقعی‌بودن را ثابت نمی‌کند.
مسیرها باید داخل پوشهٔ همان dataset باشند. ورودی‌ها read-only هستند و خروجی فقط در working نوشته می‌شود.

اگر چنین manifestای متصل نیست، فقط دو فایل منفرد زیر از viewer رسمی HF دانلود می‌شود؛
از `datasets.load_dataset` و parquet/corpus کامل استفاده نمی‌شود. هر دانلود صوت حداکثر ۲ MiB است.
revision داده `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` و id/speaker/text هر row کنترل می‌شود؛
URL امضاشده در زمان اجرا تازه گرفته می‌شود و باید همان revision را در مسیر داشته باشد.
اگر viewer تغییر کرد یا unavailable بود، بدون جایگزینی خاموش متوقف شوید؛ همان دو کلیپ یا دو کلیپ
مجاز را با manifest متصل کنید. زمان دقیق و هش waveform در `inputs.json` ثبت می‌شود.

| LibriSpeech test-clean | گوینده / row | متن مرجع کامل |
| --- | --- | --- |
| `6930-75918-0000` | 6930 / 0 | CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS |
| `1320-122617-0022` | 1320 / 100 | THE DELAWARES ARE CHILDREN OF THE TORTOISE AND THEY OUTSTRIP THE DEER |

منبع: OpenSLR 12، LibriSpeech از Panayotov و همکاران (۲۰۱۵)، مبتنی بر LibriVox؛ CC BY 4.0.
ارجاع/شرایط را همراه نتایج نگه دارید. metadata HF و صفحهٔ OpenSLR مجوز را اعلام می‌کنند.
این کلیپ‌های LibriSpeech برای sanity زیرساخت انتخاب شده‌اند؛ استقلال آن‌ها از همهٔ داده‌های
آموزش/شبیه‌سازی pretrained تضمین نشده و معیار تعمیم held-out نیستند.

مخلوط: نمونهٔ اول در t=۰ و دوم در t=۰٫۵s، gain هر کدام ۰٫۵، کل هر دو جمله حفظ می‌شود؛
هم‌پوشانی واقعی از تعداد sample اندازه‌گیری می‌شود. مجموع مدت دو clean و mixture حداکثر ۳۶٫۵ ثانیه است.
هیچ نمونه‌ای با شش نام recipe تکثیر نمی‌شود؛ labelهای 0L/0S/OV10… نداریم و این **ارزیابی شش شرط LibriCSS نیست**.
word alignment ادعا نشده؛ تنها timing کل جمله موجود است.

## feature و اندازه‌گیری

از خود `dprnn_zipformer/pretrained.py` استفاده می‌شود: `soundfile.read(dtype=float32)`،
بدون ضرب ۳۲۷۶۸؛ kaldifeat Fbank روی GPU، ۸۰ bin، ۱۶kHz، dither=۰، snip_edges=False،
high_freq=-۴۰۰؛ framing پیش‌فرض pinned kaldifeat، ۲۵ms window و ۱۰ms shift (در runtime assert می‌شوند).
Povey/preemphasis و بقیهٔ defaults در `features.options` ثبت می‌شوند. feature هر صوت `[T,80]`،
batch `[3,Tmax,80]` و padding=`log(1e-10)`؛ mask خروجی `[B,T,80*2]`، encoder batch برابر `2*B`.
log-Mel عمومی پروژه وارد این مسیر نمی‌شود؛ parity متفاوتی ادعا نمی‌شود.

`eval()` و `torch.inference_mode()`، GPU صفر و فقط **یک model load** در یک فرایند اجرا می‌شود.
یک pass warmup کامل روی همان سه ورودی، سپس یک pass اندازه‌گیری‌شده با همان مدل/feature انجام می‌شود.
CUDA قبل و بعد timer همگام می‌شود. load شامل ساخت CPU، hash/finite/strict checks، خواندن وزن و انتقال GPU است؛
feature جدا، warmup جدا، mask+encoder+greedy decode جدا و wall کل فرایند جدا ثبت می‌شوند.
RTF بر مجموع ثانیهٔ سه ورودی است و زمان batch/warm cache است؛ latency استریم یا end-to-end شامل IO نیست.
peak allocated/reserved از reset پس از warmup است؛ reserved می‌تواند cache warmup را داشته باشد.
`gpu-memory.csv` نمونه‌برداری یک‌ثانیه‌ای کل دستگاه است و peak دقیق فرایند نیست؛ model-load memory
تنها در این نمونه‌برداری پوشش دارد. یک اجرای کوتاه برای تخمین هزینهٔ آموزش کافی نیست.

متن خام هر دو branch و reference، assignment و S/D/I/N هر شاخه و کل در JSON/TXT ثبت می‌شود.
امتیاز **دو-stream permutation-invariant WER روی جمله‌های محدود** است؛ برای clean مرجع شاخهٔ دوم خالی
است و hallucination آن insertion حساب می‌شود. normalization: uppercase، کلمات A–Z و apostrophe داخلی؛
بدون LM، punctuation حذف می‌شود. این PI-WER را ORC-WER رسمی جلسات LibriCSS یا ارزیابی فارسی ننامید.

## سلول‌ها؛ به ترتیب ۱ تا ۵

هر code block یک سلول جداست و `%%bash` باید اولین خط باشد. stdout/stderr کامل و command trace
در `surt-base-1a/logs/cell-*.log` ثبت می‌شود؛ UI فقط START و OK یا انتهای ۴۰ خط خطا را نشان می‌دهد.
`set -euo pipefail` و exit code حفظ می‌شوند؛ خطا را با ادامهٔ سلول بعدی نپوشانید.
پس از شکست سلول‌های ۱ تا ۴، سلول ۵ را برای بستهٔ تشخیصی اجرا کنید؛ archive نشانهٔ عبور gate نیست.

محیط همان Python 3.12.10 ایزولهٔ `surt-venv-py312`، uv 0.8.22، Torch/torchaudio 2.3.1+cu121،
k2 و kaldifeat wheelهای قبلی، Lhotse 1.33.0، sentencepiece 0.2.3 و meeteval 0.4.3 است.
محیط موفق موجود اولویت دارد و upgrade نمی‌شود. `smoke-observed-freeze.txt` از آرشیو کاربر استخراج شده؛
`requirements-kaggle.txt` همان freeze با دو تغییر **NumPy 2.5.4 به 1.26.4 و SciPy 1.18.1 به 1.13.1**
است. مسیر pretrained پل NumPy/Torch را مصرف می‌کند؛ pin NumPy از انتخاب اولیهٔ runbook موفق می‌آید.
لاگ Kaggle در ۲۰۲۶-۱۱-۱۳ نشان داد SciPy 1.18.1 به NumPy >=2 نیاز دارد و `pip check` متوقف شد؛
SciPy 1.13.1 با Python 3.12 و NumPy 1.26.4 سازگار است و wheel آمادهٔ Linux x86_64 دارد.
منابع نسخهٔ اصلاحی: `https://docs.scipy.org/doc/scipy/dev/toolchain.html` و
`https://pypi.org/project/scipy/1.13.1/`. freeze مشاهده‌شده به‌عنوان شاهد بدون تغییر نگه داشته شده است.
برای محیط موجود فقط `requirements-compat.txt` با `--no-deps --only-binary=:all:` نصب می‌شود؛
Torch، k2، kaldifeat و CUDA دوباره نصب نمی‌شوند. before/after، `pip check` و import/عملیات کوچک
NumPy/SciPy/Torch قبل از model load ثبت و بررسی می‌شوند؛ موفقیت runtime هنوز نیازمند اجرای Kaggle است.
پس از دریافت این اصلاح از Git، سلول ۲ را در همان session دوباره اجرا کنید؛ پس از OK به ۳، ۴ و ۵ بروید.
اگر فقط سلول ۲ قبلاً اجرا شده، assets هنوز ساخته نشده‌اند؛ پوشه‌ای را حذف نکنید.
Mac هیچ GPU stack نصب نمی‌کند. freeze به‌تنهایی lock دارای wheel hash نیست؛ فایل‌های واقعی و freeze نهایی ثبت می‌شوند.

### سلول 1 — 01-source

```bash
%%bash
set -euo pipefail
LOG_DIR=/kaggle/working/surt-base-1a/logs
mkdir -p "$LOG_DIR"
CELL_LOG="$LOG_DIR/cell-01-source-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 01-source; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 01-source\n"; else printf "FAILED: 01-source (exit %s)\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
export WORK=/kaggle/working
export PROJECT="$WORK/English-Streaming-ASR"
export ICEFALL="$WORK/icefall-surt-base"
# After committing and pushing locally, paste the full output of: git rev-parse HEAD
export PROJECT_COMMIT="PASTE_FULL_PUSHED_COMMIT_SHA_HERE"
[[ "$PROJECT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'Set PROJECT_COMMIT to the pushed 40-character SHA'; exit 1; }
PROJECT_REMOTE=https://github.com/manzourali/English-Streaming-ASR.git
if [ -d "$PROJECT/.git" ]; then
  test "$(git -C "$PROJECT" remote get-url origin)" = "$PROJECT_REMOTE"
  git -C "$PROJECT" diff --quiet || { echo 'Tracked local edits: preserve/review them before updating'; exit 1; }
  git -C "$PROJECT" diff --cached --quiet || { echo 'Staged local edits: preserve/review them before updating'; exit 1; }
else
  test ! -e "$PROJECT" || { echo 'Existing project directory is not a Git checkout; preserve/review it'; exit 1; }
  git clone --filter=blob:none --no-checkout "$PROJECT_REMOTE" "$PROJECT"
fi
git -C "$PROJECT" fetch --no-tags origin "$PROJECT_COMMIT"
git -C "$PROJECT" checkout --no-overwrite-ignore --detach "$PROJECT_COMMIT"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$PROJECT_COMMIT"
printf '%s\n' "$PROJECT_COMMIT" > "$LOG_DIR/project-commit.txt"
test -f "$PROJECT/scripts/surt_base/env.sh"
ICEFALL_COMMIT=f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1
if [ ! -d "$ICEFALL/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/k2-fsa/icefall.git "$ICEFALL"
  git -C "$ICEFALL" checkout --detach "$ICEFALL_COMMIT"
fi
test "$(git -C "$ICEFALL" rev-parse HEAD)" = "$ICEFALL_COMMIT"
python3 "$PROJECT/scripts/surt_base/source.py" "$ICEFALL" /kaggle/working/surt-base-1a/logs
nvidia-smi
) > "$CELL_LOG" 2>&1
```

### سلول 2 — 02-environment

```bash
%%bash
set -euo pipefail
LOG_DIR=/kaggle/working/surt-base-1a/logs
mkdir -p "$LOG_DIR"
CELL_LOG="$LOG_DIR/cell-02-environment-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 02-environment; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 02-environment\n"; else printf "FAILED: 02-environment (exit %s)\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
export WORK=/kaggle/working
export PROJECT="$WORK/English-Streaming-ASR"
export UV_LINK_MODE=copy
unset UV_SYSTEM_PYTHON
if [ ! -x "$WORK/surt-venv-py312/bin/python" ]; then
  mkdir -p "$WORK/uv-bin"
  curl -LsSf https://astral.sh/uv/0.8.22/install.sh | env UV_INSTALL_DIR="$WORK/uv-bin" UV_NO_MODIFY_PATH=1 sh
  "$WORK/uv-bin/uv" venv --seed --python 3.12.10 "$WORK/surt-venv-py312"
  source "$WORK/surt-venv-py312/bin/activate"
  python -m pip install -r "$PROJECT/configs/surt_base/requirements-kaggle.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -f https://k2-fsa.github.io/k2/cuda.html \
    -f https://csukuangfj.github.io/kaldifeat/cuda.html
else
  source "$WORK/surt-venv-py312/bin/activate"
  python -m pip freeze > /kaggle/working/surt-base-1a/logs/environment-before.txt
  # Repair only the NumPy/Torch bridge and its SciPy dependency.
  python -m pip install --no-deps --only-binary=:all: \
    -r "$PROJECT/configs/surt_base/requirements-compat.txt"
fi
source "$PROJECT/scripts/surt_base/env.sh"
python -m pip freeze > "$BASE/logs/pip-freeze.txt"
python -m pip check
python "$PROJECT/scripts/surt_base/runtime.py" 2>&1 | tee "$BASE/logs/preflight.txt"
) > "$CELL_LOG" 2>&1
```

### سلول 3 — 03-assets

```bash
%%bash
set -euo pipefail
LOG_DIR=/kaggle/working/surt-base-1a/logs
mkdir -p "$LOG_DIR"
CELL_LOG="$LOG_DIR/cell-03-assets-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 03-assets; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 03-assets\n"; else printf "FAILED: 03-assets (exit %s)\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_base/env.sh
python "$PROJECT/scripts/surt_base/runtime.py"
python - <<'DATA'
from pathlib import Path
import os, subprocess, sys
manifests = sorted(Path('/kaggle/input').rglob('surt-english-input.json'))
assert len(manifests) <= 1, f'Ambiguous attached input: {manifests}'
argv = [sys.executable, str(Path(os.environ['PROJECT'])/'scripts/surt_base/prepare.py'),
        str(Path(os.environ['BASE'])/'assets')]
if manifests:
    argv += ['--attached', str(manifests[0])]
print('prepare argv', argv, flush=True)
subprocess.run(argv, check=True)
DATA
) > "$CELL_LOG" 2>&1
```

### سلول 4 — 04-decode

```bash
%%bash
set -euo pipefail
LOG_DIR=/kaggle/working/surt-base-1a/logs
mkdir -p "$LOG_DIR"
CELL_LOG="$LOG_DIR/cell-04-decode-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 04-decode; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 04-decode\n"; else printf "FAILED: 04-decode (exit %s)\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_base/env.sh
python "$PROJECT/scripts/surt_base/runtime.py"
nvidia-smi --id=0 --query-gpu=timestamp,uuid,memory.used --format=csv -l 1 > "$BASE/logs/gpu-memory.csv" 2>&1 &
GPU_MONITOR=$!
trap 'kill "$GPU_MONITOR" 2>/dev/null || true; wait "$GPU_MONITOR" 2>/dev/null || true' EXIT
python "$PROJECT/scripts/surt_base/run.py"
test -s "$BASE/results/result.json"
test -s "$BASE/results/transcripts.txt"
) > "$CELL_LOG" 2>&1
```

### سلول 5 — 05-archive

```bash
%%bash
set -euo pipefail
LOG_DIR=/kaggle/working/surt-base-1a/logs
mkdir -p "$LOG_DIR"
CELL_LOG="$LOG_DIR/cell-05-archive-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 05-archive; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then printf "OK: 05-archive\n"; else printf "FAILED: 05-archive (exit %s)\n" "$status"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
export PROJECT=/kaggle/working/English-Streaming-ASR
export BASE=/kaggle/working/surt-base-1a
export ICEFALL=/kaggle/working/icefall-surt-base
if [ -x /kaggle/working/surt-venv-py312/bin/python ]; then
  /kaggle/working/surt-venv-py312/bin/python -m pip freeze > "$BASE/logs/pip-freeze-final.txt"
fi
python3 "$PROJECT/scripts/surt_base/archive.py" "$BASE" "$ICEFALL"
) > "$CELL_LOG" 2>&1
```

## بازبینی و معیار عبور

سلول ۵ با Python پایه و بدون import وابستگی GPU حتی در شکست محیط اجرا شود. فایل کوچک `/kaggle/working/surt-base-1a-review.tar.gz` را برگردانید؛
وزن/صوت/venv در آن نیست. بسته شامل command argv، transcripts، result/metrics، قرارداد checkpoint،
input timing/provenance، محیط قبل/بعد، GPU sampling، stdout/stderr، source pins، bundle provenance،
diffها و هش فایل‌های واقعی است. JSONهای row شامل URL موقت‌اند؛ این URL تضمین ماندگاری داده نیست.
لاگ خود سلول archive تا لحظهٔ snapshot در بسته است؛ فایل کامل آن در working باقی می‌ماند.
صوت clean/mixture برای گوش‌دادن در `surt-base-1a/assets/*.wav` می‌ماند؛ برای ورودی خصوصی آن را عمومی نکنید.

برای عبور ۱A باید hash رسمی، token parity، strict keys/shapes و finite بودن وزن موفق باشند؛
هر سه صوت واقعی از feature تا دو hypothesis/score برسند و متن‌ها انسانی با مرجع/صوت بازرسی شوند.
`real_decode_completed` فقط اتمام فنی است؛ WER بالا/متن بی‌معنا را پنهان نکنید و کیفیت را از پیش تضمین نکنید.
تا دریافت شواهد Kaggle و بازبینی، ۱A **pending** است. در shape mismatch، فایل قرارداد و فرمان کامل
نشان می‌دهند کدام لایه اختلاف دارد: ابتدا Base در برابر Large، تعداد DPRNN/Zipformer و joint LSTM را
با همین revision و مسیر hashشده کنترل کنید؛ `strict=False` یا تعویض tokenizer راه‌حل مجاز نیست.
در خطای data viewer از ورودی متصل‌شدهٔ واقعی استفاده کنید؛ در OOM لاگ را حفظ کنید و این batch محدود را
پیش از تغییر آزمون بررسی کنید؛ مدل را در حلقهٔ نمونه‌ها مجدداً load نکنید.

پس از عبور فقط گام ۱B مطرح است: mini فارسی واقعی و مجاز، split/normalization و BPE فقط train،
انتقال معنایی vocabulary، source-feature هم‌تراز و یک forward/backward/update با save/reload.
این runbook آن را اجرا یا با متن/صوت مصنوعی جایگزین نمی‌کند و آموزش طولانی شروع نمی‌شود.
