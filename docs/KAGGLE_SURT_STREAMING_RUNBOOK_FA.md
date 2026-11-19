# گام 1C — ممیزی استریم، نمونهٔ مهندسی اجزا و منابع در Kaggle

تاریخ آماده‌سازی: ۲۰۲۶-۱۱-۱۹. **1B پذیرفته شده؛ gate استریم 1C هنوز باز است.**
[شواهد پذیرش 1B](experiments/surt_mini_1b_review_2026-11-19.md).
این کد محلی نوشته شده و روی Kaggle اجرا نشده است. تست‌های نوشته‌شده نیز اجرا نشده‌اند.

هدف 1C این است که بفهمیم مدل می‌تواند پیش از پایان فایل، با حافظهٔ محدود و زمینهٔ آیندهٔ
مشخص کار کند و هزینهٔ آن روی سخت‌افزار در دسترس چقدر است. `decode_chunk_len=32` به‌تنهایی
این موضوع را ثابت نمی‌کند. کیفیت فارسی و فایدهٔ FastMSS آزمون‌های بعدی‌اند.

## دامنهٔ این اجرای 1C و نتیجهٔ مورد انتظار

این اجرا **بخش A از 1C** است: سورس/API واقعیِ نصب‌شده را ثبت می‌کند، یک front-end قطعه‌ای
با همان قرارداد feature را می‌آزماید، state هستهٔ LSTM اتصال شاخه‌ها را در صورت قابل بازرسی‌بودن
بررسی می‌کند و منابع whole-utterance و update واقعی را اندازه می‌گیرد.

در ممیزی قبلی، adapter کاملِ stateful برای DPRNN/Zipformer/branch tying موجود نبود و wrapper
عمومی مخزن هم feature و قرارداد LSTM متفاوت داشت. از آن wrapper استفاده نمی‌کنیم.
بازیابی سورس upstream از شبکه در محیط آماده‌سازی این تغییر موفق نشد؛ به همین دلیل runner
نسخهٔ دقیق کلاس‌ها و وابستگی‌های واردشده را از checkout کگل در آرشیو می‌گذارد. امضای متدها
صرفاً موجودی API است؛ بدون بررسی، آرگومان state آن‌ها حدس زده یا فراخوانی نمی‌شود.

موفقیت سلول‌ها باید `component_audit_and_resource_probe_completed` تولید کند، اما **به‌عمد
`gate_1c=OPEN` باقی می‌ماند**. بخش B پس از بازبینی همین سورس/API، adapterهای واقعی مدل را
متصل و parity کامل را اجرا می‌کند. موفقیت feature یا یک LSTM را موفقیت استریم کل SURT ننامید.
تغییر inference به پنجره‌های مستقل و reset کردن مدل در هر پنجره، جایگزین این gate نیست.

بودجه: یک GPU، FP32، حداکثر ۹۰۰ ثانیه؛ ۲ warmup و ۵ اندازه‌گیری inference؛
۲ warmup update و ۵ update اندازه‌گیری‌شده، یعنی **مجموعاً ۷ update** روی یک مخلوط train.
مدل از checkpoint یک-step قبلی شروع می‌شود ولی optimizerِ probe یک Adam تازه است؛
resume آموزش ادعا نمی‌شود. تغییرها فقط در RAM/GPU همان process هستند و checkpoint جدیدی
ذخیره نمی‌شود. دادهٔ validation/test برای تنظیم یا resource probe مصرف نمی‌شود.

front-end صوت float32 را در ۳۲۷۶۸ ضرب نمی‌کند؛ همان Kaldi 80-bin، 16kHz، 25/10ms،
Povey، dither=0، snip_edges=False و high_freq=-400 را نگه می‌دارد. برای هر packet فقط
buffer محدود شامل هم‌پوشانی بازحساب می‌شود. خروجی با extractor مرجع در context یکسان،
packetهای ۱۶۰۰ و نامنظم، طول‌های مرزی، reset و EOF مقایسه می‌شود. tolerance ثابت:
`atol=2e-4`, `rtol=1e-4`. مرجع CPU با feature ذخیره‌شدهٔ 1B نیز مقایسه می‌شود؛ اختلاف
توقف می‌دهد و با تغییر خودکار tolerance پنهان نمی‌شود.

stress برابر ۱۲۰ ثانیه تکرار همان train mixture است؛ فقط آزمون محدودماندن buffer feature
است، نه گفت‌وگوی بلند واقعی و نه اثبات حافظهٔ محدود کل مدل. هستهٔ LSTM نیز با state واقعی
خودش و محور زمانی مشاهده‌شده مقایسه می‌شود؛ reshape/projection/tail کامل wrapper هنوز
adapter لازم دارد. هیچ داده، مدل یا corpus تازه‌ای دانلود نمی‌شود.

## فایل‌های قبلی و session جدید

| دارایی | نیاز در این اجرای 1C |
| --- | --- |
| کل `mini-20261119-d/` شامل checkpoint، tokenizer، data، features و JSONها | لازم؛ ورودی اصلی |
| آرشیو کوچک `mini-20261119-d-review.tar.gz` | برای ممیزی مفید؛ برای بازیابی اجرا کافی نیست |
| محیط Python 3.12.10 و کتابخانه‌های GPU پین‌شده | لازم؛ موجود بازاستفاده می‌شود، مفقود بازسازی می‌شود |
| Icefall با revision و دو patch ثبت‌شده | لازم؛ سلول ۱ checkout را آماده می‌کند |
| وزن و صوت 1A | برای این runner لازم نیست؛ برای provenance و انتقال آینده حفظ شوند |
| فایل‌های فاز صفر | اجرای مجدد لازم نیست؛ آرشیو شواهد قبلی را حفظ کنید |
| download-cache صوت 1B | این runner آن را مصرف نمی‌کند؛ حذف هم نمی‌کند |

پوشه‌ای که روی Mac دارید، `outputs/surt-mini-1b/mini-20261119-d/`، ورودی درست است.
آن را با تمام زیرپوشه‌ها به‌عنوان **ورودی خصوصی Kaggle** متصل کنید؛ در این راهنما نام کاربری
یا slug خاصی فرض نشده است. در session تازه، مسیر واقعی پوشهٔ متصل‌شده در `/kaggle/input/...`
را در `MINI_INPUT` سلول ۳ قرار دهید. این مرحلهٔ اتصال را خود کاربر انجام می‌دهد؛ راهنما
داده‌ای منتشر یا upload خودکار نمی‌کند.

اسکریپت ابتدا hashهای ثبت‌شدهٔ checkpoint/tokenizer/metadata و صوت را بررسی می‌کند؛ سپس
اگر مسیر اصلی وجود نداشت، آن را به این مسیر **کپی** می‌کند:

```text
/kaggle/working/surt-mini-1b/runs/mini-20261119-d
```

این مسیر لازم است چون manifestهای Lhotse مسیر مطلق دارند. پوشهٔ موجود merge/overwrite
نمی‌شود. اگر خراب/ناقص باشد، شواهد را حفظ و بررسی کنید؛ فایل‌ها را برای عبور از کنترل پاک نکنید.
اگر session موفق قبلی هنوز موجود است، upload یا copy دوباره لازم نیست.

وزن رسمی 1A برابر **۱۰۶٬۹۴۴٬۷۹۶ بایت** است؛ checkpoint فعلی 1B حدود ۲۸۳MiB است
(optimizer هم دارد). بنابراین حدود ۷GB دریافت قبلی، اندازهٔ این وزن به‌تنهایی نیست؛ نصب
Torch، بسته‌های NVIDIA/CUDA و سایر وابستگی‌ها می‌تواند سهم عمده داشته باشد. سهم دقیق
بدون گزارش دانلود همان session قابل تعیین نیست. Git و آرشیو review محیط چندگیگابایتی را
برنمی‌گردانند. راه کاهش دانلود تکراری، wheelhouse خصوصی اختیاری در انتهای راهنماست.

## اجرای پنج سلول

فقط این پنج سلول را به ترتیب اجرا کنید. سلول‌های قدیمی smoke/1A/1B را Run All نکنید.
هر block یک سلول مستقل است و `%%bash` باید اولین خط باشد. کد ابتدا روی Mac توسط کاربر
commit/push می‌شود. در سلول اول SHA کامل همان commit را قرار دهید؛ عامل push نمی‌کند.
بعد از شکست، فقط سلول ۵ را برای جمع‌آوری شواهد اجرا کنید. برای تکرار آزمایش، ID تازه لازم است.

### سلول ۱ — کد push‌شده و سورس پین‌شده

```bash
%%bash
set -euo pipefail
STREAM=/kaggle/working/surt-stream-1c
RUN_ID=stream-20261119-a
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
STREAM_RUN="$STREAM/runs/$RUN_ID"
mkdir -p "$STREAM_RUN/logs"
CELL_LOG="$STREAM_RUN/logs/cell-01-source-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 01-source; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 01-source"; else echo "FAILED: 01-source ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
PROJECT=/kaggle/working/English-Streaming-ASR
ICEFALL=/kaggle/working/icefall-surt-base
PROJECT_COMMIT=PASTE_FULL_PUSHED_COMMIT_SHA_HERE
[[ "$PROJECT_COMMIT" =~ ^[0-9a-f]{40}$ ]]
PROJECT_REMOTE=https://github.com/manzourali/English-Streaming-ASR.git
if [ -d "$PROJECT/.git" ]; then
  test "$(git -C "$PROJECT" remote get-url origin)" = "$PROJECT_REMOTE"
  git -C "$PROJECT" diff --quiet
  git -C "$PROJECT" diff --cached --quiet
else
  test ! -e "$PROJECT"
  git clone --filter=blob:none --no-checkout "$PROJECT_REMOTE" "$PROJECT"
fi
git -C "$PROJECT" fetch --no-tags origin "$PROJECT_COMMIT"
git -C "$PROJECT" checkout --no-overwrite-ignore --detach "$PROJECT_COMMIT"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$PROJECT_COMMIT"
printf '%s\n' "$PROJECT_COMMIT" > "$STREAM_RUN/logs/project-commit-$(date -u +%Y%m%dT%H%M%S)-$$.txt"
ICEFALL_COMMIT=f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1
if [ ! -d "$ICEFALL/.git" ]; then
  test ! -e "$ICEFALL"
  git clone --filter=blob:none --no-checkout https://github.com/k2-fsa/icefall.git "$ICEFALL"
  git -C "$ICEFALL" checkout --detach "$ICEFALL_COMMIT"
fi
test "$(git -C "$ICEFALL" rev-parse HEAD)" = "$ICEFALL_COMMIT"
python3 "$PROJECT/scripts/surt_base/source.py" "$ICEFALL" "$STREAM_RUN/logs/source-$(date -u +%Y%m%dT%H%M%S)-$$"
test ! -e "$STREAM_RUN/started.json"
printf '%s\n' "$STREAM_RUN" > "$STREAM/active-run.txt"
) > "$CELL_LOG" 2>&1
```

### سلول ۲ — بازاستفاده یا ساخت محیط دقیق

نسخه‌ها همان 1B هستند: Python 3.12.10، Torch/torchaudio 2.3.1+cu121،
k2 1.24.4.dev20241030+cuda12.1.torch2.3.1، kaldifeat 1.25.4.dev20240606+cuda12.1.torch2.3.1،
Lhotse 1.33.0، NumPy 1.26.4، SciPy 1.13.1، SentencePiece 0.2.3 و grpcio 1.85.1.
freeze کامل مرجع `configs/surt_base/requirements-kaggle.txt` است. محیط موجود نصب مجدد
نمی‌شود. drift توقف می‌دهد؛ تعمیر grpcio در راهنمای 1B موجود است. Torch/CUDA را تغییر ندهید.
اگر wheelhouse قبلاً ذخیره و متصل شده، مسیر آن را در `WHEELHOUSE` قرار دهید؛ در غیر این صورت
فقط برای محیط مفقود، نصب شبکه‌ای از همان نسخه‌های دقیق انجام می‌شود.

```bash
%%bash
set -euo pipefail
STREAM=/kaggle/working/surt-stream-1c
STREAM_RUN="$(cat "$STREAM/active-run.txt")"
CELL_LOG="$STREAM_RUN/logs/cell-02-environment-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 02-environment; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 02-environment"; else echo "FAILED: 02-environment ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
WORK=/kaggle/working
PROJECT="$WORK/English-Streaming-ASR"
WHEELHOUSE=""
export UV_LINK_MODE=copy
unset UV_SYSTEM_PYTHON
if [ ! -x "$WORK/surt-venv-py312/bin/python" ]; then
  mkdir -p "$WORK/uv-bin"
  curl -LsSf https://astral.sh/uv/0.8.22/install.sh | env UV_INSTALL_DIR="$WORK/uv-bin" UV_NO_MODIFY_PATH=1 sh
  "$WORK/uv-bin/uv" venv --seed --python 3.12.10 "$WORK/surt-venv-py312"
  source "$WORK/surt-venv-py312/bin/activate"
  if [ -n "$WHEELHOUSE" ]; then
    test -d "$WHEELHOUSE"
    (cd "$WHEELHOUSE" && sha256sum -c SHA256SUMS)
    python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$PROJECT/configs/surt_base/requirements-kaggle.txt"
  else
    python -m pip install -r "$PROJECT/configs/surt_base/requirements-kaggle.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      -f https://k2-fsa.github.io/k2/cuda.html \
      -f https://csukuangfj.github.io/kaldifeat/cuda.html
  fi
fi
source "$PROJECT/scripts/surt_stream/env.sh"
python -m pip freeze > "$STREAM_RUN/logs/freeze-$(date -u +%Y%m%dT%H%M%S)-$$.txt"
python -m pip check
python "$PROJECT/scripts/surt_base/dependencies.py" --report "$STREAM_RUN/logs/dependencies-$(date -u +%Y%m%dT%H%M%S)-$$.txt"
python "$PROJECT/scripts/surt_base/runtime.py"
nvidia-smi
) > "$CELL_LOG" 2>&1
```

### سلول ۳ — بازیابی 1B؛ بدون دانلود مدل/داده

در session تازه مقدار `MINI_INPUT` را با مسیر واقعی پوشهٔ خصوصی متصل‌شده جایگزین کنید.
در session موجود می‌تواند خالی بماند. پوشهٔ input باید خودش شامل `mini-one-update.pt` و
`result.json` و زیرپوشه‌های `data/`, `features/`, `tokenizer/` باشد، نه صرفاً آرشیو review.

```bash
%%bash
set -euo pipefail
STREAM=/kaggle/working/surt-stream-1c
STREAM_RUN="$(cat "$STREAM/active-run.txt")"
CELL_LOG="$STREAM_RUN/logs/cell-03-restore-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 03-restore; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 03-restore"; else echo "FAILED: 03-restore ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_stream/env.sh
MINI_INPUT=""
if [ -d "$MINI_PRIOR" ]; then
  python "$PROJECT/scripts/surt_stream/support.py" restore --run "$STREAM_RUN"
else
  test -n "$MINI_INPUT"
  python "$PROJECT/scripts/surt_stream/support.py" restore --input "$MINI_INPUT" --run "$STREAM_RUN"
fi
) > "$CELL_LOG" 2>&1
```

اگر سلول ۳ پیام `Not the reviewed 1B checkpoint/tokenizer contract` داد، پوشه پیدا شده،
اما یک یا چند مقدار در `result.json` با خروجی بررسی‌شدهٔ 1B متفاوت است. نسخهٔ جدید خطا
مسیر فایل و مقادیر `expected` و `actual` هر فیلد ناسازگار را چاپ می‌کند؛ همان خطوط را
برای بررسی نگه دارید. صرف یکسان بودن نام پوشه، یکسان بودن خروجی را ثابت نمی‌کند.
وقتی `$MINI_PRIOR` موجود است، همان پوشه بررسی می‌شود و `MINI_INPUT` استفاده نمی‌شود.
فایل‌های موجود یا pins را برای عبور از خطا تغییر ندهید. اگر شکست در همین بررسی اولیه
رخ داده و `restored-inputs.json` ساخته نشده، پس از به‌روزرسانی کد می‌توانید همان سلول ۳
را دوباره اجرا کنید؛ اجرای مجدد فاز ۰ یا دانلود مدل لازم نیست.

### سلول ۴ — API، feature/state اجزا و منابع واقعی

`AVAILABLE_GPU_HOURS=0` یعنی سهمیه/زمان مجاز session نامعلوم است؛ عددی حدس زده نمی‌شود.
اگر محدودیت واقعی مشاهده‌شدهٔ خودتان را می‌دانید، آن را بر حسب ساعت وارد کنید. برآورد
۱۰/۵۰/۱۰۰ ساعت مخلوط و ۱/۵/۱۰ epoch فقط محاسبهٔ خطی فرضی با batch کوتاه mini است؛
نباید آن را زمان معتبر Full Training دانست. زمان I/O، ذخیره و ارزیابی در آن نیست.

```bash
%%bash
set -euo pipefail
STREAM=/kaggle/working/surt-stream-1c
STREAM_RUN="$(cat "$STREAM/active-run.txt")"
CELL_LOG="$STREAM_RUN/logs/cell-04-probe-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 04-probe; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 04-probe (component/resource checks; native streaming gate OPEN)"; else echo "FAILED: 04-probe ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_stream/env.sh
AVAILABLE_GPU_HOURS=0
nvidia-smi --id=0 --query-gpu=timestamp,uuid,memory.used --format=csv -l 1 > "$STREAM_RUN/logs/gpu-$(date -u +%Y%m%dT%H%M%S)-$$.csv" 2>&1 &
GPU_MONITOR=$!
trap 'kill "$GPU_MONITOR" 2>/dev/null || true; wait "$GPU_MONITOR" 2>/dev/null || true' EXIT
python "$PROJECT/scripts/surt_stream/run.py" "$STREAM_RUN" --available-gpu-hours "$AVAILABLE_GPU_HOURS"
test -s "$STREAM_RUN/result.json"
) > "$CELL_LOG" 2>&1
```

### سلول ۵ — آرشیو شواهد، حتی در شکست

در شکست سلول اول، `RUN_ID` را دستی برابر همان تلاش قرار دهید. در حالت عادی هم نام همین
تلاش را وارد کنید؛ به نام قدیمی تکیه نکنید. از Python پایه استفاده می‌شود تا شکست محیط مانع
حفظ شواهد نشود. فایل‌های مدل، صوت، feature و credential در review بسته‌بندی نمی‌شوند.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/English-Streaming-ASR
STREAM=/kaggle/working/surt-stream-1c
RUN_ID=stream-20261119-a
STREAM_RUN="$STREAM/runs/$RUN_ID"
test -d "$STREAM_RUN/logs"
CELL_LOG="$STREAM_RUN/logs/cell-05-archive-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 05-archive; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 05-archive"; else echo "FAILED: 05-archive ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
python3 "$PROJECT/scripts/surt_stream/support.py" archive "$STREAM_RUN"
) > "$CELL_LOG" 2>&1
```

بستهٔ `/kaggle/working/surt-stream-1c/runs/stream-20261119-a-review.tar.gz` را دانلود و برای
بررسی ارسال کنید؛ تکرار archive پسوند شماره‌دار دارد. کل پوشهٔ 1B قبلی را برای restart بعدی
همچنان نگه دارید. در 1C checkpoint دائمی تازه تولید نمی‌شود.

## تفسیر گزارش و ادامهٔ 1C

| فایل | چیزی که واقعاً نشان می‌دهد |
| --- | --- |
| `api-audit.json` و `upstream-source/` | کلاس‌ها، امضای API، جهت RNNها، هش سورس و فهرست موانع |
| `frontend-report.json` | parity feature، frame count، reset/EOF و bound فقط برای buffer feature |
| `joint-lstm-report.json` | parity هستهٔ LSTM؛ در عدم امکان بازرسی، blocked صریح |
| `inference-report.json` | whole-utterance RTF و حافظه با warmup؛ latency استریم نیست |
| `training-resource-report.json` | هفت update موقت، loss/gradient و زمان/حافظهٔ واقعی همین mini |
| `resource-estimates.json` | چند سناریوی خطی کوچک با فرض‌ها و سهمیهٔ ورودی کاربر |
| `result.json` | تکمیل probe، دست‌نخوردن ورودی‌های 1B و gate باز استریم |

CUDA قبل و بعد timerها synchronize می‌شود؛ peak allocated/reserved جداست. CSV یک‌ثانیه‌ای
مصرف کل دستگاه است و peak دقیق process نیست. زمان feature روی CPU جدا ثبت شده است.
packet feature برابر ۱۰۰ms و context نسبت به مرکز frame برابر ۱۲٫۵ms است؛ context کل مدل،
زمان endpoint و latency استریم تا اتصال native نامعلوم و در JSON برابر null هستند، نه صفر.
این runner endpoint detector تازه‌ای تعریف نمی‌کند.

برای بستن 1C در بخش B هنوز لازم است: state واقعی DPRNN و Zipformer و joint wrapper و
decoder؛ parity قطعه‌بندی با مرجع دارای context یکسان؛ ورودی‌های غیرمضرب chunk؛ reset،
EOF و tail؛ state محدود روی صوت طولانی؛ حفظ channel-major ordering؛ گزارش جداگانهٔ
acquisition/context/endpoint/compute latency و RTF با warmup. نبود API یا شکست parity
یک مانع مهندسی است، نه حکم ناممکن‌بودن adaptation فارسی و نه مجوز ادعای streaming.

پس از عبور واقعی: گام ۲ منبع/manifest/split و سیاست متن؛ گام ۳ مخلوط ساده سپس FastMSS
پین‌شده با word alignment؛ گام ۴ adaptation کوتاه و بودجه‌دار؛ گام ۵ ارزیابی ORC-WER
و کیفیت دو شاخه و latency روی test مستقل؛ گام ۶ جمع‌بندی امکان‌پذیری و محدودیت‌ها.

## اختیاری؛ wheelhouse خصوصی برای جلوگیری از دانلود تکراری محیط

این کار فقط روی Kaggle و یک‌بار در محیط موفق انجام شود؛ چند گیگ فضای اضافه لازم دارد.
اجرای پنج سلول بالا به آن وابسته نیست. فایل زیر را بعد از دانلود/ذخیره در یک ورودی خصوصی
نگه دارید و در sessionهای بعد مسیر آن را در `WHEELHOUSE` سلول ۲ بدهید. فایل‌های venv را
به‌عنوان محیط قابل‌انتقال تضمین نمی‌کنیم؛ wheelhouse با نسخهٔ Python/پلتفرم یکسان نصب می‌شود.
دریافت wheel، نصب محیط فعلی را تغییر نمی‌دهد. اگر wheel دقیق موجود نباشد، نسخه را عوض نکنید.

```bash
%%bash
set -euo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_stream/env.sh
CELL_LOG="$STREAM_RUN/logs/wheelhouse-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START wheelhouse; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: wheelhouse"; else echo "FAILED: wheelhouse ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
WHEELHOUSE=/kaggle/working/surt-wheelhouse-py312-cu121
test ! -e "$WHEELHOUSE"
mkdir "$WHEELHOUSE"
python -m pip download --only-binary=:all: --dest "$WHEELHOUSE" \
  -r "$PROJECT/configs/surt_base/requirements-kaggle.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -f https://k2-fsa.github.io/k2/cuda.html \
  -f https://csukuangfj.github.io/kaldifeat/cuda.html
(cd "$WHEELHOUSE" && sha256sum ./*.whl > SHA256SUMS)
cp "$PROJECT/configs/surt_base/requirements-kaggle.txt" "$WHEELHOUSE/requirements.txt"
) > "$CELL_LOG" 2>&1
```
