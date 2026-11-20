# راهنمای مستقل Kaggle: مرحلهٔ 1C-B استریم کل SURT

تاریخ: ۲۰۲۶-۱۱-۲۰. وضعیت: **کد آمادهٔ اجرای کگل است؛ gate پذیرش باز است.**
این تغییر محلی اجرا یا تست تجربی نشده است. 1A و 1B پذیرفته‌اند؛
`stream-20261119-b` ممیزی/feature/LSTM/resource را تکمیل کرده و سه تست واقعی
kaldifeat در لاگ `unittest-kaldifeat-20261120T105449-515.log` پاس شده‌اند.
این شواهد، اجرای افزایشی کل مدل نیستند. این runbook برای تولید آن شواهد است.

[قرارداد state، شکل‌ها، lookahead، EOF و زمان](SURT_1CB_STATE_CONTRACT.md)
جزء این راهنماست. معماری و وزن/tokenizer/pinهای 1B حفظ می‌شوند. فقط train-0
برای شواهد مهندسی استفاده می‌شود؛ کیفیت فارسی مستقل است و PI-WER قبلی ۱۰۰٪ بوده است.
هیچ آموزش یا update جدیدی انجام نمی‌شود. دانلود مدل/صوت جدید نداریم. محیط موجود
بازاستفاده می‌شود؛ فقط محیط مفقود با نسخه‌های قبلی نصب می‌شود. محدودیت runner:
یک GPU، FP32، ۹۰۰ ثانیه. عبور از زمان محدود، failure و gate باز تولید می‌کند.

## آماده‌سازی و بازیابی

کاربر تغییرات را commit/push کند و SHA کامل همان commit را در سلول ۱ بگذارد؛
SHA تا پیش از push قابل تعیین نیست. Run ID هر تلاش تازه و استفاده‌نشده باشد؛
همان ID را در سلول ۵ نیز وارد کنید. سلول ۱ روی پوشهٔ موجود توقف می‌کند و آن را
حذف نمی‌کند. در شکست، ابتدا سلول ۵؛ سپس تلاش تازه با ID جدید. خروجی و cache قبلی حفظ شوند.

در session موفق قبلی هیچ بارگذاری دوبارهٔ 1B لازم نیست. در session تازه، کل پوشهٔ
`outputs/surt-mini-1b/mini-20261119-d/` را به‌صورت ورودی **خصوصی** Kaggle متصل کنید
و مسیر واقعی آن زیر `/kaggle/input/...` را در `MINI_INPUT` سلول ۳ بگذارید.
پوشه باید checkpoint، tokenizer، data، features و همهٔ JSONها را داشته باشد؛
آرشیو کوچک review کافی نیست. هیچ مرحلهٔ ۰/1A/1B را دوباره اجرا نکنید.
بازیابی پس از بررسی hash در مسیر اصلی زیر انجام می‌شود تا manifest مطلق معتبر بماند:
`/kaggle/working/surt-mini-1b/runs/mini-20261119-d`.
پوشهٔ موجود merge/overwrite نمی‌شود؛ ناسازگاری باید بررسی شود.

هر block یک سلول مستقل است. `%%bash` اولین خط باشد؛ log کامل روی دیسک می‌ماند.
هر سلول environment خودش را source می‌کند و exit code خطا را حفظ می‌کند.

## سلول ۱ — دریافت کد push‌شده با SHA دقیق

```bash
%%bash
set -euo pipefail
STREAM=/kaggle/working/surt-stream-1c
RUN_ID=streamb-20261120-a
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
STREAM_RUN="$STREAM/runs/$RUN_ID"
test ! -e "$STREAM_RUN"
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

## سلول ۲ — محیط تأییدشده، بدون نصب دوباره در صورت وجود

Python 3.12.10، Torch/torchaudio 2.3.1+cu121، k2 و kaldifeat همان buildهای
ثبت‌شده در `configs/surt_base/requirements-kaggle.txt` می‌مانند. اگر wheelhouse
خصوصی قبلی دارید، مسیرش را در `WHEELHOUSE` قرار دهید. drift با تغییر pin حل نشود.

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

## سلول ۳ — اعتبارسنجی یا بازیابی کامل 1B

`MINI_INPUT` فقط در session تازه لازم است؛ مسیر پوشهٔ اصلی run را بدهید.

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

## سلول ۴ — تست واقعی adapterها، stress و اندازه‌گیری

این سلول مجوز درخواست‌شدهٔ تست فقط روی کگل را اعمال می‌کند. runner مدل را یک‌بار
strict load می‌کند، تست‌های GPU را با همان fixture اجرا می‌کند و نتیجه را در log می‌نویسد.
اجرای مستقیم unittest مربوط به native بدون fixture به‌عمد skip می‌شود و شواهد پذیرش نیست.
مرجع DPRNN از اجرای کامل upstream و مرجع recognition از grid/context یکسان است.
Tolerance ثابت `atol=2e-4, rtol=1e-4`؛ هیچ relaxation خودکار وجود ندارد.
پس از تست، دو warmup و پنج اندازه‌گیری، با sync قبل/بعد هر push اجرا می‌شود.

```bash
%%bash
set -euo pipefail
STREAM=/kaggle/working/surt-stream-1c
STREAM_RUN="$(cat "$STREAM/active-run.txt")"
CELL_LOG="$STREAM_RUN/logs/cell-04-native-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 04-native; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 04-native; review gate evidence"; else echo "FAILED: 04-native ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_stream/env.sh
# These CPU tests already passed in 1C-A; rerun only this small frontend regression.
python -m unittest discover -s "$PROJECT/tests" -p test_surt_stream_kaldifeat.py -v
nvidia-smi --id=0 --query-gpu=timestamp,uuid,memory.used --format=csv -l 1 > "$STREAM_RUN/logs/gpu-native-$(date -u +%Y%m%dT%H%M%S)-$$.csv" 2>&1 &
GPU_MONITOR=$!
trap 'kill "$GPU_MONITOR" 2>/dev/null || true; wait "$GPU_MONITOR" 2>/dev/null || true' EXIT
python "$PROJECT/scripts/surt_stream/run_b.py" "$STREAM_RUN"
test -s "$STREAM_RUN/result.json"
) > "$CELL_LOG" 2>&1
```

## سلول ۵ — آرشیو حتی در شکست

اگر سلول ۴ شکست خورد، این سلول را اجرا کنید. اگر شکست داخل runner رخ داده باشد،
`failure.json` و log تست‌ها ذخیره می‌شوند. شکست تست frontend پیش از runner در log سلول است.
Run ID را برابر تلاش جاری بگذارید. آرشیوهای تکراری پسوند شماره‌دار دارند.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/English-Streaming-ASR
STREAM=/kaggle/working/surt-stream-1c
RUN_ID=streamb-20261120-a
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

## تفسیر خروجی و gate

بستهٔ `streamb-20261120-a-review.tar.gz` در مسیر `surt-stream-1c/runs/` را برای
بازبینی دریافت کنید. وزن، صوت و feature وارد review/Git نمی‌شوند؛ restart کامل 1B
را جدا و خصوصی حفظ کنید. این runner checkpoint تازه‌ای ذخیره نمی‌کند.

| فایل | تفسیر |
| --- | --- |
| `runtime.json` | dependency، GPU، وزن/tokenizer و هش source دقیق |
| `api-audit.json`، `upstream-source/` | نسخهٔ واقعی کد استفاده‌شده و قراردادهای لازم برای بازبینی |
| `logs/native-tests.log` | سه تست مدل واقعی؛ خطا و subtest دقیق در شکست |
| `acceptance-tests.json` | parity عددی هر دو شاخه، token ID دقیق، packet ثابت/نامنظم، reset زنده، ورودی کوتاه، EOF تکراری و EOF روی آخرین packet، tail و stress |
| `streaming-report.json` | خروجی پیش از EOF، متن دو شاخه، timeline، دو warmup/پنج اندازه‌گیری، RTF و حافظه |
| `result.json` | تکمیل آزمون و gate؛ `REVIEW_REQUIRED` به معنی پذیرش خودکار نیست |
| `failure.json` | مانع صریح؛ gate باز می‌ماند |

در `acceptance-tests.json` خروجی mask با DPRNN کامل upstream مقایسه می‌شود؛
خروجی composite با مرجع دارای context یکسان و joint/greedy مستقل مقایسه می‌شود.
شمار فریم‌ها برای هر طول دقیق است. reset روی همان instance، هم پس از EOF و هم
میان یک ورودی نیمه‌تمام بررسی می‌شود. شاخه‌ها با ترتیب ۰/۱ مقایسه می‌شوند؛ جابه‌جایی
برای پنهان‌کردن اختلاف ممنوع است. تست طولانی ۱۲۰ ثانیه تکرار train است، نه گفت‌وگوی
بلند واقعی؛ packetها به‌تدریج ساخته می‌شوند و acoustic output نگه داشته نمی‌شود.
checkpointهای storage در طول آن باید محدود بمانند؛ counter عددی می‌تواند افزایش یابد.

شرط شواهد کامل: همهٔ تست‌ها بدون skip پاس شوند، خروجی encoder و دست‌کم یک token
غیرblank پیش از پایان فایل دیده شود، state صوتی زیر حد ثابت بماند و گزارش context/EOF
قابل بررسی باشد. نبود token پیش از EOF gate را باز نگه می‌دارد حتی اگر امکان اجرای
اجزای آکوستیکی ثابت شده باشد. `REVIEW_REQUIRED` نیاز به بازبینی شواهد واقعی دارد؛
این راهنما از پیش موفقیت اعلام نمی‌کند.

زمان دریافت، context حداقل، انتظار اضافهٔ packet و compute را جدا بخوانید. timeline
شبیه‌سازی روی replay بدون pacing است؛ latency زنده یا endpoint ادعا نمی‌شود و
فیلدهای نامعلوم null هستند. زمان خروجی نسبت به grid اسمی است و timestamp کلمه نیست.
RTF compute و wall متفاوت‌اند؛ peak CUDA allocated/reserved با مصرف کل GPU در CSV
یکسان نیست. RSS کل process شامل تست و مرجع نیز هست. برای خطا pin/tolerance را تغییر
ندهید؛ فایل دقیق، exception، shape و اختلاف عددی را از log گزارش و gate را باز نگه دارید.

کیفیت فارسی با این مرحله اثبات نمی‌شود؛ تطابق streaming با مرجع می‌تواند همراه متن
ضعیف باشد. تفسیر و شنیدن خروجی و ارزیابی متن/گوینده مستقل از امکان اجرای استریم است.
