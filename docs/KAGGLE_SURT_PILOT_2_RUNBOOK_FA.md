# راهنمای مستقل Kaggle: داده و سیاست نوشتار پایلوت فارسی، مرحلهٔ ۲

تاریخ ۲۰۲۶-۱۱-۲۰. پیاده‌سازی محلی؛ **هیچ تست یا پردازش صوت اجرا نشده و gate باز است.**
[قرارداد داده، موانع دسترسی و تفسیر گزارش‌ها](SURT_PILOT_DATA_CONTRACT_FA.md) جزء این راهنماست.
اجرا فقط روی Kaggle است؛ GPU برای این مرحله لازم نیست. مراحل قبلی دوباره اجرا نمی‌شوند.
این runbook آموزش، feature یا مخلوط نمی‌سازد. کاربر commit/push می‌کند؛ SHA تازه هنوز
معلوم نیست و حتماً باید SHA کامل commit حاوی همین تغییرات در سلول اول قرار گیرد.

هر سلول در شروع می‌گوید چه کاری می‌کند؛ در پایان آخرین خط خلاصهٔ واقعی فرمان را نشان
می‌دهد و مسیر log کامل را اعلام می‌کند. در خطا، exit code حفظ و انتهای log نمایش داده
می‌شود. بنابراین نتیجه فقط «02 ok» نیست. متن یا مشخصات صوت test روی notebook چاپ نمی‌شود.
هر block یک سلول مستقل است؛ shellها با `%%bash` شروع می‌شوند. سلول ۲ب پایتون است
تا Secret از API خود Kaggle گرفته شود و در shell یا notebook نوشته نشود. از run ID استفاده‌شده دوباره
استفاده نکنید؛ نمونهٔ `pilot2-20261120-a` فقط وقتی مجاز است که هنوز وجود نداشته باشد.
تمام cacheها، خروجی‌های موفق و شکست‌خورده حفظ شوند؛ هیچ reset/delete پیشنهاد نشده است.

در session قبلی محیط و mini اصلی استفاده می‌شوند. در session تازه، wheelhouse خصوصی
اختیاری و **کل** run پذیرفته‌شدهٔ `mini-20261119-d` را متصل کنید؛ review archive کافی
نیست. mini در مسیر canonical قدیمی بازیابی می‌شود تا مسیر مطلق manifestها معتبر بماند.
restart قبلی را merge/overwrite نکنید. برای ledger سایر attemptها پوشهٔ خصوصی جدا بدهید.

کاربر اکنون خروجی HTTP 200، Range با HTTP 206 و ۵۰۰۰ ردیف metadata از Kaggle
ارائه کرده است؛ دسترسی واقعی از این شواهد پذیرفته است و کفایت subset هنوز جداست. یک کلید تازه در Kaggle Secrets با نام `Persian_ASR_SURT`
بسازید و دسترسی notebook به همان Secret را فعال کنید؛ مقدار آن را داخل سلول ننویسید.
تا رفع مانع subset، سلول‌های ۱، ۲، ۲ب، ۳ و ۷ قابل اجرا هستند. سلول ۴ با template خالی عمداً
متوقف می‌شود: subset رسمی زیر سقف و receipt واقعی لازم است. برای رفع این خطا کل corpus
را دانلود نکنید؛ از CV22 جایگزین هم استفاده نکنید. endpoint sample رسمی هنوز برای فارسی
تأیید نشده است. در صورت ارائهٔ مسیر رسمی محدود، metadata و صوت همان subset باید روی
Kaggle و در سقف config فراهم شوند. ابزار فقط API کنترلی را probe می‌کند و **هیچ URL صوتی را دریافت نمی‌کند**؛
سقف import را قبل از خواندن صوت اعمال می‌کند، اما دریافت بیرون این ابزار را اندازه نمی‌گیرد.
پس دریافت اولیه نیز باید مستقلاً مدرک سقف ۲۵۶MiB داشته باشد.

## سلول ۱ — checkout دقیق و Run ID تازه

این سلول در شکست checkout هم log مستقل نگه می‌دارد. پوشهٔ قبلی یا تغییر محلی را حذف
نمی‌کند. نسخهٔ recipe قبلی دست نمی‌خورد؛ این مرحله نیازی به clone دوبارهٔ Icefall ندارد.

```bash
%%bash
set -euo pipefail
PILOT=/kaggle/working/surt-pilot-2
RUN_ID=pilot2-20261120-a
PROJECT_COMMIT=PASTE_FULL_PUSHED_COMMIT_SHA_HERE
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$PROJECT_COMMIT" =~ ^[0-9a-f]{40}$ ]]
RUN="$PILOT/runs/$RUN_ID"
test ! -e "$RUN"
mkdir -p "$PILOT"
LOG="$PILOT/bootstrap-$RUN_ID-$(date -u +%Y%m%dT%H%M%S)-$$.log"
echo "کار: دریافت کد روی SHA تعیین‌شده و ساخت run تازه؛ log: $LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then tail -n 1 "$LOG"; else echo "خطا در checkout یا ساخت run ($status)؛ خروجی قبلی حفظ شد"; tail -n 25 "$LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
PROJECT=/kaggle/working/English-Streaming-ASR
REMOTE=https://github.com/manzourali/English-Streaming-ASR.git
if [ -d "$PROJECT/.git" ]; then
  test "$(git -C "$PROJECT" remote get-url origin)" = "$REMOTE"
  git -C "$PROJECT" diff --quiet
  git -C "$PROJECT" diff --cached --quiet
else
  test ! -e "$PROJECT"
  git clone --filter=blob:none --no-checkout "$REMOTE" "$PROJECT"
fi
git -C "$PROJECT" fetch --no-tags origin "$PROJECT_COMMIT"
git -C "$PROJECT" checkout --no-overwrite-ignore --detach "$PROJECT_COMMIT"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$PROJECT_COMMIT"
# Bootstrap uses stdlib only; activate a previous venv if present.
if [ -x /kaggle/working/surt-venv-py312/bin/python ]; then
  source "$PROJECT/scripts/surt_pilot/env.sh"
fi
python3 "$PROJECT/scripts/surt_pilot/pipeline.py" init "$RUN"
printf '%s\n' "$PROJECT_COMMIT" > "$RUN/project-commit.txt"
printf '%s\n' "$RUN" > "$PILOT/active-run.txt"
echo "کد روی SHA $PROJECT_COMMIT قرار گرفت؛ run $RUN_ID تازه ساخته شد و دریافت داده صفر بود."
) > "$LOG" 2>&1
cp -n "$LOG" "$RUN/logs/$(basename "$LOG")"
```

## سلول ۲ — محیط پین‌شده، نصب فقط در صورت نبود محیط

```bash
%%bash
set -euo pipefail
PILOT=/kaggle/working/surt-pilot-2
RUN="$(cat "$PILOT/active-run.txt")"
[[ "$RUN" == "$PILOT"/runs/* ]]
test -d "$RUN/logs"
LOG="$RUN/logs/cell-02-$(date -u +%Y%m%dT%H%M%S)-$$.log"
echo "کار: بازیابی یا بررسی محیط با pinهای قبلی؛ log: $LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then tail -n 1 "$LOG"; echo "لاگ کامل: $LOG"; else echo "خطا ($status)؛ run و cache حفظ شدند"; tail -n 25 "$LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
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
    (cd "$WHEELHOUSE" && sha256sum -c SHA256SUMS)
    python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$PROJECT/configs/surt_base/requirements-kaggle.txt"
  else
    python -m pip install -r "$PROJECT/configs/surt_base/requirements-kaggle.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      -f https://k2-fsa.github.io/k2/cuda.html \
      -f https://csukuangfj.github.io/kaldifeat/cuda.html
  fi
fi
source "$PROJECT/scripts/surt_pilot/env.sh"
python -c 'import sys; assert sys.version_info[:3] == (3, 12, 10), sys.version'
python -m pip freeze > "$RUN/logs/freeze-$(date -u +%Y%m%dT%H%M%S)-$$.txt"
python -m pip check
python "$PROJECT/scripts/surt_base/dependencies.py" --report "$RUN/logs/dependencies-$(date -u +%Y%m%dT%H%M%S)-$$.json"
echo 'محیط Python 3.12.10 و تمام بسته‌های پین‌شده بررسی شدند؛ drift مشاهده نشد و هیچ مدل یا صوتی بارگذاری نشد.'
) > "$LOG" 2>&1
```

نصب جدید ممکن است بیش از زمان پردازش پایلوت طول بکشد؛ سقف ۹۰۰ ثانیه برای هر فرمان
pipeline است. شکست resolver/drift را با تغییر pin یا حذف cache حل نکنید؛ log را نگه دارید.

## سلول ۲ب — بررسی توکن از Kaggle Secrets، بدون دریافت صوت

نام Secret حساب شما `Persian_ASR_SURT` است؛ مقدار آن فقط برای فرایند فرزند در متغیر
محیطی استاندارد `MDC_API_KEY` قرار می‌گیرد. نام Secret با نام متغیر محیطی یکی نیست.

این یک سلول **پایتون** است؛ `%%bash` نگذارید. Secret و درخواست کنترلی HTTP با همان
Python میزبان notebook و `requests` موجود در آن اجرا می‌شوند تا با کد موفق کاربر یکسان
باشند. محیط عددی پین‌شدهٔ مرحلهٔ ۲ دست‌نخورده می‌ماند؛ پکیج تازه نصب نمی‌شود و نسخهٔ
requests/Python میزبان در گزارش ثبت می‌شود. POST بدون body و بدون Content-Type اجباری
ارسال می‌شود؛ GET کاتالوگ قبل از آن حذف شده است. header، کلید و پاسخ خام چاپ نمی‌شوند.
کاراکتر نامرئی U+200D گزارش‌شده در کلید حذف و قالب ASCII بررسی می‌شود.

به‌صورت پیش‌فرض فقط `/download` بررسی می‌شود: یک درخواست صدور لینک، بدون retry،
HEAD یا دریافت صوت. `CHECK_SAMPLE=False` را برای بررسی دسترسی حساب نگه دارید؛ sample
بررسی مستقلی است و تنها با `True` یک درخواست دوم می‌فرستد. این تغییر درخواست را با کد
موفق کاربر هماهنگ می‌کند، اما بدون اجرای Kaggle نمی‌توان رفع 403 را تضمین کرد.

```python
from pathlib import Path
import os
import subprocess
import sys
import time
from kaggle_secrets import UserSecretsClient

pilot = Path('/kaggle/working/surt-pilot-2')
run = Path((pilot / 'active-run.txt').read_text().strip())
if run.parent != pilot / 'runs' or not (run / 'logs').is_dir():
    raise RuntimeError('ابتدا سلول ۱ را با Run ID تازه اجرا کنید')
CHECK_SAMPLE = False
log_path = run / 'logs' / f'cell-02b-access-{time.time_ns()}.log'
print(f'کار: بررسی دسترسی دیتاست با requests؛ sample اختیاری={CHECK_SAMPLE}؛ log: {log_path}')
try:
    secret = UserSecretsClient().get_secret('Persian_ASR_SURT')
except Exception:
    raise RuntimeError('Secret با نام Persian_ASR_SURT و دسترسی notebook را در Kaggle تنظیم کنید') from None
child_env = dict(os.environ)
child_env['MDC_API_KEY'] = secret
try:
    with log_path.open('x') as log:
        completed = subprocess.run([
            sys.executable,
            '/kaggle/working/English-Streaming-ASR/scripts/surt_pilot/access.py', str(run),
            *(['--check-sample'] if CHECK_SAMPLE else [])
        ], env=child_env, stdout=log, stderr=subprocess.STDOUT)
finally:
    child_env.pop('MDC_API_KEY', None)
    del secret
lines = log_path.read_text().splitlines()
print('\n'.join(lines[-5:]))
if completed.returncode:
    raise RuntimeError(f'بررسی API کامل نشد؛ exit={completed.returncode}؛ log و گزارش را حفظ کنید')
```

اگر فقط HTTP 403 دیده‌اید، خط کامل تعیین‌کننده است: `download HTTP=403` یعنی همین
درخواست صدور لینک رد شده است؛ sample هنوز درخواستی نداشته است. `download HTTP=200`
و `sample HTTP=403` یعنی دسترسی دیتاست تأیید شده و فقط بررسی sample رد شده است؛ این
حالت شکست توکن محسوب نمی‌شود. علت دقیق 403 در اجرای قبلی کاربر مشخص نیست: نسخهٔ قدیمی
از urllib در venv و POST با `{}` استفاده می‌کرد، درحالی‌که کد موفق کاربر requests میزبان
و POST بدون body بود. تفاوت client/بدنه علت‌های محتمل‌اند و اثبات نشده‌اند.

برای اعمال تغییر، ابتدا کاربر commit/push کند؛ سلول ۱ را با SHA دقیق تازه و Run ID تازه،
سپس سلول ۲ و ۲ب اجرا کنید. محیط موجود در سلول ۲ فقط ممیزی می‌شود. run شکست‌خوردهٔ
`pilot2-20261120-a` و logهای آن حفظ شوند؛ فایل‌ها را در همان run ویرایش یا overwrite نکنید.
از هر دو گزارش HTTP status و نوع پاسخ استفاده کنید؛ پاسخ HTML/challenge می‌تواند به
بررسی لایهٔ وب کمک کند ولی به‌تنهایی علت را اثبات نمی‌کند. کلید و URL را ارسال نکنید.

`download HTTP=200` همراه `authenticated_access_confirmed=true` یعنی صدور لینک واقعی
تأیید شده است. sample برابر 404 یعنی دیتاست یافت نشد یا sample منتشر نشده؛ از این status
به‌تنهایی «حتماً sample ندارد» نتیجه نگیرید. پاسخ 200 sample همراه حجم زیر سقف نیز هنوز
کفایت metadata/split را اثبات نمی‌کند. هیچ URL موقتی ذخیره نمی‌شود و این سلول جای سلول ۴
را نمی‌گیرد. خروجی گزارش در review archive حفظ می‌شود. نمونه‌کد `curl -o` کاربر و
`download_dataset/load_dataset` را در این پایلوت اجرا نکنید؛ سقف دریافت محدود باقی است.
پیوست جدید کاربر ثابت می‌کند HEAD موفق شرط لازم نیست: HEAD برابر 403 و GET Range
برابر 206 بوده است. این سلول HEAD نمی‌فرستد و همچنان هیچ body آرشیوی دریافت نمی‌کند.
filename/size/checksum پاسخ تازه با metadata آرشیوِ شواهد بازبینی‌شده مقایسه می‌شوند؛
تغییر/فقدان آن‌ها مانع ادامهٔ خودکار است، نه مجوز تغییر pin. هش کامل archive محاسبه نشده است.

خواندن prefix در کد ارسالی کاربر، امکان مشاهدهٔ metadata را نشان داد، اما ۹٫۲۵MiB انتقال
و فقط ۴ گوینده گزارش کرد؛ هنوز دریافت مطابق سقف metadata و صوت منتخب ثابت نیست.
آن کد را جای سلول ۴ قرار ندهید. فایل کامل ۵۰۰۰ردیفی یا ledger خصوصی همان آزمایش را
برای حفظ exposure نگه دارید؛ ۴۹۹۷ ردیف در پیوست شناسه ندارند و gate باز می‌ماند.
[جزئیات ممیزی و کنترل‌های لازم برای Range](SURT_PILOT_DATA_CONTRACT_FA.md).

اگر probe موفق شد، مسیر و SHA256 گزارش و زمان آن را در `account_access_evidence` و
`access_verified_at` receipt خصوصی ارجاع دهید. فیلدهای `official_subset_route` و مدرک
metadata را فقط پس از احراز subset تکمیل کنید؛ لینک archive کامل جای آن‌ها نیست.

## سلول ۳ — بازیابی mini و تست قرارداد مرحلهٔ ۲

```bash
%%bash
set -euo pipefail
PILOT=/kaggle/working/surt-pilot-2
RUN="$(cat "$PILOT/active-run.txt")"
[[ "$RUN" == "$PILOT"/runs/* ]]
test -d "$RUN/logs"
LOG="$RUN/logs/cell-03-$(date -u +%Y%m%dT%H%M%S)-$$.log"
echo "کار: اعتبارسنجی mini پذیرفته‌شده و اجرای تست‌های داده/متن روی Kaggle؛ log: $LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then tail -n 1 "$LOG"; echo "لاگ کامل: $LOG"; else echo "خطا ($status)؛ run و cache حفظ شدند"; tail -n 25 "$LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_pilot/env.sh
MINI_INPUT=""
python - "$RUN" "$MINI_INPUT" <<'PYCODE'
import json, sys
from pathlib import Path
from support import restore
report = restore(Path(sys.argv[2]) if sys.argv[2] else None)
with (Path(sys.argv[1]) / 'private/restored-mini.json').open('x') as f:
    json.dump(report, f, indent=2)
print('mini پذیرفته‌شده با hashهای قبلی در مسیر canonical معتبر است؛ مراحل قبلی اجرا نشدند.')
PYCODE
cd "$PROJECT"
python -m unittest discover -s tests -p test_surt_pilot.py -v
python - "$RUN" <<'PYCODE'
import hashlib, json, subprocess, sys
from pathlib import Path
run=Path(sys.argv[1])
test=Path('tests/test_surt_pilot.py')
with (run/'tests-passed.json').open('x') as f:
    json.dump(dict(status='passed', suite=str(test),
        suite_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
        project_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        scope='contract unit tests only; not real CV27 or ASR quality evidence'), f, indent=2)
print('mini بازیابی/تأیید شد و تست‌های قرارداد متن، split، حذف exposure، سقف و قفل hash پاس شدند؛ شواهد CV27 هنوز لازم است.')
PYCODE
) > "$LOG" 2>&1
```

اگر mini خصوصی ناقص است، مسیر واقعی پوشهٔ کامل را در `MINI_INPUT` قرار دهید. این سلول
وزن یا feature را برای محاسبه load نمی‌کند؛ فایل‌ها فقط hash/بازیابی می‌شوند. شکست به
معنی لزوم اجرای دوبارهٔ 1A/1B نیست. receipt و metadata هنوز لازم نیستند.

## سلول ۴ — دریافت محدودِ تأییدشده، metadata و split ثابت

```bash
%%bash
set -euo pipefail
PILOT=/kaggle/working/surt-pilot-2
RUN="$(cat "$PILOT/active-run.txt")"
[[ "$RUN" == "$PILOT"/runs/* ]]
test -d "$RUN/logs"
LOG="$RUN/logs/cell-04-$(date -u +%Y%m%dT%H%M%S)-$$.log"
echo "کار: اعتبارسنجی دسترسی subset رسمی و فریز انتخاب پیش از صوت؛ log: $LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then tail -n 1 "$LOG"; echo "لاگ کامل: $LOG"; else echo "خطا ($status)؛ run و cache حفظ شدند"; tail -n 25 "$LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_pilot/env.sh
# Fill only AFTER an official bounded route and its terms have been verified.
# No full-archive download/extraction; no unofficial replacement.
METADATA=/kaggle/input/REPLACE_PRIVATE_OFFICIAL_SUBSET/metadata.jsonl
RECEIPT=/kaggle/input/REPLACE_PRIVATE_OFFICIAL_SUBSET/access-receipt.json
EXTRA_LEDGERS=/kaggle/input/REPLACE_PRIVATE_PRIOR_LEDGERS
python "$PROJECT/scripts/surt_pilot/pipeline.py" plan "$RUN" \
  --metadata "$METADATA" --receipt "$RECEIPT" --ledgers "$EXTRA_LEDGERS"
) > "$LOG" 2>&1
```

`EXTRA_LEDGERS` باید همهٔ inspection/exposure/selectionهای attemptهای دیگر را شامل شود؛
اگر هیچ‌کدام موجود نیست، یک پوشهٔ خالی واقعی بدهید. این نسخه تنها mini ذخیره‌شده را pin
کرده است؛ کامل‌بودن تاریخچه باید بازبینی شود. receipt template در configs راهنمای فیلدهاست؛
آن را با مقدار ساختگی پر نکنید. نبود مسیر رسمی/metadata معتبر اینجا مانع است. پس از شکست
سلول ۷ را اجرا کنید. انتخاب با seed جدید برای بهترشدن نتیجهٔ test ممنوع است.

## سلول ۵ — بررسی صوت همان انتخاب، بدون پرکردن جای حذف‌ها

```bash
%%bash
set -euo pipefail
PILOT=/kaggle/working/surt-pilot-2
RUN="$(cat "$PILOT/active-run.txt")"
[[ "$RUN" == "$PILOT"/runs/* ]]
test -d "$RUN/logs"
LOG="$RUN/logs/cell-05-$(date -u +%Y%m%dT%H%M%S)-$$.log"
echo "کار: decode محدود و اندازه‌گیری صوت با حفظ raw و normalized در manifest؛ log: $LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then tail -n 1 "$LOG"; echo "لاگ کامل: $LOG"; else echo "خطا ($status)؛ run و cache حفظ شدند"; tail -n 25 "$LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_pilot/env.sh
AUDIO_ROOT=/kaggle/input/REPLACE_PRIVATE_OFFICIAL_SUBSET/clips
python "$PROJECT/scripts/surt_pilot/pipeline.py" audio "$RUN" --audio-root "$AUDIO_ROOT"
) > "$LOG" 2>&1
```

ورودی فقط فایل‌های کلیپ با basename اصلی است؛ آرشیو قابل قبول نیست. همهٔ اندازه‌ها قبل
از خواندن صوت با رزرو طرح کنترل می‌شوند؛ فایل خراب، مدت نامعتبر و duplicate با دلیل حذف
می‌شوند. decoder ورودی را resample/mono می‌کند؛ شکل و rate اصلی و خروجی جدا ثبت می‌شوند.
این سلول صرفاً سلامت فنی را گزارش می‌کند و ادعای تأیید شنیداری ندارد.

## سلول ۶ — tokenizer فقط با train، سپس گزارش بدون tune

```bash
%%bash
set -euo pipefail
PILOT=/kaggle/working/surt-pilot-2
RUN="$(cat "$PILOT/active-run.txt")"
[[ "$RUN" == "$PILOT"/runs/* ]]
test -d "$RUN/logs"
LOG="$RUN/logs/cell-06-$(date -u +%Y%m%dT%H%M%S)-$$.log"
echo "کار: ساخت BPE train، فریز vocabulary و گزارش پوشش splitها؛ log: $LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then tail -n 1 "$LOG"; echo "لاگ کامل: $LOG"; else echo "خطا ($status)؛ run و cache حفظ شدند"; tail -n 25 "$LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_pilot/env.sh
python "$PROJECT/scripts/surt_pilot/pipeline.py" tokenizer "$RUN"
python "$PROJECT/scripts/surt_pilot/pipeline.py" finalize "$RUN"
python - "$RUN" <<'PYCODE'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
report = json.loads((run/'review/tokenizer-summary.json').read_text())
result = json.loads((run/'review/result.json').read_text())
counts = '، '.join(f"{split}: {item['unknown']}/{item['tokens']} unknown" for split, item in report['splits'].items())
print(f"BPE با {report['actual_vocab']} قطعه فریز شد؛ {counts}؛ gate={result['gate_2']} با {len(result['blockers'])} مانع؛ test برای tune استفاده نشد.")
PYCODE
) > "$LOG" 2>&1
```

`tokenizer-summary.json` پوشش، unknown و round-trip train/validation/test را دارد؛
برای گزارش نامطلوب test تنظیمات عوض نکنید. کیفیت بازشناسی یا سازگاری checkpoint از آن
نتیجه نمی‌شود. `finalize` در نبود شواهد تداوم گوینده و duplicate gate را باز نگه می‌دارد.
`REVIEW_REQUIRED` نیز نیازمند بازبینی است. اگر فرمان tokenizer شکست بخورد finalize اجرا
نمی‌شود؛ لاگ و `failure-*.json` حفظ می‌شوند.

## سلول ۷ — آرشیو کوچک حتی پس از شکست؛ حفظ restart خصوصی

```bash
%%bash
set -euo pipefail
PILOT=/kaggle/working/surt-pilot-2
RUN="$(cat "$PILOT/active-run.txt")"
[[ "$RUN" == "$PILOT"/runs/* ]]
test -d "$RUN/logs"
LOG="$RUN/logs/cell-07-$(date -u +%Y%m%dT%H%M%S)-$$.log"
echo "کار: بسته‌بندی summaryهای قابل بازبینی بدون صوت و متن؛ log: $LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then tail -n 1 "$LOG"; echo "لاگ کامل: $LOG"; else echo "خطا ($status)؛ run و cache حفظ شدند"; tail -n 25 "$LOG"; fi; exit "$status"' EXIT
(
set -euo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_pilot/env.sh
python "$PROJECT/scripts/surt_pilot/pipeline.py" archive "$RUN"
python "$PROJECT/scripts/surt_pilot/restart.py" pack "$RUN"
echo "review کوچک و restart خصوصی با hash ساخته شدند؛ هر دو کنار run هستند و restart برای انتشار نیست."
) > "$LOG" 2>&1
```

آرشیو `pilot2-...-review-....tar.gz` کنار run ساخته می‌شود؛ هر بار نام تازه دارد.
فقط summaryهای review و نتیجهٔ تست و hashهای آن‌ها وارد می‌شوند؛ full log، متن، شناسه،
vocabulary، receipt، وزن و صوت بیرون review هستند. برای خطای دقیق، log همان سلول را در
فضای خصوصی ببینید؛ failureهای کامل هم خصوصی‌اند. `review/status.json` وضعیت آغازین است؛
اگر `review/result.json` وجود داشت نتیجهٔ نهایی را بخوانید، نبود آن یعنی کار کامل نشده است.

برای restart، کل `$RUN` را به‌همراه logها، manifest، tokenizer، raw metadata، receipt و
صوت canonical **خصوصی** حفظ کنید؛ ورودی رسمی اصلی، full mini پذیرفته‌شده و wheelhouse/
freeze قبلی را هم نگه دارید. سلول ۷ index دارای SHA256 تمام فایل‌ها را در restart خصوصی ذخیره می‌کند. برای بازیابی،
پس از checkout همان SHA و source محیط، فرمان زیر را فقط روی Kaggle اجرا کنید؛
restore تمام hashها را پیش از نوشتن می‌سنجد و مقصد موجود را رد می‌کند.

```bash
%%bash
set -euo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_pilot/env.sh
ARCHIVE=/kaggle/input/REPLACE_PRIVATE_RESTART/pilot2-REPLACE-PRIVATE-restart.tar.gz
RESTORE_LOG=/kaggle/working/pilot2-restore-$(date -u +%Y%m%dT%H%M%S)-$$.log
echo "کار: بازیابی restart خصوصی پس از کنترل hash؛ log: $RESTORE_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then tail -n 1 "$RESTORE_LOG"; else echo "بازیابی ناموفق؛ پوشهٔ قبلی حفظ شد"; tail -n 25 "$RESTORE_LOG"; fi; exit "$status"' EXIT
python "$PROJECT/scripts/surt_pilot/restart.py" restore "$ARCHIVE" > "$RESTORE_LOG" 2>&1
```

این بازیابی active-run را تغییر نمی‌دهد؛ run بازسازی‌شده سابقهٔ محفوظ است. برای تلاش
جدید از سلول ۱ و Run ID تازه شروع کنید.  مسیر canonical این run را عیناً حفظ کنید چون manifest مطلق است؛
پوشهٔ موجود merge یا overwrite نشود. برای تلاش تازه، Run ID جدید و plan تازهٔ ثبت‌شده
لازم است؛ test قبلاً دیده‌شده همچنان exposed محسوب می‌شود. cache قدیمی پاک نشود.
هیچ‌یک از این مراحل commit/push نمی‌کند.

فایل‌های اصلی برای تفسیر: `plan-summary.json` تعداد/رزرو و hash انتخاب؛
`audio-summary.json` آمار واقعی/حذف/محدودیت duplicate؛ `tokenizer-summary.json` پوشش
بدون tune و وضعیت انتقال UNVERIFIED؛ `result.json` gate و موانع. قرارداد کامل schema،
سیاست فارسی/لاتین و شروط پذیرش در سند همراه است. تا بازبینی و رفع موانع، مرحلهٔ ۳،
FastMSS و آموزش شروع نشوند.
