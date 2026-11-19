# گام ۱B — mini فارسی واقعی در Kaggle

وضعیت: **کد و راهنمای اجرا آماده؛ gate 1B تا بازبینی خروجی واقعی Kaggle باز است.**
گام 1A با هش ۴۴ فایل بسته، قرارداد strict، decode واقعی و تأیید شنیداری کاربر در
۲۰۲۶-۱۱-۱۳ پذیرفته شد؛ [گزارش بررسی](experiments/surt_mini_1b_preparation_2026-11-13.md).
این راهنما مستقل از سلول‌های smoke است و فقط همین شش سلول، به ترتیب، اجرا می‌شوند.
GPU تک‌کارت و Internet روشن باشد. اجرای سنگین، نصب GPU یا دانلود صوت روی Mac ممنوع است.
ابتدا کاربر تغییرات کد را commit/push کند؛ عامل این کار را انجام نمی‌دهد.

## پیش‌نیاز و حفظ شواهد

در session موفق 1A اجرا کنید: `/kaggle/working/surt-venv-py312`،
`/kaggle/working/icefall-surt-base` و `/kaggle/working/surt-base-1a` موجود باشند.
وزن، BPE انگلیسی، `results/result.json` و `checkpoint-contract.json` قبلی لازم‌اند.
در session جدید، assets ذخیره‌شده و مجاز قبلی را به همین مسیرها بازگردانید؛ اگر venv قابل
استفاده نیست، محیط را طبق سلول ۲ راهنمای 1A بازسازی کنید. صرف وجود پوشهٔ venv کافی نیست.
سلول ۲ این راهنما نصب/upgrade نمی‌کند؛ drift محیط باید با حفظ freeze بررسی شود.
نسخه‌های اصلی همان Python 3.12.10، Torch/torchaudio 2.3.1+cu121، k2، kaldifeat،
Lhotse 1.33.0، NumPy 1.26.4، SciPy 1.13.1 و SentencePiece 0.2.3 هستند.
soundfile 0.14.0 باید decoder MP3 داشته باشد. از `datasets`/torchcodec/FFmpeg استفاده نمی‌شود.

سلول اول مانند 1A ریپو را به **SHA کاملِ push‌شده** به‌روزرسانی می‌کند؛ تغییرات tracked،
تداخل فایل‌های ignored و remote اشتباه باعث توقف می‌شوند. Icefall همان revision
`f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1` و همان دو patch ثبت‌شده را نگه می‌دارد.
پوشه‌ها، وزن رسمی و خروجی‌های 1A پاک یا بازنویسی نمی‌شوند.

هر تلاش `RUN_ID` جدا دارد. برای اجرای تازه فقط `RUN_ID` سلول ۱ را عوض کنید؛
کد آماده‌سازی روی خروجی موجود متوقف می‌شود و update تکراری همان run ممنوع است.
لاگ هر اجرای سلول نام یکتا دارد؛ archive هم شماره می‌گیرد و قبلی را بازنویسی نمی‌کند.
cache دریافت صوت مشترک و محدود باقی می‌ماند؛ هنگام پرشدن، خودکار حذف نمی‌شود.
پس از هر خطا، کار مدل را متوقف و سلول ۶ را اجرا کنید. archive داشتن به معنی عبور gate نیست.

## منبع، بودجه و split

فقط `aliyzd95/common_voice_22_0_fa`، config=`default`، revision واقعی بررسی‌شده:
`5dc7ac366099292d4fbf079c99d58c1b2fba602d`. API عمومی و ungated بود؛ README پین‌شده
آن را زیرمجموعهٔ غیررسمی گردآوری‌شده توسط جامعه معرفی و در متن مجوز Public Domain, CC-0
اعلام می‌کند؛ فیلد license ساخت‌یافته ندارد. انتشار رسمی Mozilla نامیده نشود.
مراجع اولیه و schema در `configs/surt_mini/pins.json` ثبت‌اند؛ metadata و README در اجرا
دوباره ذخیره می‌شوند. revision فعلی متفاوت، دسترسی ناممکن یا مجوز تغییرکرده موجب توقف است.

| split اصلی | ردیف اعلام‌شده |
| --- | ---: |
| train | 29789 |
| validation | 10676 |
| test | 10676 |

برای هر split فقط صفحات metadata با offsetهای ۰، ۱۰۰، ۵۰۰، ۱۰۰۰ و طول ۵۰ بررسی می‌شوند:
حداکثر ۶۰۰ ردیف، بدون دانلود parquet/corpus یا `train[:N]`. کاندیدها با seed ثابت و هش شناسه
مرتب می‌شوند؛ اولین N ردیف انتخاب نمی‌شود. فقط asset منفرد viewer با hostname و مسیر شامل
revision/split/row مورد انتظار دریافت می‌شود؛ تغییر قرارداد viewer توقف روشن دارد، نه fallback.
پیش از مخلوط‌سازی، ۸ کلیپ از **۸ گویندهٔ متفاوت** به mini train=۴، validation=۲، test=۲
اختصاص می‌یابد. split اصلی صرفاً provenance است و تضمین استقلال فرض نمی‌شود.

حدود هر تلاش: ۲۴ دریافت صوت، ۳۰۰ ثانیه deadline سخت دریافت، ۳۲MiB مجموع بدنهٔ پاسخ‌ها
(metadata و صوت؛ سربار TLS/HTTP در این شمارش نیست)، ۲MiB هر فایل، ۶۴MiB cache ماندگار،
۱–۱۰ ثانیه هر کلیپ و حداکثر ۱۲۰ ثانیه مجموع clean انتخابی. رسیدن به سقف توقف می‌دهد؛
حتی فایل دقیقاً روی سقف بایت ممکن است محافظه‌کارانه رد شود. cache قبلی حذف نمی‌شود.
شکست آمار بودجه و انتخاب جزئی را حفظ می‌کند. فقط برای خطاهای HTTP 502/503/504 در metadata،
همان درخواست حداکثر سه بار با انتظار ۱ و ۲ ثانیه تکرار می‌شود؛ deadline و بودجهٔ بایت از نو
شروع نمی‌شوند. درخواست‌های HTTP و خطاها در `fetch-budget.json` ثبت می‌شوند؛ بدنهٔ پاسخ خطا
خوانده نمی‌شود. برای صوت، خطاهای دیگر یا منبع جایگزین retry خودکار نداریم.

از `audio`, `sentence`, `client_id` استفاده می‌شود؛ شناسهٔ خالی، waveform نامتناهی/ساکت،
rate/channel نامعتبر و مرجع تهی رد می‌شوند. نرخ/کانال واقعی فایل، مدت و تعداد sample ثبت است؛
تبدیل صریح: میانگین کانال‌ها به mono و `scipy.signal.resample_poly` به 16kHz، float32.
gain مخفی، clipping یا بریدن جمله نداریم. شناسهٔ منبع، split اصلی، شناسهٔ هش‌شدهٔ گوینده،
هش فایل دریافت‌شده و waveform استاندارد برای کنترل تکرار و مهاجرت نگه داشته می‌شوند.
`original_sha256` مربوط به asset دریافتی viewer است؛ viewer ممکن است فایل اصلی corpus را
تبدیل کرده باشد و این هش لزوماً هش فایل master منبع نیست.
هویت واقعی گوینده جست‌وجو نمی‌شود. صوت‌ها فقط در working خصوصی برای شنیدن باقی می‌مانند.
`inspection-ledger.json` حتی در شکست، شناسهٔ ردیف‌های بررسی‌شده و گویندگان قابل شناسایی
با هش، همراه همهٔ تلاش‌های دریافت و هش موجود آن‌ها را حفظ می‌کند. هنگام مهاجرت به‌صورت
محافظه‌کارانه تمام این source IDها و گویندگان نیز از test نهایی کنار گذاشته شوند.

## متن، انتقال و قرارداد واقعی

Normalization این مسیر با helper قدیمی `normalize_persian` متفاوت و وابستگی‌ناخواسته‌ای ندارد:
NFC؛ ي/ى→ی، ك→ک؛ حذف اعراب و کشیده؛ ارقام فارسی/عربی→ASCII؛ Latin lowercase؛
حفظ نیم‌فاصلهٔ داخلی؛ punctuation/symbol/control→فاصله؛ collapse فاصله‌ها.
اعداد به گفتار حدسی تبدیل نمی‌شوند. مثلاً `۱۲.۵` پس از حذف punctuation برابر `12 5` است؛
این سیاست سادهٔ آزمون پایپلاین است و نگاشت تلفظ اعداد/تاریخ‌ها محسوب نمی‌شود.
همین normalization در مرجع و شمارش خطا استفاده می‌شود؛ متن خام هم حفظ می‌شود.

BPE فقط از چهار متن واقعی mini-train ساخته می‌شود، درخواست vocab=۱۲۸ با
`hard_vocab_limit=False`، blank=۰، unk=۱، بدون BOS/EOS، normalization داخلی identity.
اگر تعداد نمادهای لازم بیش از بودجه باشد، بدون ساخت متن یا تنظیم با held-out متوقف می‌شود.
اندازهٔ واقعی، نگاشت piece↔ID، طول و unknown و round-trip هر جمله به تفکیک split ذخیره می‌شود.
خطای unknown/round-trip در train توقف است؛ در held-out گزارش می‌شود و tokenizer برای رفع آن
با متن held-out بازآموزی نمی‌شود. پوشش اندک Latin باید محدودیت اعلام شود؛ تست متنی شاهد
تشخیص گفتار code-switching نیست. این سیاست normalization و vocab ادعای tokenizer نهایی ندارد.

checkpoint رسمی Base و BPE انگلیسی 1A مبنا هستند؛ هش هر دو پیش از کار بررسی می‌شود.
۹ tensor وابسته به واژگان شامل embedding، joiner output، CTC و simple AM/LM projections
در `transfer-report.json` ممیزی می‌شوند. نگاشت ردیف‌ها با piece عادی یکسان و نقش special
انجام می‌شود؛ blank انگلیسی ۰ و unk انگلیسی ۲ به blank جدید ۰ و unk جدید ۱ می‌روند.
ردیف‌های دیگر initialization تازه دارند. shape برابر به‌تنهایی انتقال واژگان را مجاز نمی‌کند.
برای همهٔ tensorها action، shape و علت ثبت می‌شود؛ محور واژگانی ناشناخته توقف است.
Adam تازه و FP32، بدون state قبلی؛ فقط یک update روی اولین مخلوط train (batch=۱).

در هر split هر دو کلیپ یک جفت گویندهٔ متفاوت‌اند؛ train دو مخلوط، validation/test یک مخلوط.
offsetها ۰ و ۰٫۵ ثانیه، gain هر دو ۰٫۵؛ هر دو جمله کامل و overlap واقعی اندازه‌گیری می‌شود.
منبع clean، timing و waveform کامل حفظ می‌شود. gain قبل از fbank منبع اعمال می‌شود.
`KaldifeatFbank` خود Lhotse با تنظیمات recipe: 80 bin، 16kHz، 25ms/10ms، Povey،
preemphasis=۰٫۹۷، high_freq=-۴۰۰، dither=۰، snip_edges=False، بدون ضرب ۳۲۷۶۸.
NumPy storage به جای فشرده‌سازی lilcom تنها تفاوت ذخیره‌سازی است.
sourceها بر اساس (start,speaker) concatenate و offsets ذخیره می‌شوند؛ سپس
**K2SurtDataset(return_sources=True, strict=True)** واقعی بارگذاری می‌شود.
طول منبع/مرز زمانی، ترتیب متن هر دو شاخه، overlap و finite بودن features assert می‌شوند.
رفع اختلاف حداکثر دو فریم توسط dataset و convention اصلی `compute_heat_loss`، شامل
افزودن LOG_EPSILON به هدف و mask با `> input_lens`، حفظ و در قرارداد گزارش می‌شوند.
این parity به recipe پین‌شده است؛ اصلاح silent فرمول HEAT صورت نمی‌گیرد.

loss واقعی `train.compute_loss` در batch_idx=۰: simple RNN-T×۱، pruned RNN-T×۰٫۱،
CTC×۰٫۲ و HEAT MSE×۰٫۰۰۱ (فعال‌سازی صریح شاخهٔ کمکی؛ default upstream برای آن صفر است).
HEAT تخصیص زمانی هدف است؛ MSE کمکی را PIT نمی‌نامیم. loss ساختگی نداریم.
raw loss هر سه خروجی پیش از فیلتر upstream باید finite باشد؛ HEAT، gradient، تغییر وزن،
checkpoint کامل و save/reload strict بررسی می‌شوند. سپس validation/test با eval و
inference_mode، whole-batch greedy دوشاخه‌ای و شمارش S/D/I/N و PI-WER محدود decode می‌شوند.
این بازدید held-out جزو تصمیم‌های پایپلاین است و در ledger مهاجرت ثبت می‌شود.
یک update شاهد یادگیری/کیفیت فارسی یا استریم افزایشی نیست.

## سلول ۱ — دریافت کد push‌شده و تعیین run

هر block یک سلول مستقل و `%%bash` اولین خط آن است. `PROJECT_COMMIT` را با SHA کامل
خروجی `git rev-parse HEAD` پس از push خودتان جایگزین کنید. `RUN_ID` نام تلاش است.

```bash
%%bash
set -euo pipefail
export MINI=/kaggle/working/surt-mini-1b
export RUN_ID=mini-20261113-a
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
export MINI_RUN="$MINI/runs/$RUN_ID"
mkdir -p "$MINI_RUN/logs"
CELL_LOG="$MINI_RUN/logs/cell-01-source-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 01-source; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 01-source"; else echo "FAILED: 01-source ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
export PROJECT=/kaggle/working/English-Streaming-ASR
export ICEFALL=/kaggle/working/icefall-surt-base
export PROJECT_COMMIT=PASTE_FULL_PUSHED_COMMIT_SHA_HERE
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
printf '%s\n' "$PROJECT_COMMIT" > "$MINI_RUN/logs/project-commit-$(date -u +%Y%m%dT%H%M%S)-$$.txt"
test -d "$ICEFALL/.git"
python3 "$PROJECT/scripts/surt_base/source.py" "$ICEFALL" "$MINI_RUN/logs/source-$(date -u +%Y%m%dT%H%M%S)-$$"
printf '%s\n' "$MINI_RUN" > "$MINI/active-run.txt"
git -C "$PROJECT" status --short
) > "$CELL_LOG" 2>&1
```

## سلول ۲ — محیط و قرارداد پیشین؛ بدون نصب

```bash
%%bash
set -euo pipefail
MINI=/kaggle/working/surt-mini-1b
MINI_RUN="$(cat "$MINI/active-run.txt")"
CELL_LOG="$MINI_RUN/logs/cell-02-runtime-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 02-runtime; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 02-runtime"; else echo "FAILED: 02-runtime ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_mini/env.sh
python -m pip freeze > "$MINI_RUN/logs/freeze-$(date -u +%Y%m%dT%H%M%S)-$$.txt"
python -m pip check
python "$PROJECT/scripts/surt_base/runtime.py"
python "$PROJECT/scripts/surt_mini/runtime.py" "$MINI_RUN"
nvidia-smi
) > "$CELL_LOG" 2>&1
```

## سلول ۳ — دریافت محدود هشت کلیپ واقعی

خطای `HTTP Error 502: Bad Gateway` در خواندن `/rows` یعنی درخواست metadata پاسخ خطای
سرویس دریافت کرده است؛ در آن نقطه هنوز دریافت صوت شروع نشده است. کد اصلاح‌شده retry محدود
دارد و در تداوم خطا متوقف می‌شود. موفقیت دسترسی از داخل Kaggle باید با اجرای واقعی تأیید شود.
برای تلاش ناموفق قبلی، ابتدا سلول ۶ را با `RUN_ID` قبلی اجرا کنید؛ پوشهٔ `data` و JSONها را
پاک نکنید. پس از push اصلاحات، سلول ۱ را با SHA جدید و مثلاً `RUN_ID=mini-20261114-b`
و سپس سلول ۲ و ۳ اجرا کنید. در سلول ۶ هم نام تلاش جدید را تنظیم کنید. cache مشترک حفظ می‌شود.
تکرار مستقیم سلول ۳ با run قبلی به‌عمد مجاز نیست، چون شواهد آن نباید بازنویسی شوند.

```bash
%%bash
set -euo pipefail
MINI=/kaggle/working/surt-mini-1b
MINI_RUN="$(cat "$MINI/active-run.txt")"
CELL_LOG="$MINI_RUN/logs/cell-03-fetch-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 03-fetch; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 03-fetch"; else echo "FAILED: 03-fetch ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_mini/env.sh
python "$PROJECT/scripts/surt_mini/fetch.py" "$MINI_RUN" "$MINI/download-cache"
test -s "$MINI_RUN/selection.json"
) > "$CELL_LOG" 2>&1
```

## سلول ۴ — BPE فقط train، مخلوط و feature/source/HEAT

```bash
%%bash
set -euo pipefail
MINI=/kaggle/working/surt-mini-1b
MINI_RUN="$(cat "$MINI/active-run.txt")"
CELL_LOG="$MINI_RUN/logs/cell-04-prepare-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 04-prepare; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 04-prepare"; else echo "FAILED: 04-prepare ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_mini/env.sh
python "$PROJECT/scripts/surt_mini/prepare.py" "$MINI_RUN"
) > "$CELL_LOG" 2>&1
```

## سلول ۵ — تنها یک update، reload و decode

```bash
%%bash
set -euo pipefail
MINI=/kaggle/working/surt-mini-1b
MINI_RUN="$(cat "$MINI/active-run.txt")"
CELL_LOG="$MINI_RUN/logs/cell-05-update-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 05-update; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 05-update"; else echo "FAILED: 05-update ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
source /kaggle/working/English-Streaming-ASR/scripts/surt_mini/env.sh
nvidia-smi --id=0 --query-gpu=timestamp,uuid,memory.used --format=csv -l 1 > "$MINI_RUN/logs/gpu-$(date -u +%Y%m%dT%H%M%S)-$$.csv" 2>&1 &
GPU_MONITOR=$!
trap 'kill "$GPU_MONITOR" 2>/dev/null || true; wait "$GPU_MONITOR" 2>/dev/null || true' EXIT
python "$PROJECT/scripts/surt_mini/update.py" "$MINI_RUN"
test -s "$MINI_RUN/result.json"
test -s "$MINI_RUN/transcripts.txt"
) > "$CELL_LOG" 2>&1
```

## سلول ۶ — archive کوچک، حتی در شکست

در شکست محیط نیز این سلول با Python پایه اجرا می‌شود. در شکست سلول ۱، `RUN_ID` این سلول
را برابر نام همان تلاش قرار دهید؛ به active-run تلاش قدیمی تکیه نکنید.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/English-Streaming-ASR
ICEFALL=/kaggle/working/icefall-surt-base
MINI=/kaggle/working/surt-mini-1b
RUN_ID=mini-20261113-a
MINI_RUN="$MINI/runs/$RUN_ID"
test -d "$MINI_RUN/logs"
CELL_LOG="$MINI_RUN/logs/cell-06-archive-$(date -u +%Y%m%dT%H%M%S)-$$.log"
printf 'START 06-archive; full log: %s\n' "$CELL_LOG"
trap 'status=$?; if [ "$status" -eq 0 ]; then echo "OK: 06-archive"; else echo "FAILED: 06-archive ($status)"; tail -n 40 "$CELL_LOG"; fi; exit "$status"' EXIT
(
set -euxo pipefail
python3 "$PROJECT/scripts/surt_mini/archive.py" "$MINI_RUN" "$ICEFALL"
) > "$CELL_LOG" 2>&1
```

بسته در `/kaggle/working/surt-mini-1b/runs/mini-20261113-a-review.tar.gz` قرار می‌گیرد
(در تکرار archive پسوند شماره‌دار دارد). وزن، صوت، cache، venv و credential داخل آن نیستند؛
manifestها، BPE کوچک، loss/gradient/delta، انتقال وزن، محیط، command trace و لاگ کامل،
revisionها، patchها و هش‌ها هستند. فایل‌های 1A فقط خوانده می‌شوند.
آرشیو بیش از ۲۰MiB شواهد خام را بسته‌بندی نمی‌کند و بدون حذف لاگ متوقف می‌شود.

زمان‌های load، feature، update و decode جدا و با CUDA synchronize ثبت می‌شوند.
feature شامل IO/ذخیره/تأیید dataset و update شامل forward/backward/clip/Adam است.
load شامل ساخت مدل، خواندن/انتقال وزن و کپی GPU است؛ هش پیش‌پرواز جداست.
peak allocated/reserved هر مرحله reset می‌شود؛ baseline حافظهٔ مدل و cache قبلی در reserved
ممکن است باقی باشد؛ warmup نداریم. CSV یک‌ثانیه‌ای nvidia-smi کل دستگاه، peak دقیق فرایند نیست.
این اعداد latency استریم یا هزینهٔ آموزش طولانی را ثابت نمی‌کنند.

## gate و مسیر بعدی

برای عبور 1B باید selection بدون نشت، قرارداد source/feature/HEAT، انتقال معنایی همهٔ tensorها،
lossهای واقعی finite، gradient غیرصفر، تغییر وزن، یک step، save/reload strict و هر دو شاخهٔ decode
با مرجع و شمارش خطا در خروجی Kaggle بررسی شوند. شنیدن کلیپ‌های mini و تطبیق مرجع نیز انجام شود.
تا بازگشت archive، وضعیت **pending** است؛ تست محلی موفقیت Kaggle نامیده نمی‌شود.

این subset فقط ورودی موقت فاز صفر و اعتبارسنجی پایپلاین است. منبع برنامه‌ریزی‌شدهٔ آموزش اصلی
Common Voice 27 از Mozilla Data Collective است: فقط train برای مدل و tokenizer، حفظ validation/test.
در مهاجرت، تمام mini مصرف‌شده برای train یا تصمیم‌گیری با speaker/source ID و هش صوت و بررسی
fingerprint مقاوم به re-encoding با دادهٔ جدید مقایسه و از test نهایی کنار گذاشته شود.
BPE mini خودکار tokenizer نهایی نیست؛ استفادهٔ مجدد نیازمند پوشش‌سنجی و هر تغییر نیازمند
ممیزی دوبارهٔ انتقال وزن است. آموزش طولانی و اثبات استریم افزایشی در این گام اجرا نمی‌شوند.
