# راهنمای فارسی اجرای پژوهش SURT در Kaggle

این راهنما مسیر فعال پروژه را توضیح می‌دهد. گزارش‌های Phase قدیمی، تاریخچهٔ توسعه هستند و دستور اجرای فعلی محسوب نمی‌شوند.

## Current execution steps

The old VAD/OSD → router → WhisperRT/SURT runbook has been retired.
The complete current instructions are in this current file.

Summery:

1. Run `english-streaming-asr.ipynb` to prepare and persist real data.
2. Run `notebooks/12_surt_streaming_inference.ipynb` for official asset download
   and explicit native readiness checks. Actual inference requires the missing
   native connection/export and feature/checkpoint validation to be completed.
3. Optionally use `notebooks/13_surt_training_evaluation.ipynb` after an actual
   training backend exists; validate-only does not train or produce weights.
4. Export and validate any fine-tuned checkpoint, repeat inference, then run
   `notebooks/14_surt_evaluation.ipynb` on identical full test sets.

`notebooks/11_whisperrt_baseline.ipynb` is an optional independent comparison.
Training is not required to use a compatible pretrained model. GPU access does
not replace missing model/training integration code.

## ۱. وضعیت واقعی پروژه

این پروژه هنوز یک مدل کاملِ آمادهٔ اجرا با وزن رسمی نیست. بخش ورود صوت، ساخت ویژگی، زمان‌بندی استریم، قرارداد اتصال شبکه، ساخت مخلوط دوصدایی و محاسبهٔ معیارها پیاده‌سازی شده‌اند. اما سه اتصال مهم هنوز باقی است:

1. اتصال checkpoint رسمی به شبکهٔ واقعی با state استریم؛
2. تطبیق ویژگی‌های صوتی و native export با همان checkpoint؛
3. backend واقعی آموزش و تبدیل checkpoint آموزش به export قابل اجرای پروژه.

این‌ها کار برنامه‌نویسی و اعتبارسنجی مدل‌اند. دانلود مدل، اجرای همهٔ نوت‌بوک‌ها یا فعال‌کردن GPU به‌تنهایی آن‌ها را تکمیل نمی‌کند. نوت‌بوک 12 این وضعیت را صریح گزارش می‌کند و بدون مدل قابل اجرا، متن یا نمرهٔ ساختگی تولید نمی‌کند.

پاس‌شدن تست‌های قرارداد، به معنی تأیید دقت مدل یا تأیید سرعت روی GPU نیست.

## ۲. اجزای سیستم و نقش داده‌ها

| بخش | پیاده‌سازی/مدل | داده و ارزیابی | وضعیت |
| --- | --- | --- | --- |
| دریافت صوت | `AudioStream`، تبدیل به mono، نرخ 16 kHz و زمان برحسب نمونه | فایل گفتار واقعی؛ تست پیوستگی و تبدیل کانال | آماده |
| ویژگی صوتی | 80-bin log-Mel؛ پنجرهٔ 25 ms و گام 10 ms | تست عددی/قرارداد؛ انطباق با وزن باید تأیید شود | پیاده‌سازی شده، سازگاری با مدل رسمی تأیید نشده |
| جداکردن مسیرهای گفتار | dual-path mask estimator یا DPRNN در SURT | ویژگی‌های مخلوط دوصدا؛ آموزش انتهابه‌انتها | ماژول آموزش‌دیده/اتصال native هنوز لازم است |
| استخراج نمایش صوت | shared streaming Zipformer | خروجی ویژگی‌های ماسک‌شده؛ حرکت 32 فریم | ماژول آموزش‌دیده/اتصال native هنوز لازم است |
| تبدیل نمایش به متن | stateless predictor + RNN-T joiner | واژگان/BPE همان وزن | composer موجود؛ مدل و tokenizer سازگار لازم است |
| تشخیص پایان گفتار | blankهای RNN-T؛ سکوت پیش‌فرض 800 ms | تست رویداد پایان هر کانال | منطق موجود؛ تنظیم/ارزیابی واقعی لازم است |
| دادهٔ آموزش/آزمایش کوچک | مخلوط‌سازی قطعات واقعی LibriSpeech | متن هر منبع، شروع/پایان، نسبت هم‌پوشانی، جدایی گویندگان | قابل تولید |
| کیفیت تشخیص نمونه‌های دو منبع | PI-WER با شمارش خطا و کلمه | مخلوط‌های کوتاهِ دارای دو متن مرجع | آماده |
| زمان پردازش | RTF و زمان finalize | اجرای فایل؛ جدا از زمان واقعی میکروفن | ثبت می‌شود، اندازه‌گیری GPU هنوز انجام نشده |
| مقایسهٔ اختیاری | WhisperRT | گفتار پاک LibriSpeech، WER/CER | runtime و وزن جداگانه لازم است |
| تشخیص گفتار/هم‌پوشانی مستقل | WebRTC VAD و heuristic OSD | آزمایش‌های جانبی؛ OSD دارای P/R/F1 | در مسیر اصلی SURT استفاده نمی‌شوند |

مسیر اصلی:

```text
صدای انگلیسی 16 kHz
    → ویژگی log-Mel
    → ماسک‌های دو کانال با DPRNN
    → Zipformer مشترک
    → stateless predictor و joiner
    → متن کانال‌ها و رویداد پایان
```

دو کانال به معنی دو هویت ثابتِ گوینده نیست. هدف تشخیص گفتار هم‌زمان است، نه شناسایی اشخاص. WhisperRT، VAD و OSD پیش‌نیاز اجرای این مسیر نیستند.

LibriSpeech مجموعهٔ گفتار انگلیسی خوانده‌شده است. مخلوط‌سازی آن شرایط کنترل‌شدهٔ هم‌پوشانی می‌سازد؛ معادل ضبط یک جلسهٔ واقعی نیست.
LibriCSS بنچمارک جداگانهٔ گفتار پیوسته با شرایط اتاق است. recipe رسمی SURT برای آن ORC-WER با `meeteval` به کار می‌برد. PI-WER فعلی پروژه را نباید به‌عنوان بازتولید همان نتیجه معرفی کرد. داده و ارزیاب زمانیِ LibriCSS هنوز به مسیر فعال پروژه متصل نشده‌اند.

## ۳. کدام کمبودها با اجرای گام قبل برطرف می‌شوند؟

| نیاز مرحلهٔ بعد | آیا خروجی مرحلهٔ قبل کافی است؟ | توضیح |
| --- | --- | --- |
| مخلوط‌های صوتی و manifest | بله | فایل اصلی آن‌ها را می‌سازد؛ خروجی ذخیره و دوباره متصل می‌شود |
| جدایی گویندگان train/validation/test | بله، با اعتبارسنجی | در آماده‌سازی داده و آموزش بررسی می‌شود |
| فایل وزن رسمی و BPE | بله، برای «دریافت فایل‌ها» | نوت‌بوک 12 دانلود و نسخه را ثبت می‌کند |
| اجرای آن وزن در استریم پروژه | خیر | باید اتصال native و انطباق feature/checkpoint پیاده و آزمایش شود |
| تغییر وزن با fine-tuning | خیر | backend واقعیِ training هنوز موجود نیست |
| تولید وزن جدید، پس از تکمیل training backend | بله | آموزش واقعی checkpoint می‌سازد؛ validate-only نمی‌سازد |
| استفاده از checkpoint آموزش در loader استریم | خیر، فعلاً | exporter یا اتصالِ checkpoint به native module هنوز پیاده نشده |
| تبدیل خروجی اجرای واقعی به ورودی ارزیابی | بله | runner فایل `*_evaluation_records.jsonl` می‌سازد |
| گرفتن نتیجهٔ نهایی از مدل | تنها پس از اجرای واقعی | ارزیاب، دادهٔ گمشده یا inference اجرا‌نشده را تولید نمی‌کند |

خروجی WhisperRT وزن، tokenizer یا adapter قابل استفاده برای SURT تولید نمی‌کند؛ فقط مقایسهٔ پژوهشی است. اجرای demo تولید موج سینوسی نیز دیتاست گفتار واقعی یا آموزش مدل تولید نمی‌کند.

## ۴. نوت‌بوک‌های لازم

| ترتیب | فایل | سخت‌افزار | فایل‌های پایتونی اصلی | خروجی |
| --- | --- | --- | --- | --- |
| 1 | `english-streaming-asr.ipynb` | CPU کافی است | `download_librispeech.py`، `prepare_dataset.py`، `generate_overlap.py` | بستهٔ دادهٔ واقعی و manifest |
| 2 | `notebooks/12_surt_streaming_inference.ipynb` | GPU برای اجرا | `check_surt_assets.py`، `run_multitalker.py` | دریافت وزن رسمی، گزارش آمادگی؛ در صورت وجود native export، متن و نمرهٔ مدل پایه |
| 3، اختیاری | `notebooks/13_surt_training_evaluation.ipynb` | GPU برای آموزش واقعی | `train_asr.py` و backend بیرونی | ابتدا اعتبارسنجی؛ پس از تکمیل backend، وزن آموزش‌دیده |
| 2 دوباره، پس از آموزش | همان نوت‌بوک 12 | GPU | `run_multitalker.py` | خروجی وزن fine-tuned روی همان داده |
| 4 | `notebooks/14_surt_evaluation.ipynb` | CPU کافی است | `evaluate.py`، `benchmark.py` | جدول دقت/سرعت و گزارش مقایسه |
| مستقل و اختیاری | `notebooks/11_whisperrt_baseline.ipynb` | GPU برای baseline | `run_baseline.py` | نتیجهٔ جداگانهٔ WhisperRT |

نوت‌بوک‌های 00 تا 10، نسخه‌های تکراری/قدیمی مسیر توسعه بودند و حذف شدند. فایل `12_adaptive_e2e_surt.ipynb` با نوت‌بوک جدید 12 جایگزین شد. فایل‌های VAD/OSD و گزارش‌های تاریخی برای مطالعات جانبی حفظ شده‌اند؛ لازم نیست برای رسیدن به SURT اجرا شوند.

## ۵. اجرای دقیق مرحلهٔ اول

1. ابتدا نسخهٔ اصلاح‌شدهٔ پروژه را در GitHub خود قرار دهید. تغییرات محلیِ کامپیوتر به‌طور خودکار در Kaggle ظاهر نمی‌شوند. این کار در این نوبت به GitHub push نشده است.
2. `english-streaming-asr.ipynb` را در Kaggle باز کنید؛ Internet روشن باشد. برای این مرحله GPU ضروری نیست.
3. سلول اول پروژه را clone می‌کند. اگر نسخهٔ قدیمی در همان جلسه باشد، از آن عبور نمی‌کند؛ یک جلسهٔ تازه با سورس به‌روز استفاده کنید. هیچ `git reset` یا overwrite خودکار برای سورس شما انجام نمی‌شود.
4. نصب و تست زیرساخت اجرا می‌شود. وجود CUDA/PyTorch چاپ می‌شود؛ CUDA پیش‌فرض Kaggle خودکار تعویض نمی‌شود.
5. برای شروع، `LAYOUT="pilot"` را نگه دارید. فقط `dev-clean` واقعی دریافت می‌شود و گویندگان آن به سه گروه جدا تقسیم می‌شوند. این یک آزمایش توسعه است و **نمرهٔ رسمی test-clean نیست**.
6. برای مطالعهٔ اصلی، بعداً `LAYOUT="standard"` را انتخاب کنید: train-clean-100 برای train، dev-clean برای validation و test-clean برای test. استفاده از 100 قطعه و 20 مخلوط در هر بخش همچنان یک زیرمجموعهٔ کوچک است، نه بازتولید آموزش کامل مقاله.
7. اگر داده را از Add Input متصل کرده‌اید، `SOURCE_ROOT` را پوشهٔ شامل `dev-clean` یا سه پوشهٔ رسمی قرار دهید. در این حالت دانلود مجدد لازم نیست.
8. آماده‌سازی، متن و شناسهٔ گویندگان را می‌خواند و سپس مخلوط‌های دوصدایی می‌سازد. seed و برچسب هم‌پوشانی ثبت می‌شود؛ جدایی گویندگان بررسی می‌شود.
9. در پایان، پوشهٔ `/kaggle/working/asr_artifacts` را همراه با outputs در **Save Version** ذخیره کنید.

ساختار داده:

```text
asr_artifacts/
├── data/pilot_v1/
│   ├── audio/train|validation|test/*.wav
│   ├── manifests/train.jsonl
│   ├── manifests/validation.jsonl
│   ├── manifests/test.jsonl
│   ├── dataset_info.json
│   └── statistics.json
├── source_manifests/pilot/
├── generation.yaml
├── session.json
└── pip-freeze.txt
```

مسیر مخلوط‌ها داخل manifest نسبت به بسته ذخیره می‌شود. بنابراین اتصال همان بسته زیر `/kaggle/input/<slug>/...` نباید مسیر صوت مخلوط را خراب کند. مسیر فایل‌های خام منبع فقط provenance است؛ برای baseline روی صوت پاک باید خود دادهٔ خام نیز متصل یا دوباره دریافت شود.

**فایل اصلی، نوت‌بوک بعدی را خودکار اجرا نمی‌کند.** نوت‌بوک 12 را جداگانه با Import Notebook از فایل/مخزن باز کنید. هر نوت‌بوک bootstrap مستقل دارد و وجود همان کرنل قبلی را فرض نمی‌کند.

## ۶. اجرای مرحلهٔ مدل

1. نوت‌بوک `12_surt_streaming_inference.ipynb` را باز کنید. دانلود وزن به GPU نیاز ندارد؛ فقط پس از آماده‌شدن native export، برای probe و اجرای واقعی GPU را فعال کنید.
2. از **Add Input**، خروجی ذخیره‌شدهٔ مرحلهٔ اول را اضافه کنید. مسیر دقیق پوشهٔ `data/pilot_v1` را از فایل‌های Input بردارید و در `DATA_ROOT` بنویسید. نام پوشهٔ ورودی برای هر حساب/نسخه ممکن است فرق کند؛ آن را حدس نزنید.
3. اگر در همان جلسه هستید، مسیر پیش‌فرض `/kaggle/working/asr_artifacts/data/pilot_v1` قابل استفاده است.
4. سلول دانلود، مخزن `desh2608/icefall-surt-libricss-dprnn-zipformer` و شاخهٔ `exp/surt_base` را دریافت می‌کند؛ SHA نسخه و فایل‌های BPE ثبت می‌شوند. برای مدل عمومی، توکن به‌صورت پیش‌فرض لازم نشده است؛ اگر سرویس درخواست دسترسی کرد، آن را با Secrets مدیریت کنید.
5. این وزن‌های رسمی با فایل‌های native loader پروژه یکسان نیستند. با `EXPORT_ROOT=None`، گزارش `native_export_missing` طبیعی و دقیق است. در این نقطه، کار مهندسیِ اتصال مدل مانع ادامه است؛ این مشکل ناشی از تنظیم اشتباه GPU توسط شما نیست.
6. پس از پیاده‌سازی و بررسی اتصال، پوشهٔ native export معتبر را با Add Input متصل و مسیرش را در `EXPORT_ROOT` وارد کنید. قرارداد فایل‌ها در `docs/stages/stage2_report.md` و `models/torch_surt.py` آمده است.
7. probe یک ثانیه سکوت را برای صحت بارگذاری و فراخوانی آزمایش می‌کند. این نتیجه، دقت گفتار یا انطباق وزن/ویژگی را اثبات نمی‌کند.
8. ابتدا با `MANIFEST_SPLIT="validation"` و `MAX_EXAMPLES=1` اجرای کوتاه بگیرید. متن، کانال‌ها و خروجی نهایی را بررسی کنید. سپس تعداد را بیشتر کنید.
9. فایل‌های `*_multitalker.jsonl`، `*_evaluation_records.jsonl` و metrics را ذخیره کنید.

فایل رسمی `pretrained.py` برای آزمودن خود recipe مفید است، اما در نسخهٔ بررسی‌شده ویژگی کل فایل را یکجا محاسبه می‌کند. اجرای آن، به معنی تکمیل رابط `forward_chunk` این پروژه نیست. نام checkpoint رسمی را عوض نکنید تا ظاهراً به native export تبدیل شود.

همچنین featureهای رسمی ممکن است waveform نرمال‌شده با framing انعکاسی/non-snipped به کار ببرند؛ پروفایل فعلی پروژه `kaldi80-snip-pcm16-v1` است. قرار دادن یک برچسب metadata کافی نیست؛ باید انطباق واقعی را بررسی یا feature extractor را با checkpoint تطبیق داد.

## ۷. آیا باید fine-tune کنم؟

برای **استفاده از مدل ازپیش‌آموزش‌دیده**، خیر. ترتیب حداقلی این است:

```text
داده + وزن آماده → اتصال و اجرای مدل پایه → ارزیابی
```

برای **پژوهش دربارهٔ بهبود مدل روی دادهٔ خودتان**، مسیر این است:

```text
اجرای وزن پایه و ثبت نتیجه
    → آماده‌سازی train و validation
    → fine-tuning با backend واقعی
    → انتخاب checkpoint با validation
    → native export و بررسی انطباق
    → اجرای وزن جدید روی همان test
    → مقایسهٔ پایه و fine-tuned
```

نوت‌بوک 13 ابتدا `train_asr.py --validate-only` را اجرا می‌کند؛ این مرحله تنها داده/هدف‌ها را بررسی می‌کند. در وضعیت فعلی، `TRAINING_FACTORY=None` و `RUN_TRAINING=False` است. واقعی‌کردن آموزش به نوشتن و آزمودن backend نسخه‌بندی‌شده نیاز دارد؛ کافی نیست فقط آن متغیرها را تغییر دهید.

backend آموزش باید شبکه، tokenizer، loss مناسب HEAT/pruned RNN-T، optimizer، scheduler، precision و ذخیره/بازیابی را پیاده کند. بخشی از گزینه‌های YAML آموزشی عمومی هنوز توسط Trainer محلی اعمال نمی‌شوند؛ از وجود کلیدهای `gradient_accumulation`، `max_steps` یا precision نباید نتیجه گرفت که آمادهٔ اجرا هستند. Trainer نیز batchها را در حافظه جمع می‌کند و برای آموزش در مقیاس کامل به DataLoader/recipe مناسب نیاز است. resume فعلی در سطح epoch است و ادامهٔ دقیقِ وسط epoch را تضمین نمی‌کند. این‌ها محدودیت مهندسی‌اند و با GPU حل نمی‌شوند.

بعد از آموزش، checkpoint به‌تنهایی فایل `encoder.pt` با رابط native پروژه نیست. exporter هنوز باید ساخته شود. پس از آن مرحلهٔ 12 برای وزن جدید با `SYSTEM_NAME="surt_finetuned"` تکرار می‌شود.

## ۸. ارزیابی نهایی و مراحل پژوهش

1. سوال پژوهشی را ثابت کنید: مثلاً «آیا fine-tuning SURT روی مخلوط‌های انگلیسی، خطای تشخیص هم‌پوشانی را کاهش می‌دهد؟» مدل پایه و مقایسه را پیش از دیدن test تعیین کنید.
2. داده و تقسیم گویندگان را ثبت کنید. تعداد مخلوط‌ها، نسبت هم‌پوشانی، seed، gain و هش manifestها ثابت بمانند.
3. انطباق ویژگی/وزن و اتصال native را تکمیل کنید. با گفتار کوتاه صحت متن و با صوت طولانی حافظه و تأخیر را بررسی کنید.
4. مدل پایه را اجرا کنید. برای توسعه validation؛ برای نتیجهٔ نهایی test استفاده شود.
5. اگر آموزش جزو پرسش پژوهش است، train را برای تغییر وزن و validation را برای انتخاب checkpoint استفاده کنید. test ابزار انتخاب مدل نیست.
6. برای هر دو سیستم، یک اجرای کامل روی test یکسان بگیرید. نوت‌بوک 12: `MANIFEST_SPLIT="test"` و `MAX_EXAMPLES=None`.
7. در نوت‌بوک 14، `RECORD_FILES` را به **یک فایل اجرای کامل برای هر سیستم** تنظیم کنید. فایل smoke یک‌نمونه‌ای و فایل اجرای کامل را باهم ندهید؛ شناسه‌های تکراری رد می‌شوند.
8. نوت‌بوک 14 یکسان‌بودن نمونه‌ها، dataset ID، هش manifest و پوشش کامل test را بررسی می‌کند. سپس `evaluate.py` و `benchmark.py` اجرا می‌شوند.
9. گزارش، خطاهای خام/تعداد کلمات، PI-WER، RTF و زمان finalize را نگه دارید. دادهٔ بدون اندازه‌گیری همچنان `NOT MEASURED` می‌ماند.
10. برای ادعای عملکرد روی جلسهٔ واقعی، LibriCSS و ORC-WER را جداگانه متصل کنید. برای ادعای streaming، تأخیر acquisition، context، endpoint و compute را جدا و روی اجرای واقعی GPU اندازه بگیرید. RTF فایل به‌تنهایی این ادعا را اثبات نمی‌کند.

WhisperRT یک مقایسهٔ اختیاری مستقل است. اجرای clean-WER آن با PI-WER مخلوط‌های دوصدایی مستقیماً مقایسه‌پذیر نیست؛ باید داده و سیاست ارزیابی مناسب همان مقایسه تعریف شود.

## ۹. منابع رسمی بررسی‌شده

- SURT recipe و دستورهای train/adapt/decode: `https://github.com/k2-fsa/icefall/tree/master/egs/libricss/SURT`
- وزن منتشرشده: `https://huggingface.co/desh2608/icefall-surt-libricss-dprnn-zipformer`
- LibriSpeech، تقسیم‌ها و فایل‌های دریافت: `https://www.openslr.org/12`
- انتقال output بین نوت‌بوک‌های Kaggle: `https://www.kaggle.com/docs/notebooks`

نسخهٔ package، GPU، CUDA و سورس باید در هر اجرای واقعی ثبت شود. محدودیت و نوع GPU را از همان حساب و جلسهٔ Kaggle ببینید؛ راهنما به مدل GPU یا سهمیهٔ ثابت وابسته نیست.

## ۱۰. بررسی این اصلاحات

در محیط محلی: `pytest -q` برابر **65 passed, 3 skipped** بود. ساختار JSON و کد پایتون **5/5 نوت‌بوک فعال** بدون اجرای سلول‌ها بررسی شد. تست‌های انتقال داده به مسیر تازه، حفظ شناسهٔ گوینده، اتصال prepare→generate→validate و اتصال inference→EvaluationRecord→evaluate عبور کردند؛ backend تست، مدل گفتار واقعی نیست.

دانلود کامل LibriSpeech/وزن، اجرای نوت‌بوک روی Kaggle، اجرای native مدل واقعی روی GPU، آموزش و صحت متن واقعی در این بررسی انجام نشده‌اند. این نتیجه‌ها را از پاس‌شدن تست محلی استنتاج نکنید.
