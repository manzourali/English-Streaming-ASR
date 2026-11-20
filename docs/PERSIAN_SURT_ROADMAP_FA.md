# مسیر اجرایی SURT 2.0 برای فارسی

تاریخ: ۲۰۲۶-۱۱-۱۳. هدف: تطبیق زبانی SURT 2.0 برای رونویسی پیوستهٔ دو فارسی‌زبان، شامل گفتار هم‌زمان. دو شاخهٔ خروجی، هویت ثابت افراد در کل جلسه را تضمین نمی‌کنند. نام پوشه و پکیج قدیمی پروژه برای جلوگیری از شکستن نصب فعلاً حفظ شده است.

## وضعیت کنونی

به‌روزرسانی ۲۰۲۶-۱۱-۲۰: سورس و شواهد 1C-A بررسی و adapterهای 1C-B محلی آماده شده‌اند؛
اجرای 1C-B هنوز انجام نشده و gate باز است. مسیر اجرای فعلی:
[runbook مستقل 1C-B](KAGGLE_SURT_STREAMING_1CB_RUNBOOK_FA.md) و
[قرارداد state/context/EOF](SURT_1CB_STATE_CONTRACT.md).

به‌روزرسانی ۲۰۲۶-۱۱-۱۹: **گام‌های 1A و 1B از نظر فنی پذیرفته شده‌اند؛ gate استریم 1C باز است.**
run واقعی `mini-20261119-d` یک update، gradient/delta، reload strict و decode دوشاخه‌ای
را ثبت کرده و کاربر تطبیق هشت clean و چهار مخلوط کامل را تأیید کرده است.
[گزارش پذیرش و محدودیت‌های 1B](experiments/surt_mini_1b_review_2026-11-19.md).
PI-WER mini برابر ۱۰۰٪ و tokenizer دارای unknown در validation است؛ کیفیت فارسی هنوز
اثبات نشده است. مسیر جاری [runbook 1C](KAGGLE_SURT_STREAMING_RUNBOOK_FA.md) است:
بخش A ممیزی API/feature/state اجزا و منابع؛ بخش B اتصال native و شواهد استریم کامل.
موفقیت بخش A به‌تنهایی gate 1C را نمی‌بندد. فایل [master_prompt.md](../master_prompt.md)
قواعد دائمی اجرای فقط روی Kaggle و بازیابی session را ثبت می‌کند.

**گام 1A پذیرفته شده؛ امکان‌سنجی کامل فارسی هنوز تکمیل نشده است.** پس از آزمون زیرساخت مصنوعی، بستهٔ واقعی `outputs/surt-base-1a-review` بررسی شد: هش ۴۴ فایل، بارگذاری strict وزن رسمی و decode دو کلیپ انگلیسی و مخلوط معتبرند؛ کاربر شنیدن و تطبیق مرجع/دو شاخه را در ۲۰۲۶-۱۱-۱۳ تأیید کرد. PI-WER دو clean برابر ۰/۸ و ۱/۱۲ و مخلوط ۴/۲۰ است. این whole-batch decode است. [شواهد و آماده‌سازی 1B](experiments/surt_mini_1b_preparation_2026-11-13.md).

## ارزیابی پرامپت پیشنهادی

ترتیب کلی پیشنهادی مناسب است، با این اصلاح‌ها:

- قبل از ساخت دادهٔ بزرگ، یک زیرمجموعهٔ بسیار کوچک فارسی برای آزمون امکان‌سنجی لازم است؛ گردآوری در مقیاس اصلی بعد از عبور از gate انجام شود.
- استفاده از مسیر اصلی Icefall برای baseline و adaptation اولویت دارد. تکمیل exporter اختصاصی این مخزن پیش‌نیاز baseline انگلیسی نیست؛ برای ادعای استریم افزایشی، رابط واقعی state همچنان باید ساخته/آزموده شود.
- FastMSS طبق README رسمی ورودی word-aligned می‌خواهد. کلیپ کامل برای مخلوط‌ساز ساده مناسب است، اما عبور مستقیم همان کلیپ‌ها به FastMSS بدون alignment یا تغییر نسخه‌ثبت‌شدهٔ ورودی، فرض مجاز نیست.
- عوض‌شدن معنای token ID حتی با تعداد واژگان یکسان، انتقال مستقیم embedding/خروجی‌ها را معتبر نمی‌کند. انتقال vocabulary باید از نظر معنا و شکل بررسی شود؛ گزارش صرفاً shape-matching کافی نیست.
- HEAT تخصیص هدف است و با PIT یکسان نیست. loss و وزن ضرایب باید از نسخهٔ دقیق recipe استخراج شوند، نه از نام عمومی گزینه‌ها.
- Latin-preserving normalization/tokenizer شرط لازم برای کلمات انگلیسی است؛ کیفیت code-switching آزمایش فرعی و نیازمند صوت واقعی همان واژه‌هاست.

## گام‌ها و معیار عبور

| گام | کار محدود و قابل بررسی | معیار عبور / خروجی |
| --- | --- | --- |
| ۰ — زیرساخت | همان runbook مصنوعی اجراشده | موفق بر اساس شواهد کاربر؛ archive و metadata حفظ شوند |
| ۱A — baseline انگلیسی | وزن رسمی **SURT Base** و BPE همراه همان وزن؛ چند صوت واقعی انگلیسی و یک مخلوط واقعی در recipe پین‌شده | بارگذاری معتبر، متن‌های قابل بازرسی، امتیاز بر اساس مرجع، ثبت نوع decode و منابع؛ کیفیت را از قبل تضمین نکنید |
| ۱B — mini فارسی | تعداد محدود کلیپ فارسی مجاز با گوینده/متن؛ normalization و BPE فقط روی train؛ انتقال وزن با واژگان تازه؛ feature/source-target و یک forward/backward/update | loss و گرادیان متناهی، تغییر وزن، save/reload، نبود خطای token/HEAT/mask؛ ثبت VRAM و ثانیه بر گام |
| ۱C — استریم و هزینه | نمونهٔ کوتاه سپس بلند؛ تغذیهٔ chunk با state پایدار و مقایسه با مرجع مطابق تعریف recipe | state محدود، EOF درست، گزارش context/تأخیر/RTF با همگام‌سازی CUDA؛ اگر فقط whole-batch کار می‌کند، gate استریم باز است |
| ۲ — دادهٔ اصلی | انتخاب نسخه/مجوز، split پیش از مخلوط‌سازی، قواعد ثابت فارسی و لاتین، آمار ساعت/گوینده/کیفیت | train/validation/test بدون اشتراک گوینده یا کلیپ/نسخهٔ تکراری؛ هش manifest و قواعد متن |
| ۳ — شبیه‌سازی | ابتدا ساده؛ سپس FastMSS پین‌شده با alignment معتبر؛ دقیقاً دو گوینده؛ حفظ clean sources و timing | نسبت هم‌پوشانی اندازه‌گیری‌شده، source-featureهای هم‌تراز و قرارداد قابل بارگذاری در K2SurtDataset؛ نویز/RIR مرحله‌ای |
| ۴ — adaptation | ابتدا پایدارکردن بازشناسی فارسی با انتقال معتبر؛ سپس دوشاخه‌ای روی مخلوط؛ انتخاب checkpoint فقط با validation | گزارش وزن‌های منتقل/ریست‌شده، loss واقعی recipe، resume و بودجهٔ اندازه‌گیری‌شده؛ آزمایش ساده و FastMSS با بودجهٔ قابل مقایسه |
| ۵ — ارزیابی | ORC-WER روی test ثابت برای هر سطح overlap؛ code-switching مستقل و اختیاری | خروجی گفتار واقعی، شمارش خطا/کلمات، پروتکل normalization/segmentation ثابت؛ PI-WER کوتاه را ORC-WER ننامید |
| ۶ — جمع‌بندی | مدل/recipe تطبیق‌یافته، تولید داده، جدول آزمایش‌ها و محدودیت‌ها | ادعا فقط متناسب با داده، کیفیت و streaming اندازه‌گیری‌شده |

هر گام به خروجی گام قبل وابسته است. یک اجرای کوتاه نمی‌تواند کیفیت نهایی یا هزینهٔ کل آموزش را تضمین کند. ساخت دادهٔ انبوه و fine-tuning طولانی پیش از روشن‌شدن ۱A–۱C برنامه‌ریزی نشده‌اند. آزمون overfit احتمالی فقط روی train انجام شود و با نتیجهٔ held-out اشتباه نشود.

## داده و منابع

Common Voice فارسی می‌تواند منبع اولیه باشد، اما ابتدا نسخهٔ انتخاب‌شده، شرایط دسترسی/بازنشر و ساعتِ واقعاً انتخاب‌شده را ثبت کنید. نسخهٔ ۲۷ در صفحهٔ رسمی ۳۷۳٫۱۵ ساعت validated دارد؛ این مقدار برابر train آماده نیست. آن صفحه بازنشر داده و تلاش برای تعیین هویت گویندگان را ممنوع می‌کند؛ صوت را در Git یا خروجی عمومی Kaggle بازنشر نکنید. شناسهٔ ناشناس گوینده برای کنترل split کافی است.

برای mini فارسی فقط زیرمجموعهٔ غیررسمی `aliyzd95/common_voice_22_0_fa` با revision پین‌شده و هشت کلیپ محدود در Kaggle استفاده می‌شود. این ورودی موقت فاز صفر است؛ منبع برنامه‌ریزی‌شدهٔ آموزش اصلی Common Voice 27 از Mozilla Data Collective خواهد بود، فقط train برای آموزش مدل/BPE و validation/test محفوظ. تمام mini مصرف‌شده در آموزش یا تصمیم‌گیری هنگام مهاجرت از نظر speaker/source ID/audio با test نهایی تطبیق و حذف تداخل شود. BPE mini خودکار tokenizer نهایی نیست؛ پوشش‌سنجی و با هر تغییر ممیزی دوبارهٔ وزن لازم است. Neyshekar و code-switching وارد این گام نمی‌شوند.

## وضعیت ابزارهای موجود

آماده‌سازی محلی گام ۱A در [runbook مستقل baseline واقعی انگلیسی](KAGGLE_SURT_BASE_ENGLISH_RUNBOOK_FA.md)
قرار دارد: وزن رسمی Base و BPE پین‌شده، دو کلیپ واقعی و یک مخلوط، whole-batch decode و PI-WER محدود.
اجرای واقعی 1A و 1B و تأییدهای شنیداری بررسی شده‌اند. [runbook 1B](KAGGLE_SURT_MINI_PERSIAN_RUNBOOK_FA.md)
برای سابقه و بازتولید حفظ می‌شود؛ مسیر جاری [runbook 1C](KAGGLE_SURT_STREAMING_RUNBOOK_FA.md) است.
برای 1C از پوشهٔ کامل run موفق بازیابی کنید؛ اجرای مجدد فاز صفر یا 1A لازم نیست.

چهار نوت‌بوک باقی‌مانده ابزار داده/قرارداد native export هستند و هنوز recipe کامل adaptation فارسی نیستند. `train_asr.py --validate-only` آموزش انجام نمی‌دهد. adapter دادهٔ فعلی هم به‌تنهایی source-featureهای هم‌تراز مورد نیاز recipe را نمی‌سازد. هستهٔ audio، mixture، manifest، معیار و state حفظ شده است؛ مدل قبلی و router حذف شده‌اند. VAD/OSD موجود ابزار تشخیصی عمومی‌اند و ورودی اجباری مسیر SURT نیستند.

## پرامپت‌ها و آرشیو

- [runbook جاری: 1C و بازیابی session](KAGGLE_SURT_STREAMING_RUNBOOK_FA.md)
- [پرامپت تاریخی baseline ۱A](prompts/PERSIAN_SURT_NEXT_STEP.md)
- [پرامپت‌های ادامه پس از عبور از gate](prompts/PERSIAN_SURT_FOLLOWUPS.md)
- [آرشیو یک‌فایلی لاگ‌ها و قواعد Git](KAGGLE_SURT_ARTIFACTS_FA.md)

## منابع اولیهٔ بررسی‌شده

- SURT 2.0، مقاله و مسیر رسمی: https://arxiv.org/abs/2306.10559 و https://sites.google.com/view/surt2
- recipe و قرارداد HEAT/ORC-WER: https://github.com/k2-fsa/icefall/tree/f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1/egs/libricss/SURT (revision راهنما؛ تغییر sampler جدا ثبت شود)
- وزن رسمی: https://huggingface.co/desh2608/icefall-surt-libricss-dprnn-zipformer
- FastMSS و الزام alignment: https://github.com/popcornell/FastMSS
- شرایط و آمار Common Voice 27 فارسی: https://mozilladatacollective.com/datasets/cmu5xazgy00finq074vaqo1b3
