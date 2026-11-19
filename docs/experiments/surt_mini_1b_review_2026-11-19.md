# پذیرش فنی 1B و محدودیت‌های نتیجه

تاریخ: ۲۰۲۶-۱۱-۱۹. run واقعی: `mini-20261119-d`.
مرجع محلی بررسی: `outputs/surt-mini-1b/mini-20261119-d/`؛ فایل‌های بزرگ و صوت وارد Git نمی‌شوند.

**gate فنی 1B پذیرفته شد.** کاربر اجرای موفق سلول‌های ۳ تا ۶ Kaggle را گزارش کرد؛
JSONها و transcriptهای ذخیره‌شده خوانده شدند. کاربر در همین گفت‌وگو تأیید کرد هر هشت
clean با مرجع مطابق‌اند و در چهار mixture هر دو گفتار حضور دارند و جمله‌ای قطع نشده است.
عامل مدل را بارگذاری یا صوت را پردازش/پخش نکرد و تستی اجرا نکرد.
فیلد `pending human review` در `result.json` تصویر وضعیت پیش از بازبینی است؛ فایل اصلی
برای حفظ شواهد تغییر داده نشده و پذیرش پس از بازبینی در همین گزارش ثبت می‌شود.

## شواهد

- انتخاب: ۸ منبع، ۸ speaker hash، ۸ هش asset و ۸ هش waveform متفاوت؛ train/validation/test برابر ۴/۲/۲.
- feature contract: batchهای واقعی K2SurtDataset و source shape/boundary برای هر سه split ثبت‌اند.
- tokenizer: ۱۲۸ piece، blank=۰ و unk=۱؛ train بدون unknown و خطای round-trip.
- انتقال: ۵۸۸ state tensor کامل؛ ۹ tensor وابسته به vocabulary با semantic remap؛ سه ردیف مشترک شامل blank، unk و مرز کلمه منتقل شده و ردیف‌های دیگر تازه مقداردهی شده‌اند. Adam تازه است.
- loss ثبت‌شده: کل 699.046997؛ simple RNN-T برابر 489.468567، pruned برابر 97.369736، CTC برابر 993.383179 و HEAT برابر 1164.765503؛ ضرایب طبق قرارداد batch صفر هستند.
- norm گرادیان پیش از clipping برابر 7890.222168؛ سقف clipping برابر ۵. gradient غیرصفر و delta در mask_encoder، encoder، joint_encoder_layer، decoder، joiner، CTC و simple AM/LM ثبت شده‌اند؛ مجموعاً ۴۴۳ پارامتر تغییر کرده‌اند.
- `optimizer_steps=1`، `strict_reload=true`؛ update حدود 1.264601 ثانیه. peak allocated برابر 558542336 بایت و reserved برابر 788529152 بایت برای همین update است؛ warmup نداشته است.
- هش checkpoint محلی با `result.json` یکسان است: `b9682169c965c7980868b3c950ebb8b84fc8d7718cbcb346a72ca09aae974530`.
- هش tokenizer محلی نیز مطابق نتیجه است: `af7642a00393f2772fc83c9f403b542e545b542dfc9ff7fca54548ee6001d085`.

## چیزی که پذیرفته نشده است

PI-WER دوشاخه‌ای validation برابر ۱۲ خطا از ۱۲ کلمه و test برابر ۱۳ از ۱۳ است؛ هر دو ۱۰۰٪.
خروجی validation «ست» و «ویی» و test «ست» و «ست» است. این مرحله مسیر محاسبات فارسی را
آزموده، نه مدل فارسی با کیفیت قابل استفاده. از یک update، موفقیت یا شکست adaptation نهایی
نتیجه گرفته نمی‌شود. در validation سه unknown از ۴۳ token و دو شکست round-trip وجود دارد؛
پوشش لاتین این هشت نمونه صفر است. BPE mini برای آموزش نهایی تأیید نشده است.

decode موجود whole-utterance است. استریم با state، resume کامل optimizer، هزینهٔ آموزش
بلند، کیفیت FastMSS، ORC-WER جلسه و هویت ثابت دو شاخه هنوز اثبات نشده‌اند.
تمام exposure/inspection ledgerهای تلاش‌های موفق و ناموفق برای حذف برخورد با test نهایی
حفظ شوند. منبع mini فقط subset موقت غیررسمی است؛ منبع برنامه‌ریزی‌شدهٔ آموزش اصلی
Common Voice 27 از Mozilla Data Collective با بررسی دسترسی/شرایط در گام داده است.

## اقدام بعدی

[راهنمای 1C](../KAGGLE_SURT_STREAMING_RUNBOOK_FA.md): بازیابی همین checkpoint/featureها؛
ممیزی دقیق API و نمونهٔ مهندسی feature/state اجزا؛ اندازه‌گیری منابع با warmup.
gate استریم تا adapter کامل و شواهد end-to-end باز می‌ماند. مدل و صوت اصلی دوباره دانلود
نمی‌شوند؛ محیط مفقود باید با pinهای قبلی بازسازی یا از wheelhouse خصوصی نصب شود.
