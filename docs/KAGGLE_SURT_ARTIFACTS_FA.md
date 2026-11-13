# نگهداری خروجی‌های SURT در Kaggle

نیازی به دانلود تک‌تک لاگ‌ها یا نگه‌داشتن محیط ۵ گیگی در Git نیست. لاگ‌های متنی را یک‌بار در فایل فشرده جمع کنید و همان یک فایل را از پنل Output/Files بگیرید. این سلول در کرنل اصلی هم اجرا می‌شود و وابستگی مدل ندارد:

```bash
%%bash
set -euo pipefail
WORK=/kaggle/working
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$WORK/surt-smoke-logs-$STAMP.tar.gz"
test -d "$WORK/surt-smoke-logs"
tar -czf "$ARCHIVE" -C "$WORK" surt-smoke-logs
sha256sum "$ARCHIVE"
ls -lh "$ARCHIVE"
```

فایل `surt-smoke-logs-....tar.gz` را از لیست فایل‌های working دانلود کنید. اگر جلسه تمام شده، خروجی نسخهٔ ذخیره‌شده را با Add Input متصل کنید و مسیر دقیقِ پوشهٔ logs در آن ورودی را جایگزین مسیر ثابت کنید؛ نام slug را حدس نزنید. فایل archive را در working بسازید. اگر خروجی نسخه فقط پوشهٔ logs را دارد و checkpoint را ندارد، لاگ به‌تنهایی جای checkpoint را نمی‌گیرد.

برای استخراج محلی، از ابزار Archive سیستم یا `tar -xzf <archive>` در پوشهٔ انتخابی استفاده کنید. دانلود روی Mac برای ادامهٔ کار داخل Kaggle ضروری نیست؛ نسخهٔ ذخیره‌شده را می‌توان به جلسهٔ بعد متصل کرد. داشتن یک نسخهٔ کوچک مستقل از لاگ‌ها برای سابقهٔ پژوهش مفید است.

## چه چیزی کجا بماند؟

| مورد | محل مناسب |
| --- | --- |
| خلاصهٔ آزمایش، revisionها و patch، فرمان‌ها، نسخه‌ها و هش manifest/checkpoint/archive | Git؛ نمونه در `docs/experiments/surt_smoke_2026-11-12.md` |
| لاگ کامل | output ذخیره‌شدهٔ Kaggle و در صورت تمایل archive محلی؛ الزام به commit همهٔ لاگ‌ها نیست |
| checkpoint و داده | output/input یا ذخیره‌سازی مناسب مدل/داده؛ نه Git معمولی |
| venv، checkout کپی‌شدهٔ Icefall، cacheهای pip/uv/HF و محیط چندگیگی | وارد Git نشوند؛ cache یا ابزار قابل بازسازی‌اند |

checkpoint مصنوعی فعلی برای شاهد اجرای زیرساخت مفید است؛ وزن مناسب شروع adaptation فارسی نیست. برای گام بعد از checkpoint رسمی SURT Base استفاده خواهد شد. نگهداری فایل‌های محیط مجازی هم تضمین بازسازی همان session نیست؛ فایل وابستگی‌ها، patchها و hashes را حفظ کنید.

در Git این پروژه فایل‌های وزن و archiveهای بزرگ ignore شده‌اند. سابقهٔ آزمایش فعلی بر اساس فایل decode ارائه‌شده و تأیید پیام checkpoint توسط کاربر ثبت شده؛ هش‌هایی که هنوز دریافت نکرده‌ایم به‌صورت عدد ساختگی وارد نشده‌اند. پس از archive، hash و شناسهٔ نسخهٔ Kaggle را به سابقه اضافه کنید.
