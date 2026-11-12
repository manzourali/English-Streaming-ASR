# سازگاری با محیط گزارش‌شدهٔ Kaggle

مبنای این راهنما `kaggle_env_info.md` است. نسخهٔ ماشین‌خوان همان اطلاعات در `configs/kaggle_environment.json` قرار دارد. تاریخ گرفتن گزارش مشخص نیست؛ تاریخ تغییر این راهنما ۲۰۲۶-۱۱-۱۲ است. این اطلاعات، تصویر یک جلسه‌اند و قفل تضمین‌شدهٔ تمام وابستگی‌های پروژه نیستند.

| مؤلفه | مقدار ثبت‌شده | تصمیم پروژه |
| --- | --- | --- |
| سیستم | Linux x86_64، مسیر `/kaggle/working` | مسیرهای خروجی در working؛ ورودی‌های متصل‌شده در input |
| Python | 3.13.15 | کرنل اصلی حفظ می‌شود؛ Conda محلی Python 3.13 را پیشنهاد می‌کند |
| NumPy | 2.1.3 | نصب اولیه نسخهٔ موجود را حفظ می‌کند |
| Torch و torchaudio | هر دو 2.11.0+cu128 | در کرنل اصلی تعویض نمی‌شوند |
| Transformers | 5.16.1 | نسخهٔ موجود حفظ می‌شود؛ پیش‌نیاز مسیر اصلی SURT نیست |
| soundfile | 0.14.0 | برای فایل‌های صوتی؛ import آن در پروفایل kaggle بررسی می‌شود |
| GPU | دو Tesla T4، هرکدام 15360 MiB | هر دو ثبت می‌شوند؛ اجرای عادی SURT روی یک GPU است |
| RAM | حدود 31.3 GiB کل | آزمایش کوچک ابتدا؛ تعداد نمونه‌ها را با مصرف واقعی تنظیم کنید |
| فضای آزاد | 1069 GB در زمان گزارش | اندازه‌گیری لحظه‌ای؛ تضمین سهمیهٔ قابل استفاده یا فضای پایدار نیست |

برچسب `Google Colab` در گزارش با مسیر کگل هم‌خوان نیست. فایل اصلی دست‌نخورده نگه داشته شده است. تشخیص جدید متغیر محیط Kaggle، وجود مسیر `/kaggle/working` و نشانهٔ Colab را جدا ثبت می‌کند؛ نام پلتفرم از وجود یک پکیج نتیجه گرفته نمی‌شود.

## نصب در نوت‌بوک‌های فعال

هر پنج نوت‌بوک setup یکسان دارند و فرمان‌ها را با `sys.executable` کرنل اجرا می‌کنند:

```python
subprocess.run([sys.executable, str(ROOT / "scripts/setup_kaggle.py"),
                "--artifacts", str(ARTIFACTS)], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/check_environment.py"),
                "--profile", "kaggle", "--json",
                "--output", str(ARTIFACTS / "environment.json")], check=True)
```

`setup_kaggle.py` ابتدا نسخهٔ پکیج‌های نصب‌شدهٔ ML و صوت، از جمله Torch، NumPy، Transformers و وابستگی‌های Hub/tokenizer را در constraint ثبت می‌کند. سپس فقط extraهای `kaggle,dev` پروژه نصب می‌شوند. وابستگی‌های جدید باید wheel باینری داشته باشند؛ نبود wheel یا تضاد نسخه باعث توقف می‌شود. این کار از تلاش ناخواسته برای کامپایل وابستگی‌ها در Python 3.13 جلوگیری می‌کند. نسخهٔ نصب‌شده پیش و پس از نصب هم مقایسه می‌شود.

این constraint بر اساس **همان جلسهٔ واقعی** ساخته می‌شود. نصب نسخه‌های گزارش‌شده روی تمام تصاویر آینده اجباری نشده است. اگر کگل تصویر خود را عوض کند، گزارش اختلاف نسخه‌ها را هشدار می‌دهد؛ صرف اختلاف نسخه به معنی خرابی نیست. خطای import پکیج لازم یا نبود GPU در اجرای نیازمند GPU، خطای واقعی است.

در صورت تضاد resolver، خروجی خطا را نگه دارید و سازگاری وابستگی را بررسی کنید. حذف constraint یا ارتقای عمومی پکیج‌ها راه‌حل پیش‌فرض نیست. اگر قبلاً Torch یا NumPy کرنل عوض شده است، یک جلسهٔ تازه با تصویر اصلی شروع کنید. constraint و pip freeze یک نصب خراب را به تصویر اصلی برنمی‌گردانند.

فایل‌های هر اجرا در `/kaggle/working/asr_artifacts`:

- `installed-stack-constraints.txt`: نسخه‌های موجودِ محافظت‌شده قبل از نصب؛
- `environment.json`: Python، executable، زمان UTC، وضعیت importها، CUDA runtime و تمام GPUهای قابل مشاهده؛
- `pip-freeze.txt`: نسخهٔ توزیع‌های نصب‌شده بعد از setup.

گزارش فقط import پکیج‌های لازم برای پروفایل انتخاب‌شده را آزمایش می‌کند. `installed` با `import: not checked` به معنی تأیید سازگاری باینری آن پکیج نیست. TensorFlow و Transformers برای تهیهٔ داده یا اجرای native SURT عمداً import نمی‌شوند.

فرمان‌های دستی پس از نصب:

```bash
python scripts/check_environment.py --profile kaggle --json
python scripts/check_environment.py --profile kaggle --require-gpu
```

`--strict` برای بررسی اختیاری وابستگی‌های مسیرهای جانبی نیز باقی است. برای مسیر اصلی، نبود WebRTC، PEFT، Transformers، k2 یا kaldifeat خطای setup عمومی محسوب نمی‌شود. نیاز به k2/kaldifeat مختص recipe خارجی است. بررسی محیط، مدل را بارگذاری یا دقت آن را تأیید نمی‌کند.

## بهره‌وری روی T4

- `configs/multitalker.yaml` اکنون `runtime.device: auto` دارد. backend بومی PyTorch آن را به CUDA در صورت دسترسی و در غیر این صورت CPU تبدیل می‌کند. نوت‌بوک 12 پیش از اجرای GPU، `--require-gpu` را هم بررسی می‌کند.
- دو کارت حافظهٔ مستقل دارند؛ دو T4 به معنی یک کارت با 30 GiB حافظه نیست. استفاده از GPU دوم نیازمند backend آموزش توزیع‌شده یا تقسیم کار مستقل است. چنین قابلیتی با تغییر YAML ایجاد نشده است.
- برای انتخاب صریح یک کارت در backend بومی یا WhisperRT می‌توانید `device: cuda:0` یا `device: cuda:1` قرار دهید؛ شناسهٔ خارج از تعداد کارت‌های قابل مشاهده با خطا متوقف می‌شود. این انتخاب به‌تنهایی اجرای هم‌زمان روی دو کارت نیست.
- T4 با compute capability برابر 7.5 پشتیبانی بومی BF16 ندارد. درخواست صریح BF16 در مسیر WhisperRT زودهنگام رد می‌شود. FP16 را فقط زمانی استفاده کنید که loader و مدل آن را پشتیبانی کنند؛ حالت پیش‌فرض SURT و export خودکار به half تبدیل نمی‌شود.
- گزینهٔ dtype در adapter فعلی WhisperRT به loader رسمی تبدیل وزن ارسال نمی‌شود؛ از تنظیم آن نباید نتیجه گرفت که precision تغییر کرده است. `actual_dtype` مقدار واقعی را ثبت می‌کند. پردازش آن اکنون در `torch.inference_mode()` انجام می‌شود تا گراف گرادیان برای inference ساخته نشود.
- اندازهٔ batch آموزشی همچنان 1 است. مصرف حافظه و پایداری loss را روی مدل واقعی اندازه بگیرید و سپس batch/accumulation را در backend واقعی تنظیم کنید. دقت و سرعت با سخت‌افزار موجود هنوز اندازه‌گیری نشده‌اند.
- فعال‌سازی سراسری `torch.compile`، AMP یا TF32 انجام نشده است؛ exportهای stateful و پکیج‌های خارجی به آزمون اختصاصی نیاز دارند.

## محیط جداگانهٔ Icefall smoke

`docs/KAGGLE_SURT_RUNBOOK.md` مسیر آزمایشی قدیمی با Python 3.12، NumPy 1.26.4، Torch/torchaudio 2.3.1+cu121 و k2/kaldifeat متناظر دارد. دستورهای فعلی آن، محیط `/kaggle/working/surt-venv-py312` را می‌سازند و در هر سلول فعال می‌کنند. این محیط با `--system-site-packages` به کرنل اصلی وصل نشده است.

اختلاف نسخه‌های این دو محیط عمدی است. wheelهای کامپایل‌شدهٔ آن recipe را در Python 3.13/Torch cu128 نصب نکنید. نسخه‌های جدیدتر k2/kaldifeat نیز بدون تأیید سازگاری recipe جایگزین نشده‌اند. وجود driver و GPU، عبور import و اجرای recipe را تضمین نمی‌کند؛ اجرای smoke باید مستقلاً لاگ و گزارش شود.

محیط `environment.yml` فقط تقریب سبک محلی است؛ آن را داخل کرنل Kaggle اجرا نکنید. این فایل CUDA، k2 یا backend کامل مدل را فراهم نمی‌کند. حداقل Python عمومی پکیج `>=3.10` باقی مانده است چون هسته هنوز با نسخه‌های قبلی سازگار است؛ Python 3.13 محیط هدفِ گزارش‌شده است.

## منابع فنی

- مستند رسمی PyTorch: previous versions، ترکیب Torch/torchaudio 2.3.1 با CUDA 12.1؛ `https://pytorch.org/get-started/previous-versions/`
- مستند رسمی NVIDIA: جدول compute capability کارت T4؛ `https://developer.nvidia.com/cuda/gpus`
- CUDA Programming Guide: جدول انواع دادهٔ Tensor Core برای معماری‌ها؛ `https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html`

این منابع برای تصمیم‌های سازگاری بررسی شدند؛ نسخه‌های دقیق تصویر کاربر از فایل خودش آمده‌اند و با اجرای مستقیم در حساب Kaggle تأیید نشده‌اند.
