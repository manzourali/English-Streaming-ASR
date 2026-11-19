**گزارش ممیزی امکان‌سنجی SURT 2.0 فارسی — ۱۹ نوامبر ۲۰۲۶**

دامنه: بازبینی کد موجود، به‌ویژه `docs/prompts/PERSIAN_SURT_FOLLOWUPS.md` و گام‌های ۱A، ۱B و ۱C. این گزارش پیاده‌سازی آموزش کامل یا FastMSS را مطالبه نمی‌کند. مبنای checkout، commit برابر `bc628086b685c191394dd5e8881b7336767a4fca` است؛ هنگام شروع ممیزی درخت Git تمیز بود. هیچ کد اجرایی، داده یا checkpoint در این بررسی تغییر نکرد.

**۱. تصمیم فنی و سطح شواهد**

پروژه برای اجرای یک آزمون محدود انتقال زبان و یک update واقعی، طراحی قابل دفاعی دارد؛ اما امکان‌سنجی فارسی هنوز اثبات نشده است. مانع معماری ذاتی برای آزمایش ۱B از بررسی ایستا پیدا نشد. دو مانع قطعی برای استفادهٔ مستقیم از مسیر عمومی به‌عنوان پایپلاین نهایی وجود دارد: هدف‌گذاری HEAT در حالت بدون overlap، و ناسازگاری front-end/state/branch tying با checkpoint در مسیر استریم. FastMSS هنوز پیاده‌سازی نشده و نباید نتیجهٔ آن به مخلوط‌ساز ساده نسبت داده شود.

| گام / قابلیت | شواهد بررسی‌شده | نتیجه |
| --- | --- | --- |
| ۱A انگلیسی | هر ۴۴ هش بستهٔ `outputs/surt-base-1a-review` دوباره بررسی شد؛ هیچ اختلافی نبود. گزارش checkpoint شامل ۵۹۷ tensor، تعداد ۲۶٬۶۷۵٬۶۳۹ پارامتر و نبود missing/unexpected/shape mismatch است. | baseline محدود طبق artifact موجود پذیرفته است؛ مدل در این ممیزی دوباره اجرا نشد. |
| ۱B فارسی | `surt_mini` انتخاب ۸ کلیپ، BPE فقط train، انتقال واژگان و یک update واقعی را کدنویسی کرده است. در شواهد محلی، نتیجهٔ موفق اجرای فارسی پیدا نشد؛ یادداشت موجود شکست metadata با HTTP 502 در ۱۴ نوامبر را ثبت کرده است. | آمادهٔ آزمون؛ عبور نکرده است. |
| ۱C استریم | wrapper عمومی و تست‌های مصنوعی وجود دارند؛ export معتبر و شواهد stateful مدل واقعی موجود نیست. | باز؛ نیازمند اصلاح و آزمون مدل واقعی. |
| FastMSS / RIR / noise | کد اتصال، revision و ورودی word alignment در مسیر اجرایی موجود نیست. | کار آتی گام ۳؛ نبود آن مانع ۱B نیست. |

اعتبارسنجی محلی: `python3 -m pytest -q` با **۱۱۷ passed، ۶ skipped، در ۱۰٫۹۷ ثانیه** پایان یافت. Python محلی 3.12.0 است و Torch، k2، Lhotse، kaldifeat، SentencePiece، SciPy و Hazm در این interpreter نصب نیستند. تست‌های mini از doubles برای APIهای tokenizer و tensor استفاده می‌کنند. این نتیجه، اجرای واقعی fbank، autograd، CTC یا RNN-T را تأیید نمی‌کند.

برای تطبیق قرارداد، فایل‌های اصلی Icefall در revision پین‌شده `f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1` خوانده شدند. SHA256 فایل‌های `train.py`، `model.py`، `local/add_source_feats.py` و `local/compute_fbank_lsmix.py` با pins مخزن تطابق داشت. قرارداد `K2SurtDataset` نیز از سورس رسمی Lhotse نسخهٔ 1.33.0 بررسی شد. نسخهٔ جاری master جایگزین نسخهٔ پین‌شده نشد.

**۲. داده، شبیه‌سازی و FastMSS**

**یافتهٔ D1 — مهم برای HEAT آینده: gain مؤثر مخلوط‌ساز عمومی در manifest ثبت نمی‌شود.**

مرجع: `src/streaming_asr/datasets/overlap_generator.py:68–89` و `datasets/manifests.py:20–30`.

در حالت پیش‌فرض `normalize_rms=True`، هر منبع بر RMS خودش تقسیم می‌شود؛ سپس gain نسبی B و در پایان ضریب مشترک مهار peak اعمال می‌شوند. `OverlapSource.gain_db` فقط صفر و gain درخواستی B را ذخیره می‌کند. در نتیجه، صوت منبع اصلی و gain ثبت‌شده برای بازسازی mixture یا استخراج هدف mask هم‌دامنه کافی نیستند. در یک probe با دو سینوس ۱ثانیه‌ای و دامنه‌های ۰٫۲ و ۰٫۰۵، اختلاف بیشینهٔ mixture ذخیره‌شده با بازسازی از source و gain ثبت‌شده حدود **۰٫۷۴۹** بود؛ هر دو gain ثبت‌شده صفر بودند. این یک آزمون عددی قرارداد است، نه شاهد ASR.

اصلاح: `applied_gain_linear` هر منبع، ضریب مشترک peak، روش resample، offset در نمونه و hash صوت تبدیل‌شده را ذخیره کنید، یا contribution واقعی هر source را بنویسید. با `a' = a/rms_a`، ضرایب مؤثر عبارت‌اند از `peak_scale/rms_a` و `peak_scale * 10**(relative_gain_db/20)/rms_b`. برای حالت RMS خاموش، مخرج یک است؛ سکوت باید سیاست صریح داشته باشد. بازسازی را روی PCM float پیش از quantization یا با tolerance متناسب با encoding مقایسه کنید.

مسیر `scripts/surt_mini/prepare.py:89–119` این نقص خاص را ندارد: gains برابر ۰٫۵ صریح‌اند، همان gain قبل از fbank منبع اعمال می‌شود و پاک‌منبعها حفظ شده‌اند. بااین‌حال این انتخاب هدف، باید به‌عنوان قرارداد mini ثبت بماند؛ یکسان‌بودن extractor به‌تنهایی اثبات نمی‌کند تمام augmentation/targetهای recipe اصلی عیناً بازتولید شده‌اند.

**یافتهٔ D2 — متوسط: تعریف overlap بین دو مسیر متفاوت است و activity واقعی اندازه‌گیری نشده است.**

در generator عمومی نسبت برابر `overlap_seconds / min(source_durations)` است. در mini، `overlap_fraction` برابر `overlap_seconds / mixture_duration` است. هر دو از نظر ریاضی معتبرند، ولی قابل جایگزینی نیستند. برای دو کلیپ ۲ثانیه‌ای با offset برابر ۰٫۵ ثانیه، نسبت‌ها ۰٫۷۵ و ۰٫۶ هستند. همچنین طول کلیپ شامل سکوت احتمالی ابتدا، انتها و داخل آن است؛ overlap فعلی، اشتراک بازهٔ کلیپ است و الزاماً اشتراک گفتار فعال نیست.

اصلاح: هر دو کمیت را با نام و مخرج صریح ذخیره کنید؛ گروه‌بندی آزمایش‌ها را با تعریف واحد انجام دهید. هنگام نیاز به speech overlap از word/activity alignment استفاده کنید و تعریف سنجه را مستند نگه دارید. offset ثابت ۰٫۵ثانیه برای smoke test مناسب است، اما شبیه‌ساز turn-taking یا توزیع چند سطح overlap نیست.

**یافتهٔ D3 — مهم برای مقایسهٔ علمی، خارج از gate ۱B: gain و regime در generator عمومی مخدوش‌کننده‌اند.**

در `generate_dataset`، regime و gain هر دو از `index % count` انتخاب می‌شوند. با چهار regime و دو gain در `configs/overlap_dataset.yaml`، low/high همیشه gain صفر و medium/control همیشه gain منفی ۳ دریافت می‌کنند. بنابراین اثر gain از اثر regime جدا نمی‌شود. برای گام ۳ از طرح متوازن regime×gain یا sampling مستقل و ثبت seed استفاده کنید؛ برای baseline/FastMSS pool، مدت، speakers، تعداد mixtures و budget را کنترل کنید.

`relative_gain_db` خودبه‌خود SNR نویز نیست: با RMS normalization، نسبت سطح دو منبع را کنترل می‌کند؛ با normalization خاموش، انرژی خام دو منبع هم مؤثر است. mini هم gain برابر دارد، نه الزاماً سطح انرژی برابر. برای SNR هدف از توان منبع و نویز با تعریف ثابت استفاده کنید و منبع سکوت را رد کنید. اکنون noise/RIR وجود ندارند؛ افزودن آن‌ها پس از سلامت clean baseline با نقشهٔ راه سازگار است. در نسخهٔ آینده، نوع هدف clean/reverberant، تأخیر RIR، tail و مرجع زمانی باید صریح باشند؛ نباید زمان‌بندی متن و feature هدف بی‌صدا جابه‌جا شود.

**یافتهٔ D4 — مهم برای گام ۳: FastMSS هنوز یک dependency اجرایی نیست.**

adapter عمومی در `datasets/icefall_adapter.py` فقط schema داخلی را به recording/supervision/cut تبدیل می‌کند. parser خروجی FastMSS، pin کد آن، alignment کلمه، speaker-turn و تولید source features در این adapter وجود ندارد. README رسمی FastMSS ورودی word-aligned می‌خواهد؛ فرستادن مستقیم کلیپ‌های Common Voice با متن جمله، قرارداد معادل نیست.

اقدام متناسب با دامنه: برای ۱B همین مخلوط ساده را نگه دارید. برای گام ۳ ابتدا نمونه‌ای کوچک با word alignment و دقیقاً دو گوینده، mapping شناسه‌ها و زمان‌ها، provenance و load واقعی در `K2SurtDataset` بسازید. فایل‌های adapter خروجی Lhotse هستند؛ export مستقیم Kaldi مانند `wav.scp/text/utt2spk/segments` تولید نمی‌کنند و نباید با نام Kaldi manifest معرفی شوند.

**۳. نشت داده و متن فارسی**

مسیر mini نقاط قوت مشخصی دارد: clip و speaker و دو نوع hash را قبل از mixing کنترل می‌کند؛ هر clip از speaker متفاوت است؛ ۴/۲/۲ کلیپ به splitها اختصاص می‌دهد؛ BPE فقط چهار متن train را می‌بیند؛ raw text، original split، revision، conversion و exposure ledger حفظ می‌شوند. تخصیص دوبارهٔ splitهای اولیه برای این mini در کد و مستندات اعلام شده است؛ آن را benchmark رسمی Common Voice ننامید. `assert_disjoint` برای mini حتی تکرار درون split را هم رد می‌کند؛ این سخت‌گیری را بدون تغییر به دادهٔ اصلی چندکلیپی هر speaker تعمیم ندهید.

در مسیر عمومی، `check_speaker_disjointness` شناسهٔ مفقود را رد نمی‌کند؛ `training.data.leakage_errors` مسیر mixture را می‌سنجد و hash محتوای clean source را بررسی نمی‌کند. همچنین `iter_librispeech`، `client_id` Common Voice را به speaker تبدیل نمی‌کند. بنابراین صرف تغییر dataset در config عمومی، pipeline فارسی امن ایجاد نمی‌کند. ورود speaker/source ID خالی باید خطا باشد؛ جداسازی را بر clean source ID، speaker hash و hash صوت canonical پیش از augmentation انجام دهید. hash بایت‌به‌بایت، نسخهٔ re-encoded همان گفتار را لزوماً پیدا نمی‌کند؛ مهاجرت به corpus نهایی به بررسی duplicate مقاوم‌تر نیاز دارد.

دو normalization با معنای متفاوت موجودند: `scripts/surt_mini/contracts.py:36` رقم را نگه می‌دارد، Latin را lowercase می‌کند و punctuation را حذف می‌کند؛ `datasets/persian_text.py:14` عدد صحیح را به واژه تبدیل می‌کند و decimal/date را رد می‌کند. مثال mini، `۱۲.۵ → 12 5`، در تست‌ها صریح است؛ این نگاشت آوایی «دوازده ممیز پنج» نیست. برای gate مکانیکی قابل قبول است، ولی برای کیفیت فارسی و مقایسهٔ WER، policy واحد و نسخه‌دار لازم است: ی/ک، نیم‌فاصله، اعراب، اعداد/تاریخ/اعشار، نشانه‌گذاری و Latin. متن خام را حفظ کنید؛ ambiguity تلفظ را با تبدیل حدسی پنهان نکنید.

BPE کوچک چهار جمله‌ای ممکن است در held-out حروف یا کلمات Latin را ندیده باشد. mini به‌درستی unknown/round-trip را گزارش می‌کند و برای اصلاح آن روی held-out آموزش نمی‌دهد. این برای structural gate پذیرفتنی است؛ برای ادعای کیفیت فارسی یا code-switching کافی نیست. معیار پوشش را از train/validation از پیش تعیین کنید و test را صرف انتخاب tokenizer نکنید. اطلاعات mini-test فعلی exposed است و باید هنگام طراحی test نهایی کنار گذاشته شود.

**۴. قرارداد Lhotse، HEAT و شکل تنسورها**

| مرحله | قرارداد مورد انتظار و وضعیت mini |
| --- | --- |
| ورودی آکوستیکی | `inputs: float [B,T,80]`؛ `input_lens: [B]` بر حسب frame، نه تعداد sample |
| متن | `text[b][channel]`؛ recipe آن را channel-major تخت می‌کند: تمام batch شاخهٔ ۰، سپس شاخهٔ ۱ |
| خروجی mask | `[B,T,160] → [B,T,80,2]`؛ سپس دو feature پوشیده روی batch concatenate می‌شوند: `[2B,T,80]` |
| encoder | `encoded: [2B,T',D]` و lengths واقعی؛ طول را نباید صرفاً از `T//4` حدس زد |
| source target | `source_feats` ذخیره‌شده و `source_feat_offsets` به تعداد supervision؛ dataset، `source_boundaries` و آرایه‌های per-source را برمی‌گرداند |
| decoder/CTC | embedding، joiner، simple AM/LM و CTC همگی باید همان vocabulary و special-IDها را داشته باشند |

mini در `prepare.py:102–140` از `KaldifeatFbank` و `K2SurtDataset(return_sources=True, return_cuts=True, num_channels=2, strict=True)` واقعی استفاده می‌کند؛ 80 bin، 16kHz، window/shift برابر 25/10ms، `dither=0`، `snip_edges=False` و `high_freq=-400` را کنترل می‌کند. features منبع به ترتیب onset با offsets ذخیره و پس از load، finite بودن، مرزها و شکل `[end-start,80]` آزموده می‌شوند. با onsetهای متمایز ۰ و ۰٫۵ ثانیه، ترتیب sourceها و HEAT سازگار است. برای onset مساوی در آینده، ترتیب `(start,speaker)` آرایهٔ source و ترتیب تخصیص متن باید یکسان و آزموده باشند.

در مقابل، `SurtTrainingCollator` خروجی waveform با شکل `[B,N_samples]` می‌دهد و adapter عمومی features و source targets نمی‌سازد. این ابزارها واسط آماده‌سازی‌اند؛ خروجی آن‌ها ورودی مستقیم `train.compute_loss` نیست. نبود این فیلدها مانع استفادهٔ مستقیم از adapter عمومی است، ولی mini مستقل آن‌ها را تولید می‌کند.

**یافتهٔ A1 — مهم، قطعی: helper عمومی HEAT برای zero overlap اشتباه است.**

مرجع: `training/data.py:37–68`، `validate_training_examples:104` و الگوریتم تخصیص `K2SurtDataset`.

`surt_training_example` دو source را فقط بر onset مرتب و به دو channel می‌دهد. در حالی که dataset، هر supervision را به اولین channel آزاد اختصاص می‌دهد؛ اگر جملهٔ B پس از پایان A شروع شود، هر دو روی channel صفر قرار می‌گیرند و channel یک خالی می‌ماند. probe روی control موجود، خروجی helper را `('الف','ب')` و بازتولید الگوریتم dataset را `('الف ب','')` نشان داد. validator عمومی هم رشتهٔ خالی را رد می‌کند، در حالی که این شاخهٔ خالی معتبر است. این مشکل بر mini ثابت که overlap مثبت دارد اثر مستقیم ندارد.

اصلاح پیشنهادی برای helper دو-source، همراه با تغییر validator به پذیرش شاخهٔ خالی معتبر:

```python
# Match the pinned dataset's first-available-channel assignment.
channels = [[], []]
last_end = [0.0, 0.0]
for source in sorted(record.sources, key=lambda item: item.start):
    channel = next(
        (i for i in range(2) if not channels[i] or last_end[i] <= source.start),
        None,
    )
    if channel is None:
        raise ValueError("More active sources than output channels")
    channels[channel].append(source.transcript.strip())
    last_end[channel] = max(last_end[channel], source.end)
targets = tuple(" ".join(items) for items in channels)
if len(targets) != 2 or not any(targets):
    raise ValueError("Expected two channels and at least one reference")
```

این snippet برای sourceهای همین قرارداد است؛ جایگزین general session dataset یا تخصیص هویت گوینده نیست. بهترین راه جلوگیری از drift، استفاده از متنِ تخصیص‌یافتهٔ dataset واقعی است. HEAT تخصیص زمانی است و دو branch هویت ثابت دو فرد را تضمین نمی‌کنند.

**۵. انتقال vocabulary و objective واقعی**

`contracts.VOCAB_TENSORS` و `transfer.py` هر ۹ tensor حساس را پوشش می‌دهند: embedding decoder، weight/bias خروجی joiner، simple AM/LM و CTC. ردیف‌ها بر اساس piece معمولی یکسان یا نقش special منتقل می‌شوند، نه صرفاً shape. blank قدیمی ۰ و unk قدیمی ۲ به blank جدید ۰ و unk جدید ۱ نگاشت می‌شوند؛ ردیف‌های فارسی جدید initialization تازه دارند. state-keyها، shapeها، مقادیر finite و محورهای مشکوک بررسی و همهٔ تصمیم‌ها گزارش می‌شوند. optimizer نیز Adam تازه است. بنابراین «جاافتادن CTC یا انتقال کل embedding صرفاً به دلیل برابری اندازه» ایراد این مسیر نیست.

اندازهٔ مورد استفاده `new_sp.get_piece_size()` است، نه عدد درخواستی ۱۲۸؛ این انتخاب با `hard_vocab_limit=False` درست است. محدودیت: چهار جمله تضمین نمی‌کنند alphabet لازم در بودجه جا شود؛ در آن صورت توقف train tokenizer باید حفظ شود. نگاشت piece مشترک، راه‌حل initialization است و تضمین سازگاری زبانی کامل embedding یا distribution جدید نیست.

`update.py` به `train.compute_loss` واقعی وصل است؛ در batch صفر، objective برابر است با:

```text
L = 1.0 * L_simple_RNNT + 0.1 * L_pruned_RNNT
  + 0.2 * L_CTC + 0.001 * L_HEAT_MSE
```

در منبع پین‌شده، ضریبهای RNN-T با warmup عوض می‌شوند؛ formula ثابت mini فقط برای همان یک step معتبر است. maskها در recipe روی log-fbank اعمال می‌شوند و خروجی DPRNN ReLU است؛ افزودن sigmoid/softmax یا جایگزینی با waveform separation، اصلاح بی‌هزینه و معادل recipe نیست.

**یافتهٔ A2 — ریسک شناخته‌شدهٔ upstream، نه خطای پنهان mini:** `compute_heat_loss` هدف را با `LOG_EPSILON` مقداردهی می‌کند و source را به آن می‌افزاید؛ همچنین padding mask با `>` به‌جای `>= input_lens` است. این رفتار در mini مستند و حفظ شده است. در batch دارای padding، فریم index برابر length معتبر نیست ولی وارد loss می‌شود. تک-mixture فعلی معمولاً این حالت را نمی‌آزماید. برای ۱B parity را حفظ کنید؛ پیش از آموزش اصلی یک regression با batch چندطولی بسازید و اصلاح احتمالی را در patch مستقل با hash و نسخهٔ objective ثبت کنید. تغییر silent را به‌عنوان همان baseline گزارش نکنید.

**یافتهٔ A3 — مهم برای قدرت gate:** `update.py:132–160` missing gradients را گزارش می‌کند ولی فقط nonzero بودن مجموع gradient و تغییر حداقل یک پارامتر را شرط می‌گذارد. از این شرط به‌تنهایی نمی‌توان آموزش‌پذیری تمام مسیر mask/encoder/predictor/joiner/CTC را نتیجه گرفت. نبود gradient ممکن است عمدی باشد؛ برای هر گروه مهم حداقل یک gradient و delta متناهی و غیرصفر، یا علت مستند نبود آن، الزام شود. همچنین قبل از loss طول توکن، دامنهٔ ID و شرط لازم alignment برای CTC بررسی شود؛ متن بسیار پرتوکن نسبت به خروجی کوتاه encoder می‌تواند loss نامتناهی بدهد.

```python
# Run this after backward(), with architecture-specific parameter prefixes.
for prefix in ("mask_encoder.", "encoder.", "decoder.",
               "joiner.", "ctc_output."):
    gradients = [p.grad for name, p in model.named_parameters()
                 if name.startswith(prefix) and p.requires_grad
                 and p.grad is not None]
    if not gradients or not any(
        torch.isfinite(g).all().item() and torch.count_nonzero(g).item() > 0
        for g in gradients
    ):
        raise RuntimeError(f"No finite nonzero gradient in {prefix}")
```

این کنترل مکمل بررسی finite همهٔ gradientهاست که اکنون وجود دارد. معیار delta هر گروه باید جدا از ذخیرهٔ optimizer گزارش شود. reload فعلی مدل strict و برابری tensorها را بررسی می‌کند و optimizer را ذخیره/finite-check می‌کند، اما اجرای resume با optimizer reload و step دوم را نمی‌آزماید؛ برای یک-step gate کافی است، برای ادعای resume آموزش کافی نیست.

**۶. استریم، branch tying و هزینه**

**یافتهٔ S1 — مانع قطعی استفادهٔ مستقیم از checkpoint در wrapper عمومی: front-end متفاوت است.**

`audio/features.py:34–45` صوت را در ۳۲۷۶۸ ضرب می‌کند، framing نوع snip و EOF zero-padding خودش را دارد؛ mini صوت float بدون این ضرب و `snip_edges=False` دارد. در ناحیهٔ غیرکف و غیرمرزی، فقط تفاوت scale می‌تواند حدود `2*ln(32768) = 20.79` به log-energy اضافه کند. این مقدار پیش‌بینی تحلیلی است؛ parity عددی با kaldifeat در این محیط اندازه‌گیری نشد. حتی اگر هر دو ۸۰ feature بدهند، توزیع و زمان مرجع یکسان نیست. شناسهٔ `kaldi80-snip-pcm16-v1` در loader این تفاوت را رفع نمی‌کند.

اصلاح برای ۱C: front-end آنلاین مطابق تنظیمات recipe بسازید و روی waveform واحد، frame count و مقادیر داخلی/مرزی/EOF را با extractor مرجع مقایسه کنید؛ tolerance عددی و mapping indexها صریح باشند. ورودی کوتاه، سکوت، طول غیرمضرب shift و chunkبندی‌های مختلف باید پوشش داده شوند. آزمون صرف برابری خروجی chunkبندی‌های extractor سفارشی، parity با مدل را اثبات نمی‌کند.

**یافتهٔ S2 — مانع قطعی export مستقیم branch tying رسمی.**

artifact رسمی `use_joint_encoder_layer='lstm'` دارد. در recipe، دادهٔ `[2B,T,D]` به `[B,T,2D]` تبدیل می‌شود و LSTM در محور زمان حرکت می‌کند؛ خروجی wrapper رسمی یک tensor است. اما `TorchSURTBackend.step` در `models/torch_surt.py:110–114` عبارت `encoded, _ = self.joint_encoder_layer(encoded)` و توضیح recurrence روی محور channel دارد. بدون adapter اختصاصی، این قرارداد با ماژول رسمی یکسان نیست. برای batch دوشاخه‌ای، unpack کردن tensor ممکن است به‌جای خطای فوری، محور batch را جدا کند و شکل را خراب کند. hidden/cell اتصال LSTM نیز در `reset`، `step`، `finish` و `_check_state` جایی ندارند. این ایراد از روی هر دو سورس قطعی است؛ رفتار runtime مدل واقعی در این ممیزی اجرا نشد.

اصلاح پیشنهادی قرارداد؛ implementation باید وزن و reshaping دقیق recipe را نگه دارد:

```python
# reset(): initialize this cache alongside mask and encoder caches.
self.joint_state = self.joint_encoder_layer.initial_state()

# step(): the adapter must map [2B,T,D] <-> [B,T,2D].
# Its LSTM must recur over time, retain (h,c), and exclude padded frames.
encoded, self.joint_state = self.joint_encoder_layer.forward_chunk(
    encoded, self.joint_state, lengths, final
)
```

state تازه باید در reset/EOF آزاد و در budget حافظه شمرده شود. یک مقایسهٔ one-chunk در برابر چند chunk، با output/length یکسان و branch tying فعال، ضروری است. `surt_mini/update.py` اتصال را با فراخوانی whole-batch مطابق recipe انجام می‌دهد و این ایراد wrapper را ندارد.

**یافتهٔ S3 — نیازمند مهندسی و شواهد:** DPRNN پین‌شده intra-RNN دوطرفه و inter-RNN یکطرفه دارد؛ در نتیجه limited-context بودن آن معادل API آمادهٔ `forward_chunk` نیست. window تقسیم/merge، هم‌پوشانی پنجره‌ها، حافظهٔ inter-RNN، subsampling و attention/convolution cache Zipformer باید با هم بررسی شوند. wrapper عمومی فقط chunk ۳۲ و right-context صفر یا یک را می‌پذیرد و فرض می‌کند خروجی tail معوق نداریم. بدون export و آزمون مرجع نمی‌توان پذیرفت این محدودیت‌ها با recipe سازگارند. محدودیت مصنوعی را قبل از استخراج context واقعی تثبیت نکنید.

برای عبور ۱C، آزمون‌های reset، EOF، طول غیرمضرب chunk، partition invariance در context یکسان، state محدود در صوت بلند و parity branch tying لازم‌اند. emission limit نیز در wrapper عمومی پیش‌فرض ۴ symbol/frame و در baseline ثبت‌شده ۱ است؛ برای parity decoder یک تنظیم واحد اعلام کنید. تست‌های `test_torch_surt.py` از مدل اسباب‌بازی و branch tying غیرفعال استفاده می‌کنند؛ شاهد پوشش این مانع نیستند.

ثبت زمان با CUDA synchronization و allocated/reserved memory در mini درست طراحی شده، اما یک step بدون warmup برآورد throughput پایدار نیست. دانلود، fbank، load، forward/backward/Adam و decode را جدا گزارش کنید. برآورد آموزش اصلی به batch-duration، طول token، mixed precision، sampler، I/O، warmup و optimizer اصلی وابسته است؛ ضرب زمان اولین step در تعداد دلخواه گام قابل دفاع نیست. allocated/reserved حافظهٔ فرایند را نیز با مصرف کل GPU یکسان نگیرید.

برای دادهٔ اصلی، بارگذاری هم‌زمان تمام waveformها در generator، استخراج per-source پی‌درپی GPU و NumPy-file-per-array هزینهٔ RAM/I/O دارند. در mini با هشت کلیپ مسئلهٔ اساسی نیستند. بهینه‌سازی worker/cache/batch و storage فشرده را به بعد از صحت قرارداد موکول کنید. resampler عمومی `np.interp` ضد alias مناسب ندارد؛ مسیر mini از `resample_poly` استفاده می‌کند و برای corpus اصلی هم باید resampling استاندارد و نسخه‌ثبت‌شده حفظ شود.

**۷. ترتیب اقدام پیشنهادی با حفظ دامنهٔ امکان‌سنجی**

1. **۱B:** همان هشت کلیپ و یک update واقعی را در محیط freezeشدهٔ Kaggle اجرا کنید. selection، tokenizer report، shape/HEAT contracts، transfer report، raw component loss، gradient/delta گروه‌ها، save/reload و decode دوشاخه‌ای باید شواهد داشته باشند. در صورت شکست metadata یا kernel، همان gate باز می‌ماند؛ دادهٔ ساختگی جایگزین نشود.
2. **پیش از مصرف ابزار عمومی در آموزش:** ناسازگاری zero-overlap/empty-channel، recording gainهای مؤثر و تعریف overlap را رفع و با dataset واقعی مقایسه کنید. validator باید تعداد source نادرست را با خطای قابل خواندن متوقف کند؛ اکنون پس از ثبت خطا همچنان `a,b = record.sources` می‌تواند exception بدهد.
3. **۱C:** front-end مطابق checkpoint و adapter stateful برای mask، Zipformer و branch-tying LSTM؛ سپس آزمون کوتاه/بلند، EOF/context و سنجش تأخیر. موفقیت whole-batch mini این گام را نمی‌بندد.
4. **گام ۳:** بعد از این gateها، اتصال کوچک و پین‌شدهٔ FastMSS با word alignment و طراحی مقایسهٔ متوازن؛ سپس RIR/noise.

زیرمجموعهٔ `aliyzd95/common_voice_22_0_fa` فقط ورودی موقت اعتبارسنجی فاز صفر است. منبع برنامه‌ریزی‌شدهٔ Full Training، طبق نقشهٔ راه، Common Voice 27 اصلی از Mozilla Data Collective است؛ این گزارش دسترسی یا آمادگی آن را دوباره تأیید نکرده است. پیش از test نهایی باید speaker/source/audioهای exposed در تمام تلاش‌های mini، از جمله تلاش‌های شکست‌خورده، بررسی و برخوردها حذف شوند. mini BPE توکنایزر نهایی محسوب نمی‌شود. PI-WER مخلوط کوتاه در این gate، جای ORC-WER جلسه یا شاهد یادگیری فارسی و code-switching نیست.

**۸. مراجع قابل بازتولید**

مراجع محلی: `docs/prompts/PERSIAN_SURT_FOLLOWUPS.md`، `docs/PERSIAN_SURT_ROADMAP_FA.md`، `docs/KAGGLE_SURT_MINI_PERSIAN_RUNBOOK_FA.md`، `docs/experiments/surt_mini_1b_preparation_2026-11-13.md`، pins هر دو مسیر، کدهای اشاره‌شده و بستهٔ baseline.

مراجع اولیهٔ بیرونی: repository رسمی `k2-fsa/icefall` در revision بالا، مسیر `egs/libricss/SURT/dprnn_zipformer/{train.py,model.py,dprnn.py}` و `egs/libricss/SURT/local/{add_source_feats.py,compute_fbank_lsmix.py}`؛ repository رسمی `lhotse-speech/lhotse`، tag `v1.33.0`، فایل `lhotse/dataset/surt.py`؛ README رسمی repository `popcornell/FastMSS`. خواندن این منابع برای تطبیق قرارداد بود؛ نرم‌افزار یا مدل تازه‌ای نصب/اجرا نشد.

| فایل upstream بررسی‌شده | SHA256 |
| --- | --- |
| Icefall `train.py` | `e3c5f6b4d453ec6ed31a69ee140a8187501d0a9fe87fb2d9871de410ade59b5d` |
| Icefall `model.py` | `f16830b5c687f6f55c755358f6011db9a1698535994e0fdffa3d4bc00f86c40f` |
| Icefall `add_source_feats.py` | `0659320b226ce6e2d1ef74bc17c06fc97b718ef55d06053e32735ab2d484b82a` |
| Icefall `compute_fbank_lsmix.py` | `d08568be11bbe9c8e62ca6e6ab652a1ba9096bbbdec4d79cd6e51b5db4ac0099` |
