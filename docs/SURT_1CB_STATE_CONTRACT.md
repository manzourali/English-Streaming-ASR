# قرارداد اجرایی 1C-B

مبنای ممیزی: `outputs/surt-stream-1c/run/upstream-source/`، اجرای
`stream-20261119-b`، revision برابر `f8c8578c9dbfa1183cc949b89b5d21bd33aa2ce1`.
هش هشت فایل مؤثر در `configs/surt_stream/native-source.json` پین شده است؛ runner
مسیرهای resolveشدهٔ Zipformer/decoder/joiner را هم بررسی می‌کند. نسخهٔ dependency،
وزن یک-update، tokenizer ۱۲۸تایی و آرگومان‌های معماری 1B تغییر نکرده‌اند.
این قرارداد در اجرای `streamb-20261120-a` روی Kaggle آزموده و gate فنی آن در ۲۰۲۶-۱۱-۲۰ پذیرفته شد. شواهد در `outputs/streamb-20261120-a/` است؛ پذیرش کیفیت فارسی یا latency زنده را شامل نمی‌شود.

## DPRNN

ورودی `1,T,80`؛ embedding به `1,T,256`. segment=32 و stride=16 است.
`split_feature` ابتدا ۱۶ صفر **پس از embedding** در چپ می‌گذارد، سپس segmentهای
با شروع `-16,0,16,...` می‌سازد. چهار بلوک، هرکدام row دوطرفه و col یک‌طرفه دارد.
row روی `1,32,256` کامل اجرا می‌شود و state آن برای segment بعدی حمل نمی‌شود،
همانند upstream. col روی `32,1,256` با `batch_first=True` اجرا می‌شود:
موقعیت‌های داخل segment محور batch و segmentها محور زمان‌اند. برای هر بلوک
`h,c` با شکل `1,32,256` نگهداری می‌شود. توضیح `(S,N,E)` در docstring لایه با
کد واقعی `batch_first=True` سازگار نیست؛ adapter از کد اجرایی پیروی می‌کند.
`RNNEncoderLayer.forward` برخلاف annotation فقط tensor برمی‌گرداند؛ adapter
برای col هستهٔ موجود LSTM و همان residual/dropout/norm را فراخوانی می‌کند.

دو نیمهٔ هم‌پوشان **جمع** می‌شوند؛ تقسیم بر دو نداریم. سپس `out_embed` و ReLU،
و reshape به `1,T,80,2` اعمال می‌شود. ورودی خام فقط تا تولید ماسک نگه داشته می‌شود.
نیمهٔ اول معتبر پس از segment دوم آماده است. هر فریم برای پایان segment حاوی آن
منتظر می‌ماند؛ تأخیر آیندهٔ mask بسته به موقعیت حدود ۱۶ تا ۳۱ فریم است.
EOF: صفرهای embedded فقط برای تکمیل آخرین segment مورد نیاز؛ خروجی دقیقاً T
فریم است. segmentهای انتهایی upstream که فقط خروجی دورریختنی دارند اجرا نمی‌شوند؛
col علّی است و این حذف روی خروجی معتبر اثری ندارد. مرجع مستقل، خود DPRNN کامل است.

## Zipformer و اتصال شاخه‌ها

ماسک‌ها به شکل channel-major `2,T,80` به recognition می‌روند. محدودیت B=1 صریح است.
هر شاخه cache مستقل دارد؛ `get_init_state` upstream خودش برای batch=1 ساخته شده است.
هر فراخوانی `streaming_forward` ورودی `1,39,80` می‌گیرد، ۳۲ فریم جلو می‌رود و
۷ فریم overlap دارد. سه convolution subsampling میدان دریافتی ۹ و stride=2 دارند؛
۳۹ فریم دقیقاً ۱۶ فریم داخلی می‌سازد. downsample خروجی ۲ است، در نتیجه هر بار
۸ فریم `1,8,256` بیرون می‌آید. هیچ history کامل بازحساب نمی‌شود.

برای هر شاخه ۳۵ tensor: پنج گروه encoder، هفت خانوادهٔ cache. برای گروه i با
`L=2`، `ds=(1,2,4,8,2)`، `C=256` و `A=192`:

| state | شکل |
| --- | --- |
| len | `L,1` شمارندهٔ int64 |
| avg | `L,1,C` میانگین تمام گذشته؛ حافظه ثابت، context چپ نامحدود به صورت خلاصه |
| key | `L,64/ds,1,A` |
| val و val2 | هرکدام `L,64/ds,1,A/2` |
| conv1 و conv2 | هرکدام `L,1,C,30` |

attention چپ چهار chunk داخلی ۱۶تایی است؛ convها cache چپ ۳۰تایی دارند؛
pooling خلاصهٔ علّی گذشته را نگه می‌دارد. attention آینده فقط داخل chunk فعلی است.
`encoder.forward` آفلاین به دلیل pooling/convolution و ابتدای cache مرجع هم‌context
برای native API نیست. مرجع تعریف‌شده از خود native API با grid ثابت استفاده می‌کند؛
صحت محاسبات داخلی آن ادعای تازهٔ این adapter نیست.

اتصال شاخه‌ها: `2,T,256 -> 1,T,512`، LSTM یک‌طرفه با `h,c=1,1,512`، ReLU،
سپس `2,T,256`. state پس از هر chunk نگهداری می‌شود؛ padding وارد LSTM اتصال نمی‌شود.
مرجع اتصال، wrapper کامل upstream روی tensor معتبر concatenated است.

EOF recognition: تعداد خروجی معتبر `ceil(max(0,floor((F-7)/2))/2)` است.
chunk آخر با مقدار ثابت `log(1e-10)` **پس از masking** به ۳۹ فریم می‌رسد؛ خروجی‌های
اضافی پیش از joint/decoder حذف می‌شوند. این سیاست context انتهایی صریح و یکسان در
مرجع و adapter است و با batch forward بدون padding همسان فرض نمی‌شود. F<9 خروجی
encoder ندارد؛ featureهای معتبر حذف نشده‌اند، ولی receptive field کافی موجود نیست.
EOF تکراری تهی است؛ صوت پس از EOF خطا است؛ reset تمام stateها و decoder را نو می‌کند.

## Decoder و زمان

context اولیه دقیق greedy upstream: `[-1,blank]`؛ فقط دو ID آخر هر شاخه نگهداری
می‌شوند. `need_pad=False` و joiner projection همان upstream است؛ حداکثر یک نماد
در هر فریم. سیاست blank/unknown دقیق `getattr(model,'unk_id',blank)` است، بدون
تغییر پنهانی آن. transcript/events متعلق به مصرف‌کننده‌اند و state صوتی نیستند.
ترتیب شاخه‌های ۰ و ۱ در همهٔ مقایسه‌ها ثابت است؛ تطبیق با permutation برای پوشاندن
اختلاف انجام نمی‌شود و این شاخه‌ها هویت ثابت گوینده نیستند.

برای اولین ۸ خروجی، ۳۹ feature ماسک‌شده لازم است؛ grid ماسک نیازمند ۶۴ feature خام
است. feature ۶۳ پس از `(63*160+280)=10360` نمونه، یعنی ۶۴۷٫۵ms صوت دریافت‌شده
آماده می‌شود. این زمان context اسمی نسبت به anchor صفر خروجی است؛ زمان محاسبه،
تجمیع packet و تشخیص پایان را شامل نمی‌شود. با packetهای ۱۰۰ms اولین فرصت دریافت
۷۰۰ms است. anchorهای خروجی grid اسمی ۴۰ms دارند؛ alignment واژه یا timestamp
آکوستیکی اثبات‌شده نیستند. `actual_live_delivery_ms` و endpoint نامعلوم و null هستند.

timeline دریافت تجمعی، حداقل صوت مورد نیاز، اضافه‌انتظار packet، compute هر push،
compute تجمعی، صف پردازش و زمان آماده‌شدن شبیه‌سازی‌شده را جدا دارد.
زمان آماده‌شدن با `max(previous_ready,received_time)+push_compute` محاسبه می‌شود؛
محاسبات قبلی که در فاصلهٔ دریافت packetها تمام شده‌اند دوباره به تأخیر افزوده نمی‌شوند. replay عمداً بدون sleep است؛
زمان زندهٔ واقعی اندازه‌گیری نشده است. RTF از compute شامل frontend CPU، انتقال،
ماسک/encoder/joint/decoder، ثبت اندازهٔ state و CUDA sync تقسیم بر طول صوت است. دو warmup و پنج اندازه‌گیری؛
wall جداگانه overhead گزارش‌سازی را هم دارد. peak allocated/reserved CUDA و RSS
کل process جدا ثبت می‌شوند؛ RSS شامل مرجع finite-recording و تست‌هاست.

state صوتی با اندازهٔ **storage** محاسبه می‌شود؛ sliceها clone می‌شوند تا history در
storage پنهان نماند. صف‌های frontend/mask/encoder محدودند؛ col/joint/cache شکل ثابت
دارند. stress ۱۲۰ ثانیه‌ای packet را از تکرار صوت واقعی می‌سازد و خروجی را نگه نمی‌دارد.
تکرار صوت فقط آزمون حافظه است، نه شواهد کیفیت جلسهٔ بلند. حد پذیرش storage شانزده MiB
است؛ مقادیر واقعی و شمار فریم‌ها باید در آرشیو بررسی شوند.
