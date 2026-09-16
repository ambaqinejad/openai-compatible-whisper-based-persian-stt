# app/core/text_corrector.py

import logging

from openai import AsyncOpenAI

from app.config import get_settings


logger = logging.getLogger(__name__)

settings = get_settings()


CORRECTION_SYSTEM_PROMPT = """
تو یک ویراستار تخصصی متون فارسی هستی.

متن ورودی معمولاً خروجی یک سیستم Speech-to-Text است و ممکن است
دارای خطاهای تشخیص گفتار، نویز، غلط املایی، کلمات تکراری،
جمله‌های تکراری، تکرارهای حلقه‌ای و کلمات اشتباه تشخیص داده‌شده باشد.

وظیفه تو اصلاح متن است، نه بازنویسی آزادانه آن.

قوانین:

1. غلط‌های املایی فارسی را اصلاح کن.

2. کلمات و عبارت‌هایی که به دلیل خطای Speech-to-Text به صورت
   غیرطبیعی تکرار شده‌اند را حذف کن.

3. تکرارهای حلقه‌ای را شناسایی و حذف کن.
   برای مثال اگر یک عبارت چند بار پشت سر هم تکرار شده باشد،
   فقط نسخه صحیح و منطقی آن را نگه دار.

4. کلمات یا عبارت‌هایی که به وضوح ناشی از خطای ASR هستند
   و از context قابل اصلاح هستند را اصلاح کن.

5. جمله‌های ناقص یا به‌هم‌ریخته را تا حد امکان به شکل طبیعی
   اصلاح کن.

6. علائم نگارشی مناسب فارسی را اضافه یا اصلاح کن.

7. فاصله‌گذاری و نیم‌فاصله‌های فارسی را در صورت امکان اصلاح کن.

8. معنی و منظور گوینده را تغییر نده.

9. اطلاعات جدیدی که در متن وجود ندارد اضافه نکن.

10. اگر بخشی از متن مبهم است، حدس بی‌مورد نزن.

11. لحن متن را حفظ کن.

12. متن را خلاصه نکن.

13. هیچ جمله‌ای را فقط به دلیل اینکه از نظر نگارشی بهتر می‌شود
    حذف نکن، مگر اینکه واضحاً نویز یا تکرار ناشی از ASR باشد.

14. خروجی فقط متن اصلاح‌شده باشد.

15. هیچ توضیح، تحلیل، مقدمه، نتیجه‌گیری، Markdown یا توضیحی
    درباره تغییرات ارائه نکن.

فقط متن نهایی اصلاح‌شده را برگردان.
"""


class TextCorrector:

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=settings.qwen_base_url,
            api_key=settings.qwen_api_key,
        )

        self.model = settings.qwen_model

    async def correct(self, text: str) -> str:

        if not text or not text.strip():
            return text

        logger.info(
            "Sending text to Qwen for correction. "
            "Length=%d",
            len(text),
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=settings.qwen_max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": CORRECTION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        if not response.choices:
            raise RuntimeError(
                "Qwen returned no choices"
            )

        corrected_text = (
            response.choices[0]
            .message
            .content
        )

        if not corrected_text:
            raise RuntimeError(
                "Qwen returned empty text"
            )

        corrected_text = corrected_text.strip()

        logger.info(
            "Qwen text correction completed. "
            "Original=%d chars, Corrected=%d chars",
            len(text),
            len(corrected_text),
        )

        return corrected_text