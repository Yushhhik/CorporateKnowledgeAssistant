import asyncio
import os
import httpx
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.client.session.aiohttp import AiohttpSession

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
PROXY_URL = "socks5://host.docker.internal:10808"

FASTAPI_URL = "http://app:8000/webhook"


async def main():
    if not TOKEN:
        print("❌ ОШИБКА: Токен не найден! Проверь файл .env")
        return

    session = AiohttpSession(proxy=PROXY_URL)

    bot = Bot(token=TOKEN, session=session)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: types.Message):
        await message.answer("Привет! Я Corporate Knowledge Assistant. Задай мне любой вопрос по документации.")

    @dp.message(lambda message: message.text is not None)
    async def handle_text(message: types.Message):
        print(f"📥 Бот получил сообщение: {message.text}")

        user_text = message.text
        user_id = message.from_user.id

        await bot.send_chat_action(chat_id=message.chat.id, action="typing")

        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                response = await client.post(
                    FASTAPI_URL,
                    json={"user_id": user_id, "text": user_text},
                    timeout=30.0
                )
                data = response.json()
                reply_text = data.get("reply", "Ошибка: пустой ответ от сервера.")
            except Exception as e:
                reply_text = f"Ошибка подключения к серверу: {e}"

        await message.answer(reply_text)

    @dp.message()
    async def handle_non_text(message: types.Message):
        await message.answer("Извините, я понимаю только текст. Пожалуйста, напишите ваш вопрос словами.")

    print("🚀 Бот запущен! Напиши ему в Телеграме.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())