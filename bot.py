import asyncio
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    user_name = message.from_user.first_name
    
    greeting_text = (
        f"Здравствуйте, {user_name}! 👋\n\n"
        "Я - виртуальный помощник компании **БАРС Груп**. "
        "Я изучил всю внутреннюю документацию "
        "и готов ответить на ваши вопросы.\n\n"
        "Если понадобится живой человек, просто напишите «оператор» или нажмите соответствующую кнопку. "
        "Чем я могу Вам помочь?"
    )
    
    await message.answer(greeting_text, parse_mode="Markdown")

@dp.message(Command("stop"))
async def cmd_stop(message: Message):
    await message.answer("Всего доброго! Если появятся вопросы - я всегда на связи. 👋")

@dp.message()
async def echo_handler(message: Message):
    await message.answer(f"Вы написали: {message.text}. (Скоро я научусь отвечать по документации!)")

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
