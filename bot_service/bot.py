import asyncio
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

def get_faq_keyboard():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Как выглядит интерфейс БО?", callback_data="faq_interface")],
        [InlineKeyboardButton(text="Как настроить фильтр?", callback_data="faq_filter")],
        [InlineKeyboardButton(text="График офис-менеджера", callback_data="faq_manager")],
        [InlineKeyboardButton(text="Как запустить командировку?", callback_data="faq_trip")]
    ])
    return keyboard

@dp.message(Command("start"))
async def cmd_start(message: Message):
    user_name = message.from_user.first_name
    
    greeting_text = (
        f"Здравствуйте, {user_name}! 👋\n\n"
        "Я - виртуальный помощник компании **БАРС Груп**. "
        "Я изучил всю внутреннюю документацию "
        "и готов ответить на ваши вопросы.\n\n"
        "Выберите частый вопрос ниже или напишите свой текстом. "
        "Если понадобится живой человек, просто напишите «оператор»."
    )
    
    await message.answer(greeting_text, parse_mode="Markdown", reply_markup=get_faq_keyboard())

@dp.message(Command("stop"))
async def cmd_stop(message: Message):
    await message.answer("Всего доброго! Если появятся вопросы - я всегда на связи. 👋")

@dp.callback_query(F.data.startswith("faq_"))
async def process_faq(callback: CallbackQuery):
    questions = {
        "faq_interface": "Как выглядит интерфейс БО?",
        "faq_filter": "Как работает фильтр документа?",
        "faq_manager": "Какой график работы у офис-менеджера?",
        "faq_trip": "Как запустить бизнес-процесс командировки?"
    }
    question_text = questions.get(callback.data, "Вопрос не найден")
    
    await callback.answer()
    await callback.message.answer(f"Вы выбрали вопрос: {question_text}\n\n(Скоро я отправлю его на сервер и дам ответ из документации!)")

@dp.message(F.text)
async def echo_handler(message: Message):
    await message.answer(f"Вы написали: {message.text}. (Скоро я научусь отвечать по документации!)")

@dp.message()
async def handle_non_text(message: Message):
    await message.answer("Я понимаю только текст. Пожалуйста, напишите ваш вопрос текстом.")

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())