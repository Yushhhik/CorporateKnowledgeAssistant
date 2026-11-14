import asyncio
import html
import logging
import os
import re
from collections import defaultdict, deque

import httpx
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    User,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bot")

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPERATOR_CHAT_ID = os.getenv("OPERATOR_CHAT_ID")
BASE_API_URL = os.getenv("API_URL", "http://app:8000").rstrip("/")
CORE_API_URL = f"{BASE_API_URL}/webhook"
PROXY_URL = os.getenv("PROXY_URL") or None
CORE_TIMEOUT = float(os.getenv("CORE_TIMEOUT", "180"))
TG_LIMIT = 4000
TEXT_CHUNK = 3500
CAPTION_SAFE = 900
OPERATOR_BUTTON_TEXT = "👤 Связаться с оператором"

if not BOT_TOKEN:
    raise SystemExit("TELEGRAM_TOKEN не задан")

bot: Bot | None = None
http: httpx.AsyncClient | None = None
dp = Dispatcher()

FAQ = {
    "faq_interface": "Как выглядит интерфейс БО?",
    "faq_filter": "Как работает фильтр документа?",
    "faq_manager": "Какой график работы у офис-менеджера?",
    "faq_trip": "Как запустить бизнес-процесс командировки?",
}
FAQ_BUTTONS = {
    "faq_interface": "Как выглядит интерфейс БО?",
    "faq_filter": "Как настроить фильтр?",
    "faq_manager": "График офис-менеджера",
    "faq_trip": "Как запустить командировку?",
}
OPERATOR_RE = re.compile(r"\b(оператор\w*|человек\w*|живой)\b", re.IGNORECASE)

local_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))


# вспомогательные функции
def get_faq_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=k)] for k, t in FAQ_BUTTONS.items()]
    )


def operator_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Связаться с оператором", callback_data="call_operator")]]
    )


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=OPERATOR_BUTTON_TEXT)]],
        resize_keyboard=True,
        is_persistent=True,
    )


def local_history_text(user_id: int) -> str:
    items = local_history.get(user_id)
    return "\n".join(items) if items else "История пуста."


def chunks(text: str, size: int = TG_LIMIT) -> list[str]:
    parts: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > size:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:size])
            line = line[size:]
        if cur and len(cur) + len(line) + 1 > size:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur or not parts:
        parts.append(cur)
    return parts


def md_to_html(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", r"<pre>\1</pre>", t, flags=re.S)
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    t = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", t, flags=re.M)
    t = re.sub(r"^(\s*)[-*]\s+", r"\1• ", t, flags=re.M)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.S)
    t = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', t)
    return t


def is_operator_request(text: str) -> bool:
    return bool(OPERATOR_RE.search(text)) and len(text.split()) <= 4


async def keep_typing(chat_id: int) -> None:
    while True:
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            return
        await asyncio.sleep(4)


async def send_text(message: Message, text: str, markup: InlineKeyboardMarkup | None = None) -> None:
    parts = chunks(text, TEXT_CHUNK)
    for i, part in enumerate(parts):
        kb = markup if i == len(parts) - 1 else None
        try:
            await message.answer(md_to_html(part), parse_mode=ParseMode.HTML, reply_markup=kb)
        except TelegramBadRequest:
            log.warning("HTML не разобрался, отправляю простым текстом")
            await message.answer(part, reply_markup=kb)


async def load_photo(image_path: str):
    try:
        if os.path.isfile(image_path):
            return FSInputFile(image_path)
        url = image_path if image_path.startswith("http") else f"{BASE_API_URL}/{image_path.lstrip('/')}"
        r = await http.get(url)
        r.raise_for_status()
        return BufferedInputFile(r.content, filename=os.path.basename(image_path) or "image.png")
    except Exception:
        log.exception("Не удалось получить картинку %s", image_path)
        return None


async def send_photo(message: Message, photo, caption: str | None, markup) -> bool:
    try:
        await message.answer_photo(
            photo,
            caption=md_to_html(caption) if caption else None,
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )
        return True
    except TelegramBadRequest:
        try:
            await message.answer_photo(photo, caption=caption, reply_markup=markup)
            return True
        except Exception:
            log.exception("Не удалось отправить фото")
            return False


async def fetch_dialog_history(user_id: int) -> str:
    try:
        r = await http.get(f"{BASE_API_URL}/history/{user_id}")
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            return r.text or local_history_text(user_id)
        if isinstance(data, dict):
            return str(data.get("history") or local_history_text(user_id))
        return str(data)
    except Exception as e:
        log.warning("Ошибка получения истории для %s: %s", user_id, e)
        return local_history_text(user_id)


# основная логика
async def answer_question(message: Message, user: User, text: str) -> None:
    if is_operator_request(text):
        await escalate(message, user)
        return

    local_history[user.id].append(f"Пользователь: {text}")
    typing_task = asyncio.create_task(keep_typing(message.chat.id))
    try:
        r = await http.post(CORE_API_URL, json={"user_id": user.id, "text": text})
        r.raise_for_status()
        data = r.json()
    except Exception:
        log.exception("Ошибка запроса к серверу")
        await message.answer(
            "Произошла ошибка при обработке вашего запроса. Попробуйте позже или свяжитесь с оператором.",
            reply_markup=operator_keyboard(),
        )
        return
    finally:
        typing_task.cancel()

    if data.get("status") == "error":
        await message.answer(
            data.get("reply") or "Не удалось обработать запрос.", reply_markup=operator_keyboard()
        )
        return

    reply = data.get("reply") or "Извините, я не смог сформировать ответ."
    local_history[user.id].append(f"Бот: {reply}")
    markup = operator_keyboard() if data.get("suggest_operator") else None

    image_path = data.get("image_path")
    if image_path:
        photo = await load_photo(image_path)
        if photo:
            short = len(reply) <= CAPTION_SAFE
            sent = await send_photo(message, photo, reply if short else None, markup if short else None)
            if sent and short:
                return
            if sent:
                await send_text(message, reply, markup)
                return
    await send_text(message, reply, markup)


async def escalate(message: Message, user: User) -> None:
    if not OPERATOR_CHAT_ID:
        log.error("OPERATOR_CHAT_ID не задан")
        await message.answer("Оператор сейчас недоступен. Попробуйте, пожалуйста, позже.")
        return

    history = await fetch_dialog_history(user.id)
    text = (
        "🚨 Запрос на оператора!\n"
        f"Пользователь: {user.full_name} (@{user.username or 'нет username'}), ID: {user.id}\n\n"
        f"История диалога:\n{history}"
    )
    try:
        for part in chunks(text):
            await bot.send_message(OPERATOR_CHAT_ID, part)
    except Exception:
        log.exception("Не удалось отправить сообщение оператору")
        await message.answer("Не получилось связаться с оператором. Попробуйте, пожалуйста, чуть позже.")
        return

    await message.answer("Я передал ваш вопрос специалисту, он скоро ответит.")


# хэндлеры
@dp.message(CommandStart())
async def cmd_start(message: Message):
    user_name = html.escape(message.from_user.first_name or "")
    greeting_text = (
        f"Здравствуйте, {user_name}! 👋\n\n"
        "Я - <b>БАРС-Помощник</b>, виртуальный ассистент компании <b>БАРС Груп</b>. "
        "Я изучил всю внутреннюю документацию и готов ответить на ваши вопросы.\n\n"
        "Выберите частый вопрос ниже или напишите свой текстом. "
        "Если понадобится живой человек, нажмите кнопку «Связаться с оператором» внизу или напишите «оператор»."
    )
    await message.answer(greeting_text, parse_mode=ParseMode.HTML, reply_markup=main_keyboard())
    await message.answer("Частые вопросы:", reply_markup=get_faq_keyboard())


@dp.message(Command("stop"))
async def cmd_stop(message: Message):
    await message.answer(
        "Всего доброго! Если появятся вопросы - я всегда на связи. 👋",
        reply_markup=ReplyKeyboardRemove(),
    )


@dp.message(Command("operator"))
async def cmd_operator(message: Message):
    await escalate(message, message.from_user)


@dp.message(F.text == OPERATOR_BUTTON_TEXT)
async def handle_operator_button(message: Message):
    await escalate(message, message.from_user)


@dp.callback_query(F.data.startswith("faq_"))
async def process_faq(callback: CallbackQuery):
    await callback.answer()
    question_text = FAQ.get(callback.data)
    if not question_text:
        return
    await callback.message.answer(f"❓ {question_text}")
    await answer_question(callback.message, callback.from_user, question_text)


@dp.callback_query(F.data == "call_operator")
async def process_call_operator(callback: CallbackQuery):
    await callback.answer()
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass
    await escalate(callback.message, callback.from_user)


@dp.message(F.text)
async def handle_text(message: Message):
    log.info("Сообщение от %s, длина %s", message.from_user.id, len(message.text))
    await answer_question(message, message.from_user, message.text.strip())


@dp.message()
async def handle_non_text(message: Message):
    await message.answer("Я понимаю только текст. Пожалуйста, напишите ваш вопрос текстом.")


# запуск
async def main():
    global http, bot

    if PROXY_URL:
        bot = Bot(token=BOT_TOKEN, session=AiohttpSession(proxy=PROXY_URL))
    else:
        bot = Bot(token=BOT_TOKEN)

    http = httpx.AsyncClient(timeout=httpx.Timeout(CORE_TIMEOUT, connect=10.0), trust_env=False)
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        log.info("Бот запущен, сервер: %s, прокси: %s", BASE_API_URL, "да" if PROXY_URL else "нет")
        await dp.start_polling(bot)
    finally:
        await http.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())