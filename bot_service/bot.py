import asyncio
import base64
import html
import io
import json
import logging
import os
import re
import time
from collections import OrderedDict, defaultdict, deque

import httpx
from dotenv import load_dotenv
from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
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
BASE_API_URL = os.getenv("API_URL", "http://app:8000").rstrip("/").removesuffix("/webhook")
CORE_API_URL = f"{BASE_API_URL}/webhook"
PROXY_URL = os.getenv("PROXY_URL") or None
CORE_TIMEOUT = float(os.getenv("CORE_TIMEOUT", "1300"))
SLOW_NOTICE_DELAY = float(os.getenv("SLOW_NOTICE_DELAY", "8"))
SYNC_INTERVAL = float(os.getenv("SYNC_INTERVAL", "300"))
STATS_FILE = os.getenv("STATS_FILE", "/data/stats.json")
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", "/data/feedback.jsonl")
AUTH_FILE = os.getenv("AUTH_FILE", "/data/authorized.json")
ADMIN_IDS = {int(x) for x in re.findall(r"-?\d+", os.getenv("ADMIN_IDS", ""))}
TG_LIMIT = 4000
TEXT_CHUNK = 3500
CAPTION_SAFE = 900
OPERATOR_BUTTON_TEXT = "👤 Связаться с оператором"
ADMIN_BUTTON_TEXT = "⚙️ Панель управления"
EMPLOYEE_ID_MAX_LEN = 100
DOC_EXTENSIONS = (".pdf", ".docx")
DOC_MAX_SIZE = 20 * 1024 * 1024
PHOTO_MAX_SIZE = 10 * 1024 * 1024
PHOTO_WAIT = 600

if not BOT_TOKEN:
    raise SystemExit("TELEGRAM_TOKEN не задан")

try:
    OPERATOR_CHAT_INT = int(OPERATOR_CHAT_ID) if OPERATOR_CHAT_ID else None
except ValueError:
    OPERATOR_CHAT_INT = None

bot: Bot | None = None
http: httpx.AsyncClient | None = None
dp = Dispatcher()
operator_router = Router()
router = Router()
router.message.filter(F.chat.type == "private")
dp.include_routers(operator_router, router)


class Registration(StatesGroup):
    waiting_id = State()


class AdminPanel(StatesGroup):
    waiting_document = State()


OPERATOR_RE = re.compile(r"\b(оператор\w*|человек\w*|живой)\b", re.IGNORECASE)

HELP_TEXT = (
    "🤖 <b>Что я умею</b>\n\n"
    "Отвечаю на вопросы по внутренней документации БАРС Груп: БАРС-Офис, БП «Командировка», реестр мебели.\n\n"
    "<b>Примеры вопросов:</b>\n"
    "• Как выглядит интерфейс БО? Где найти производственный календарь?\n"
    "• Как настроить фильтр документа?\n"
    "• Кто согласует БП «Командировка»?\n"
    "• Какой график работы у офис-менеджера?\n"
    "• Какие статусы мебели есть в реестре?\n"
    "• Как добавить запись в реестр мебели?\n\n"
    "Если ответ не помог, нажмите «Связаться с оператором» или напишите «оператор».\n\n"
    "<b>Команды:</b>\n"
    "/start — приветствие и частые вопросы\n"
    "/help — эта справка\n"
    "/operator — связаться с оператором\n"
    "/stop — попрощаться"
)

ADMIN_HELP_TEXT = "\n/admin — панель управления\n/cancel — отменить текущее действие"

BASE_COMMANDS = [
    BotCommand(command="start", description="Приветствие и частые вопросы"),
    BotCommand(command="help", description="Что умеет бот и примеры вопросов"),
    BotCommand(command="operator", description="Связаться с оператором"),
    BotCommand(command="stop", description="Попрощаться с ботом"),
]

ADMIN_COMMANDS = [
    BotCommand(command="admin", description="Панель управления"),
    BotCommand(command="stats", description="Статистика бота"),
    BotCommand(command="cancel", description="Отменить действие"),
]

OPEN_CALLBACKS = {"call_admin"}

COUNTERS = (
    "requests",
    "answered",
    "not_found",
    "errors",
    "escalations",
    "operator_replies",
    "feedback_positive",
    "feedback_negative",
    "registrations",
    "access_requests",
)

local_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
exchanges: OrderedDict[int, tuple[int, str, str]] = OrderedDict()
exchange_seq = int(time.time())
rated: set[int] = set()
operator_threads: OrderedDict[int, int] = OrderedDict()
pending_access: dict[int, str] = {}
pending_photos: dict[int, tuple[float, str]] = {}


def load_stats() -> dict:
    data = {k: 0 for k in COUNTERS}
    data.update(users=[], latency_sum=0.0, latency_n=0)
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            data.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("Не удалось прочитать статистику: %s", e)
    return data


stats = load_stats()


def load_authorized() -> set[int]:
    try:
        with open(AUTH_FILE, encoding="utf-8") as f:
            return {int(x) for x in json.load(f)}
    except FileNotFoundError:
        return set()
    except Exception as e:
        log.warning("Не удалось прочитать список авторизованных: %s", e)
        return set()


authorized = load_authorized()


# вспомогательные функции
def operator_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Связаться с оператором", callback_data="call_operator")]]
    )


def feedback_keyboard(exchange_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍 Полезно", callback_data=f"fb:{exchange_id}:1"),
                InlineKeyboardButton(text="👎 Не помогло", callback_data=f"fb:{exchange_id}:0"),
            ]
        ]
    )


def admin_contact_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Связаться с админом", callback_data="call_admin")]]
    )


def access_request_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Одобрить", callback_data=f"access:approve:{user_id}"),
                InlineKeyboardButton(text="❌ Отклонить", callback_data=f"access:deny:{user_id}"),
            ]
        ]
    )


def admin_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Статистика", callback_data="adm:stats")],
            [InlineKeyboardButton(text="📄 Добавить документ в базу знаний", callback_data="adm:upload")],
            [InlineKeyboardButton(text="📚 Список документов", callback_data="adm:docs")],
            [InlineKeyboardButton(text="✖️ Закрыть", callback_data="adm:close")],
        ]
    )


def main_keyboard(admin: bool = False) -> ReplyKeyboardMarkup:
    rows = [[KeyboardButton(text=OPERATOR_BUTTON_TEXT)]]
    if admin:
        rows.append([KeyboardButton(text=ADMIN_BUTTON_TEXT)])
    return ReplyKeyboardMarkup(
        keyboard=rows,
        resize_keyboard=True,
        is_persistent=True,
    )


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def is_authorized(user_id: int) -> bool:
    return user_id in authorized or is_admin(user_id)


def local_history_text(user_id: int) -> str:
    items = local_history.get(user_id)
    return "\n".join(items) if items else "История пуста."


def save_stats() -> None:
    try:
        os.makedirs(os.path.dirname(STATS_FILE) or ".", exist_ok=True)
        tmp = STATS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)
        os.replace(tmp, STATS_FILE)
    except Exception as e:
        log.warning("Не удалось сохранить статистику: %s", e)


def save_authorized() -> None:
    try:
        os.makedirs(os.path.dirname(AUTH_FILE) or ".", exist_ok=True)
        tmp = AUTH_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sorted(authorized), f)
        os.replace(tmp, AUTH_FILE)
    except Exception as e:
        log.warning("Не удалось сохранить список авторизованных: %s", e)


def count(name: str, user_id: int | None = None) -> None:
    stats[name] = stats.get(name, 0) + 1
    if user_id is not None and user_id not in stats["users"]:
        stats["users"].append(user_id)
    save_stats()


def write_feedback_log(payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(FEEDBACK_LOG) or ".", exist_ok=True)
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Не удалось записать отзыв: %s", e)


def stats_text() -> str:
    def pct(a: int, b: int) -> str:
        return f"{a / b * 100:.0f}%" if b else "—"

    total = stats["requests"]
    pos = stats["feedback_positive"]
    neg = stats["feedback_negative"]
    avg = stats["latency_sum"] / stats["latency_n"] if stats["latency_n"] else None
    lines = [
        "📊 Статистика бота",
        f"Пользователей: {len(stats['users'])}",
        f"Вопросов: {total}",
        f"Ответ найден: {stats['answered']} ({pct(stats['answered'], total)})",
        f"Не найдено, предложен оператор: {stats['not_found']} ({pct(stats['not_found'], total)})",
        f"Ошибок: {stats['errors']} ({pct(stats['errors'], total)})",
        f"Вызовов оператора: {stats['escalations']}",
        f"Регистраций: {stats['registrations']}, запросов доступа: {stats['access_requests']}",
        f"Ответов оператора через бота: {stats['operator_replies']}",
        f"Оценки: 👍 {pos} / 👎 {neg}, полезных {pct(pos, pos + neg)}",
        f"Среднее время ответа: {avg:.1f} с" if avg is not None else "Среднее время ответа: —",
    ]
    return "\n".join(lines)


def remember_exchange(user_id: int, question: str, answer: str) -> int:
    global exchange_seq
    exchange_seq += 1
    exchanges[exchange_seq] = (user_id, question, answer)
    while len(exchanges) > 500:
        exchanges.popitem(last=False)
    return exchange_seq


def remember_thread(message_id: int, user_id: int) -> None:
    operator_threads[message_id] = user_id
    while len(operator_threads) > 1000:
        operator_threads.popitem(last=False)


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


class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        user = data.get("event_from_user")
        if user is None or is_authorized(user.id):
            return await handler(event, data)
        if isinstance(event, CallbackQuery):
            if event.data in OPEN_CALLBACKS:
                return await handler(event, data)
            await event.answer("Сначала пройдите авторизацию", show_alert=True)
            return None
        state: FSMContext = data["state"]
        current = await state.get_state()
        text = event.text or ""
        if current == Registration.waiting_id.state and text and not text.startswith("/"):
            return await handler(event, data)
        await begin_registration(event, state)
        return None


router.message.outer_middleware(AccessMiddleware())
router.callback_query.outer_middleware(AccessMiddleware())


async def keep_typing(chat_id: int) -> None:
    while True:
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            return
        await asyncio.sleep(4)


async def slow_notice(message: Message) -> None:
    await asyncio.sleep(SLOW_NOTICE_DELAY)
    try:
        await message.answer("⏳ Ищу ответ в документации, это может занять до минуты…")
    except Exception:
        log.warning("Не удалось отправить уведомление о долгом ответе")


async def send_text(
    message: Message, text: str, markup: InlineKeyboardMarkup | None = None, parse_mode: str = "HTML"
) -> None:
    parts = chunks(text, TEXT_CHUNK)
    for i, part in enumerate(parts):
        kb = markup if i == len(parts) - 1 else None
        try:
            content = part if parse_mode == "Markdown" else md_to_html(part)
            await message.answer(content, parse_mode=parse_mode, reply_markup=kb)
        except TelegramBadRequest:
            log.warning("Разметка не разобралась, отправляю простым текстом")
            await message.answer(part, parse_mode=None, reply_markup=kb)


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


async def send_photo(message: Message, photo, caption: str | None, markup, parse_mode: str = "HTML") -> bool:
    try:
        if caption:
            safe_caption = caption if parse_mode == "Markdown" else md_to_html(caption)
        else:
            safe_caption = None
        await message.answer_photo(
            photo,
            caption=safe_caption,
            parse_mode=parse_mode,
            reply_markup=markup,
        )
        return True
    except TelegramBadRequest:
        try:
            await message.answer_photo(photo, caption=caption, parse_mode=None, reply_markup=markup)
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
async def answer_question(message: Message, user: User, text: str, state: FSMContext | None = None) -> None:
    if is_operator_request(text):
        await escalate(message, user)
        return

    count("requests", user.id)
    local_history[user.id].append(f"Пользователь: {text}")
    started = time.monotonic()
    typing_task = asyncio.create_task(keep_typing(message.chat.id))
    notice_task = asyncio.create_task(slow_notice(message))
    try:
        r = await http.post(CORE_API_URL, json={"user_id": user.id, "text": text})
        if r.status_code in (401, 403):
            data = {"status": "unauthorized"}
        else:
            r.raise_for_status()
            data = r.json()
        if not isinstance(data, dict):
            raise ValueError("Неверный формат ответа сервера")
    except Exception:
        log.exception("Ошибка запроса к серверу")
        count("errors")
        await message.answer(
            "Произошла ошибка при обработке вашего запроса. Попробуйте позже или свяжитесь с оператором.",
            reply_markup=operator_keyboard(),
        )
        return
    finally:
        typing_task.cancel()
        notice_task.cancel()

    if data.get("status") == "unauthorized":
        authorized.discard(user.id)
        save_authorized()
        if not is_admin(user.id):
            if state is not None:
                await begin_registration(message, state)
            else:
                await message.answer(
                    "Ваш доступ к боту закрыт. Если это ошибка, напишите /start и пройдите авторизацию заново.",
                    reply_markup=ReplyKeyboardRemove(),
                )
            return

    if data.get("status") == "error":
        count("errors")
        await message.answer(
            data.get("reply") or "Не удалось обработать запрос.", reply_markup=operator_keyboard()
        )
        return

    stats["latency_sum"] += time.monotonic() - started
    stats["latency_n"] += 1
    reply = data.get("reply") or "Извините, я не смог сформировать ответ."
    parse_mode = data.get("parse_mode")
    if parse_mode not in ("Markdown", "HTML"):
        parse_mode = "HTML"
    local_history[user.id].append(f"Бот: {reply}")
    if data.get("suggest_operator"):
        count("not_found")
        markup = operator_keyboard()
    else:
        count("answered")
        markup = feedback_keyboard(remember_exchange(user.id, text, reply))

    image_path = data.get("image_path")
    if image_path:
        photo = await load_photo(image_path)
        if photo:
            short = len(reply) <= CAPTION_SAFE
            sent = await send_photo(message, photo, reply if short else None, markup if short else None, parse_mode)
            if sent and short:
                return
            if sent:
                await send_text(message, reply, markup, parse_mode)
                return
    await send_text(message, reply, markup, parse_mode)


async def answer_question_with_photo(
    message: Message, user: User, image_base64: str, text: str, state: FSMContext | None = None
) -> None:
    count("requests", user.id)
    local_history[user.id].append(f"Пользователь: [фото] {text}")
    started = time.monotonic()
    typing_task = asyncio.create_task(keep_typing(message.chat.id))
    notice_task = asyncio.create_task(slow_notice(message))
    try:
        r = await http.post(
            CORE_API_URL,
            json={"user_id": user.id, "text": text, "image_base64": image_base64},
        )
        if r.status_code in (401, 403):
            data = {"status": "unauthorized"}
        else:
            r.raise_for_status()
            data = r.json()
        if not isinstance(data, dict):
            raise ValueError("Неверный формат ответа сервера")
    except Exception:
        log.exception("Ошибка запроса к серверу с фото")
        count("errors")
        await message.answer(
            "Произошла ошибка при обработке фото. Попробуйте позже или свяжитесь с оператором.",
            reply_markup=operator_keyboard(),
        )
        return
    finally:
        typing_task.cancel()
        notice_task.cancel()

    if data.get("status") == "unauthorized":
        authorized.discard(user.id)
        save_authorized()
        if not is_admin(user.id) and state is not None:
            await begin_registration(message, state)
            return

    if data.get("status") == "error":
        count("errors")
        await message.answer(
            data.get("reply") or "Не удалось обработать фото.", reply_markup=operator_keyboard()
        )
        return

    stats["latency_sum"] += time.monotonic() - started
    stats["latency_n"] += 1
    reply = data.get("reply") or "Извините, я не смог распознать изображение."
    parse_mode = data.get("parse_mode")
    if parse_mode not in ("Markdown", "HTML"):
        parse_mode = "HTML"
    local_history[user.id].append(f"Бот: {reply}")
    if data.get("suggest_operator"):
        count("not_found")
        markup = operator_keyboard()
    else:
        count("answered")
        markup = feedback_keyboard(remember_exchange(user.id, f"[фото] {text}", reply))
    await send_text(message, reply, markup, parse_mode)


async def begin_registration(message: Message, state: FSMContext) -> None:
    await state.set_state(Registration.waiting_id)
    await message.answer(
        "Привет! Я корпоративный ассистент. Для начала работы, пожалуйста, введите ваш рабочий Email или табельный номер.",
        reply_markup=ReplyKeyboardRemove(),
    )


async def check_employee(user: User, employee_id: str) -> dict | None:
    r = await http.post(
        f"{BASE_API_URL}/auth",
        json={"telegram_id": user.id, "employee_id": employee_id, "username": user.username},
    )
    if r.status_code == 404:
        try:
            body = r.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("status") == "not_found":
            return None
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "success":
        return data
    return None


async def register_user(telegram_id: int, employee_id: str | None, admin_id: int) -> bool:
    if not employee_id or employee_id == "—":
        return False
    try:
        r = await http.post(
            f"{BASE_API_URL}/users",
            json={"telegram_id": telegram_id, "employee_id": employee_id, "admin_id": admin_id},
        )
        r.raise_for_status()
        return True
    except Exception:
        log.exception("Не удалось записать пользователя %s на сервер", telegram_id)
        return False


async def sync_authorized() -> None:
    try:
        r = await http.get(f"{BASE_API_URL}/users")
        r.raise_for_status()
        data = r.json()
        items = data.get("users", []) if isinstance(data, dict) else data
        ids = {int(x.get("telegram_id")) if isinstance(x, dict) else int(x) for x in items}
    except Exception as e:
        log.warning("Не удалось синхронизировать пользователей с сервером, работаю по кэшу: %s", e)
        return
    removed = authorized - ids
    authorized.clear()
    authorized.update(ids)
    save_authorized()
    log.info("Синхронизировано с сервером, авторизованных: %s, удалено из кэша: %s", len(ids), len(removed))


async def sync_loop() -> None:
    while True:
        await asyncio.sleep(SYNC_INTERVAL)
        await sync_authorized()


async def notify_admins(text: str, user_id: int, markup: InlineKeyboardMarkup | None = None) -> bool:
    # Сначала чат операторов; если туда не доставилось (бот не в группе, группу переименовали в супергруппу,
    # неверный ID), пишем лично администраторам. Админы должны хотя бы раз нажать /start у бота.
    groups = [[OPERATOR_CHAT_ID]] if OPERATOR_CHAT_ID else []
    groups.append(sorted(ADMIN_IDS))
    for targets in groups:
        delivered = False
        for target in targets:
            try:
                sent = await bot.send_message(target, text, reply_markup=markup)
                remember_thread(sent.message_id, user_id)
                delivered = True
            except Exception:
                log.exception("Не удалось отправить сообщение админам в %s", target)
        if delivered:
            return True
    return False


async def fetch_documents() -> list[str]:
    r = await http.get(f"{BASE_API_URL}/documents")
    r.raise_for_status()
    data = r.json()
    items = data.get("documents", []) if isinstance(data, dict) else data
    return [str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in items]


async def upload_document(name: str, content: bytes, mime: str) -> None:
    r = await http.post(f"{BASE_API_URL}/documents", files={"file": (name, content, mime)})
    r.raise_for_status()


async def show_admin_panel(message: Message) -> None:
    await message.answer("⚙️ Панель управления", reply_markup=admin_keyboard())


async def escalate(message: Message, user: User) -> None:
    if not OPERATOR_CHAT_ID:
        log.error("OPERATOR_CHAT_ID не задан")
        await message.answer("Оператор сейчас недоступен. Попробуйте, пожалуйста, позже.")
        return

    history = await fetch_dialog_history(user.id)
    text = (
        "🚨 Запрос на оператора!\n"
        f"Пользователь: {user.full_name} (@{user.username or 'нет username'}), ID: {user.id}\n"
        "↩️ Ответьте реплаем на это сообщение, и бот перешлёт ответ пользователю.\n\n"
        f"История диалога:\n{history}"
    )
    try:
        for part in chunks(text):
            sent = await bot.send_message(OPERATOR_CHAT_ID, part)
            remember_thread(sent.message_id, user.id)
    except Exception:
        log.exception("Не удалось отправить сообщение оператору")
        await message.answer("Не получилось связаться с оператором. Попробуйте, пожалуйста, чуть позже.")
        return

    count("escalations", user.id)
    await message.answer("Я передал ваш вопрос специалисту, он скоро ответит.")


# хэндлеры
@operator_router.message(F.chat.id == OPERATOR_CHAT_INT, F.reply_to_message)
async def operator_reply(message: Message):
    replied = message.reply_to_message
    if not replied.from_user or replied.from_user.id != bot.id:
        return
    target = operator_threads.get(replied.message_id)
    if target is None:
        found = re.search(r"ID:\s*(\d+)", replied.text or "")
        target = int(found.group(1)) if found else None
    if target is None:
        await message.reply("Не понял, какому пользователю отправить. Ответьте реплаем на сообщение «Запрос на оператора».")
        return
    if not message.text:
        await message.reply("Пока я пересылаю только текстовые ответы.")
        return
    try:
        await bot.send_message(target, f"👤 Ответ специалиста:\n{message.text}")
    except Exception:
        log.exception("Не удалось доставить ответ оператора пользователю %s", target)
        await message.reply("Не удалось доставить ответ: пользователь мог заблокировать бота.")
        return
    local_history[target].append(f"Оператор: {message.text}")
    count("operator_replies")
    await message.reply("✅ Ответ отправлен пользователю.")


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    user_name = html.escape(message.from_user.first_name or "")
    greeting_text = (
        f"Здравствуйте, {user_name}! 👋\n\n"
        "Я - <b>БАРС-Помощник</b>, виртуальный ассистент компании <b>БАРС Груп</b>. "
        "Я изучил всю внутреннюю документацию и готов ответить на ваши вопросы.\n\n"
        "Если понадобится живой человек, нажмите кнопку «Связаться с оператором» внизу или напишите «оператор»."
    )
    await message.answer(greeting_text, parse_mode=ParseMode.HTML, reply_markup=main_keyboard(is_admin(message.from_user.id)))


@router.message(Command("help"))
async def cmd_help(message: Message):
    text = HELP_TEXT + (ADMIN_HELP_TEXT if is_admin(message.from_user.id) else "")
    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Действие отменено.")


@router.message(Command("admin"))
async def cmd_admin(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("Команда недоступна.")
        return
    await show_admin_panel(message)


@router.message(F.text == ADMIN_BUTTON_TEXT)
async def handle_admin_button(message: Message):
    if not is_admin(message.from_user.id):
        return
    await show_admin_panel(message)


@router.message(Command("stats"))
async def cmd_stats(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("Команда недоступна.")
        return
    await message.answer(stats_text())


@router.message(Command("stop"))
async def cmd_stop(message: Message):
    await message.answer(
        "Всего доброго! Если появятся вопросы - я всегда на связи. 👋",
        reply_markup=ReplyKeyboardRemove(),
    )


@router.message(Command("operator"))
async def cmd_operator(message: Message):
    await escalate(message, message.from_user)


@router.message(F.text == OPERATOR_BUTTON_TEXT)
async def handle_operator_button(message: Message):
    await escalate(message, message.from_user)


@router.callback_query(F.data == "call_operator")
async def process_call_operator(callback: CallbackQuery):
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass
    await escalate(callback.message, callback.from_user)


@router.callback_query(F.data == "call_admin")
async def process_call_admin(callback: CallbackQuery):
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass
    user = callback.from_user
    entered = pending_access.get(user.id, "—")
    text = (
        f"🔐 Пользователь @{user.username or 'нет username'} (ID: {user.id}) просит доступ.\n"
        f"Имя: {user.full_name}\n"
        f"Введенные данные: {entered}"
    )
    if not await notify_admins(text, user.id, access_request_keyboard(user.id)):
        await callback.message.answer("Не получилось связаться с администратором. Попробуйте, пожалуйста, позже.")
        return
    count("access_requests")
    await callback.message.answer("Я передал вашу заявку специалисту. Как только доступ откроют, я напишу вам.")


@router.callback_query(F.data.startswith("access:"))
async def process_access_decision(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("Недоступно", show_alert=True)
        return
    try:
        _, action, raw_id = callback.data.split(":")
        target = int(raw_id)
    except ValueError:
        await callback.answer()
        return
    approved = action == "approve"
    entered = pending_access.pop(target, None)
    if entered is None and isinstance(callback.message, Message):
        found = re.search(r"Введенные данные:\s*(.+)", callback.message.text or "")
        entered = found.group(1).strip() if found else None
    saved = True
    if approved:
        authorized.add(target)
        save_authorized()
        count("registrations", target)
        saved = await register_user(target, entered, callback.from_user.id)
    await callback.answer("Готово")
    if isinstance(callback.message, Message):
        verdict = "✅ Одобрено" if approved else "❌ Отклонено"
        if not saved:
            verdict += " (⚠️ на сервере не сохранено, доступ открыт только в боте)"
        try:
            await callback.message.edit_text(
                f"{callback.message.text}\n\n{verdict}: @{callback.from_user.username or callback.from_user.id}",
                reply_markup=None,
            )
        except TelegramBadRequest:
            pass
    try:
        if approved:
            await bot.send_message(
                target,
                "Доступ открыт! Чем я могу помочь?",
                reply_markup=main_keyboard(is_admin(target)),
            )
        else:
            await bot.send_message(target, "К сожалению, в доступе отказано. Уточните данные у администратора.")
    except Exception:
        log.exception("Не удалось уведомить пользователя %s о решении", target)


@router.callback_query(F.data.startswith("adm:"))
async def process_admin_action(callback: CallbackQuery, state: FSMContext):
    if not is_admin(callback.from_user.id):
        await callback.answer("Недоступно", show_alert=True)
        return
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    action = callback.data.split(":", 1)[1]
    if action == "stats":
        await callback.message.answer(stats_text())
    elif action == "upload":
        await state.set_state(AdminPanel.waiting_document)
        await callback.message.answer(
            "Отправьте документ файлом (PDF или DOCX). Для отмены — /cancel."
        )
    elif action == "docs":
        try:
            names = await fetch_documents()
        except Exception:
            log.exception("Не удалось получить список документов")
            await callback.message.answer("Не удалось получить список документов.")
            return
        await callback.message.answer("📚 Документы в базе:\n" + "\n".join(f"• {n}" for n in names) if names else "База документов пуста.")
    elif action == "close":
        try:
            await callback.message.delete()
        except TelegramBadRequest:
            pass


@router.callback_query(F.data.startswith("fb:"))
async def process_feedback(callback: CallbackQuery):
    try:
        _, raw_id, value = callback.data.split(":")
        exchange_id = int(raw_id)
    except ValueError:
        await callback.answer()
        return
    if exchange_id in rated:
        await callback.answer("Вы уже оценили этот ответ")
        return
    rated.add(exchange_id)
    positive = value == "1"
    user = callback.from_user
    question, answer = "", ""
    exchange = exchanges.get(exchange_id)
    if exchange and exchange[0] == user.id:
        question, answer = exchange[1], exchange[2]
    await callback.answer("Спасибо за оценку!")
    if isinstance(callback.message, Message):
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass
    count("feedback_positive" if positive else "feedback_negative", user.id)
    write_feedback_log(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user.id,
            "rating": "positive" if positive else "negative",
            "question": question,
            "answer": answer,
        }
    )


@router.message(Registration.waiting_id, F.text)
async def process_employee_id(message: Message, state: FSMContext):
    employee_id = message.text.strip()
    if not employee_id or len(employee_id) > EMPLOYEE_ID_MAX_LEN:
        await message.answer("Введите, пожалуйста, корректный рабочий Email или табельный номер.")
        return
    user = message.from_user
    try:
        employee = await check_employee(user, employee_id)
    except Exception:
        log.exception("Ошибка авторизации пользователя %s", user.id)
        await message.answer("Не удалось проверить данные: сервер недоступен. Попробуйте ещё раз чуть позже.")
        return
    await state.clear()
    if employee:
        authorized.add(user.id)
        save_authorized()
        pending_access.pop(user.id, None)
        count("registrations", user.id)
        full_name = employee.get("full_name")
        greeting = f"Отлично, {full_name}, вы успешно авторизованы!" if full_name else "Отлично, вы успешно авторизованы!"
        await message.answer(
            f"{greeting} Чем я могу помочь?",
            reply_markup=main_keyboard(is_admin(user.id)),
        )
        return
    pending_access[user.id] = employee_id
    await message.answer(
        "Я не нашел вас в базе сотрудников. Возможно, вас еще не добавили. "
        "Вы можете написать администратору или отправить данные заново через /start.",
        reply_markup=admin_contact_keyboard(),
    )


@router.message(AdminPanel.waiting_document, F.document)
async def process_admin_document(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        await state.clear()
        return
    doc = message.document
    name = doc.file_name or "document"
    if not name.lower().endswith(DOC_EXTENSIONS):
        await message.answer("Поддерживаются только файлы PDF и DOCX. Отправьте другой файл или /cancel.")
        return
    if doc.file_size and doc.file_size > DOC_MAX_SIZE:
        await message.answer("Файл слишком большой, максимум 20 МБ. Отправьте другой файл или /cancel.")
        return
    try:
        buf = io.BytesIO()
        await bot.download(doc, destination=buf)
        await upload_document(name, buf.getvalue(), doc.mime_type or "application/octet-stream")
    except Exception:
        log.exception("Не удалось загрузить документ %s", name)
        await message.answer("Не удалось добавить документ в базу знаний. Попробуйте позже.")
        return
    await state.clear()
    await message.answer(f"✅ Документ «{name}» принят. Индексация идёт в фоне, обычно это занимает несколько минут.")


@router.message(AdminPanel.waiting_document)
async def process_admin_document_invalid(message: Message):
    await message.answer("Пришлите документ файлом (PDF или DOCX) или отправьте /cancel.")


@router.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    log.info("Сообщение от %s, длина %s", message.from_user.id, len(message.text))
    pending = pending_photos.pop(message.from_user.id, None)
    if pending and time.monotonic() - pending[0] <= PHOTO_WAIT:
        await answer_question_with_photo(message, message.from_user, pending[1], message.text.strip(), state)
        return
    await answer_question(message, message.from_user, message.text.strip(), state)


@router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    photo = message.photo[-1]
    log.info("Фото от %s, размер %s", message.from_user.id, photo.file_size)
    if photo.file_size and photo.file_size > PHOTO_MAX_SIZE:
        await message.answer("Фото слишком большое, максимум 10 МБ. Отправьте другое изображение.")
        return
    try:
        buf = io.BytesIO()
        await bot.download(photo, destination=buf)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        log.exception("Не удалось скачать фото")
        await message.answer("Не удалось получить фото. Попробуйте ещё раз.")
        return
    user = message.from_user
    caption = (message.caption or "").strip()
    if caption:
        pending_photos.pop(user.id, None)
        await answer_question_with_photo(message, user, image_base64, caption, state)
        return
    now = time.monotonic()
    for uid in [u for u, (t, _) in pending_photos.items() if now - t > PHOTO_WAIT]:
        pending_photos.pop(uid, None)
    pending_photos[user.id] = (now, image_base64)
    await message.answer(
        "Скриншот получил. Напишите, что вы хотите узнать о нём, например: «Как настроить этот фильтр?». "
        "Я отвечу по документации с учётом изображения."
    )


@router.message()
async def handle_non_text(message: Message):
    await message.answer("Я понимаю только текст и фотографии. Пожалуйста, напишите ваш вопрос текстом или пришлите скриншот.")


# запуск
async def main():
    global http, bot

    if PROXY_URL:
        bot = Bot(token=BOT_TOKEN, session=AiohttpSession(proxy=PROXY_URL))
    else:
        bot = Bot(token=BOT_TOKEN)

    http = httpx.AsyncClient(timeout=httpx.Timeout(CORE_TIMEOUT, connect=10.0), trust_env=False)
    sync_task = None
    try:
        await sync_authorized()
        sync_task = asyncio.create_task(sync_loop())
        await bot.delete_webhook(drop_pending_updates=True)
        await bot.set_my_commands(BASE_COMMANDS)
        for admin_id in ADMIN_IDS:
            try:
                await bot.set_my_commands(
                    BASE_COMMANDS + ADMIN_COMMANDS,
                    scope=BotCommandScopeChat(chat_id=admin_id),
                )
            except Exception:
                log.warning("Не удалось задать меню для администратора %s", admin_id)
        log.info(
            "Бот запущен, сервер: %s, прокси: %s, админов: %s",
            BASE_API_URL,
            "да" if PROXY_URL else "нет",
            len(ADMIN_IDS),
        )
        await dp.start_polling(bot)
    finally:
        if sync_task:
            sync_task.cancel()
        await http.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())