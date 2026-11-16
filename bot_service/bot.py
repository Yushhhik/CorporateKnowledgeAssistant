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
from datetime import datetime, timedelta

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
    InputMediaPhoto,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    User,
)

import analytics

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
LOOKUP_MIN_INTERVAL = float(os.getenv("LOOKUP_MIN_INTERVAL", "5"))
SUPPORT_HOURS = os.getenv("SUPPORT_HOURS", "").strip()
STATS_FILE = os.getenv("STATS_FILE", "/data/stats.json")
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", "/data/feedback.jsonl")
AUTH_FILE = os.getenv("AUTH_FILE", "/data/authorized.json")
ADMIN_IDS = {int(x) for x in re.findall(r"-?\d+", os.getenv("ADMIN_IDS", ""))}
TG_LIMIT = 4000
TEXT_CHUNK = 3500
CAPTION_SAFE = 900
OPERATOR_BUTTON_TEXT = "Связаться с оператором"
ADMIN_BUTTON_TEXT = "Панель управления"
FOLLOWUP_BUTTON_TEXT = "Уточнить вопрос"
NEW_TOPIC_BUTTON_TEXT = "Новая тема"
RATE_PROMPT_TEXT = "Прежде чем перейти к новой теме, оцените, пожалуйста, предыдущий ответ:"
MAX_FOLLOWUPS = 5
FOLLOWUP_WAIT = 900
EMPLOYEE_ID_MAX_LEN = 100
DOC_EXTENSIONS = (".pdf", ".docx")
DOC_MAX_SIZE = 20 * 1024 * 1024
PHOTO_MAX_SIZE = 10 * 1024 * 1024
PHOTO_WAIT = 600
STATS_DEFAULT_PERIOD = "7d"

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

ADMIN_HELP_TEXT = "\n/admin — панель управления\n/stats — аналитика бота\n/cancel — отменить текущее действие"

BASE_COMMANDS = [
    BotCommand(command="start", description="Начать работу"),
    BotCommand(command="help", description="Что умеет бот"),
    BotCommand(command="operator", description="Связаться с оператором"),
    BotCommand(command="stop", description="Попрощаться с ботом"),
]

ADMIN_COMMANDS = [
    BotCommand(command="admin", description="Панель управления"),
    BotCommand(command="stats", description="Аналитика бота"),
    BotCommand(command="cancel", description="Отменить действие"),
]

OPEN_CALLBACKS = {"call_admin", "call_operator"}
GUEST_COMMANDS = {"/start", "/stop", "/operator", "/help", "/cancel"}

COUNTERS = (
    "requests",
    "answered",
    "not_found",
    "errors",
    "escalations",
    "operator_replies",
    "feedback_positive",
    "feedback_negative",
    "followups",
    "new_topics",
    "registrations",
    "access_requests",
)

EVENT_MAP = {
    "answered": ("question", "answered"),
    "not_found": ("question", "not_found"),
    "errors": ("question", "error"),
    "escalations": ("escalation", None),
    "operator_replies": ("operator_reply", None),
    "feedback_positive": ("feedback", "positive"),
    "feedback_negative": ("feedback", "negative"),
    "followups": ("followup", None),
    "new_topics": ("new_topic", None),
    "registrations": ("registration", None),
    "access_requests": ("access_request", None),
}

local_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
exchanges: OrderedDict[int, tuple[int, str, str]] = OrderedDict()
exchange_seq = int(time.time())
rated: set[int] = set()
operator_threads: OrderedDict[int, int] = OrderedDict()
pending_access: dict[int, str] = {}
pending_photos: dict[int, tuple[float, str]] = {}
followup_chain: dict[int, int] = {}
awaiting_followup: dict[int, float] = {}
db_admins: set[int] = set()
access_sent: dict[int, str] = {}
menu_admins: set[int] = set()
last_sync_attempt = 0.0
lookup_checked: dict[int, float] = {}
lookup_lock = asyncio.Lock()


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


def followups_left(user_id: int) -> int:
    return MAX_FOLLOWUPS - followup_chain.get(user_id, 0)


def register_question(user_id: int) -> None:
    """Вопрос после кнопки «Уточнить вопрос» продолжает цепочку, любой другой начинает новую."""
    asked = awaiting_followup.pop(user_id, None)
    if asked is not None and time.monotonic() - asked <= FOLLOWUP_WAIT:
        followup_chain[user_id] = followup_chain.get(user_id, 0) + 1
        count("followups", user_id)
    else:
        followup_chain.pop(user_id, None)


def reset_followups(user_id: int) -> None:
    followup_chain.pop(user_id, None)
    awaiting_followup.pop(user_id, None)


def followup_keyboard(exchange_id: int, user_id: int) -> InlineKeyboardMarkup:
    row = []
    if followups_left(user_id) > 0:
        row.append(InlineKeyboardButton(text=FOLLOWUP_BUTTON_TEXT, callback_data=f"fu:{exchange_id}"))
    row.append(InlineKeyboardButton(text=NEW_TOPIC_BUTTON_TEXT, callback_data=f"nt:{exchange_id}"))
    return InlineKeyboardMarkup(inline_keyboard=[row])


def not_found_keyboard(user_id: int) -> InlineKeyboardMarkup:
    rows = []
    if followups_left(user_id) > 0:
        rows.append([InlineKeyboardButton(text=FOLLOWUP_BUTTON_TEXT, callback_data="fu:nf")])
    rows.append([InlineKeyboardButton(text="Связаться с оператором", callback_data="call_operator")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def rating_keyboard(exchange_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍 Полезно", callback_data=f"fb:{exchange_id}:1"),
                InlineKeyboardButton(text="👎 Не помогло", callback_data=f"fb:{exchange_id}:0"),
            ],
            [InlineKeyboardButton(text="Пропустить", callback_data=f"fb:{exchange_id}:skip")],
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
            [InlineKeyboardButton(text="Статистика", callback_data="adm:stats")],
            [InlineKeyboardButton(text="Добавить документ в базу знаний", callback_data="adm:upload")],
            [InlineKeyboardButton(text="Список документов", callback_data="adm:docs")],
            [InlineKeyboardButton(text="Закрыть", callback_data="adm:close")],
        ]
    )


def stats_keyboard(period: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=("• " if key == period else "") + label, callback_data=f"st:view:{key}")
        for key, label in analytics.PERIODS.items()
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            buttons[:2],
            buttons[2:],
            [
                InlineKeyboardButton(text="Слабые места", callback_data=f"st:weak:{period}"),
                InlineKeyboardButton(text="Частые вопросы", callback_data=f"st:top:{period}"),
            ],
            [
                InlineKeyboardButton(text="Экспорт CSV", callback_data=f"st:csv:{period}"),
                InlineKeyboardButton(text="Обновить", callback_data=f"st:view:{period}"),
            ],
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
    return user_id in ADMIN_IDS or user_id in db_admins


def is_authorized(user_id: int) -> bool:
    return user_id in authorized or is_admin(user_id)


WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
WEEKDAYS_ACC = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]


def parse_schedule(raw: str):
    m = re.fullmatch(
        r"\s*(пн|вт|ср|чт|пт|сб|вс)(?:\s*-\s*(пн|вт|ср|чт|пт|сб|вс))?\s+(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*",
        raw.lower(),
    )
    if not m:
        return None
    first = WEEKDAYS.index(m.group(1))
    last = WEEKDAYS.index(m.group(2)) if m.group(2) else first
    days = {(first + i) % 7 for i in range((last - first) % 7 + 1)}
    start = int(m.group(3)) * 60 + int(m.group(4))
    end = int(m.group(5)) * 60 + int(m.group(6))
    if not 0 <= start < end <= 24 * 60:
        return None
    return days, start, end


SCHEDULE = parse_schedule(SUPPORT_HOURS) if SUPPORT_HOURS else None
if SUPPORT_HOURS and SCHEDULE is None:
    log.warning("SUPPORT_HOURS не распознан, ожидается формат «пн-пт 09:00-18:00»: %s", SUPPORT_HOURS)


def support_note(now: datetime | None = None) -> str:
    if not SCHEDULE:
        return ""
    days, start, end = SCHEDULE
    now = now or datetime.now(analytics.TZ)
    minutes = now.hour * 60 + now.minute
    head = f"\n\n🕘 Часы работы поддержки: {SUPPORT_HOURS}."
    if now.weekday() in days and start <= minutes < end:
        return head + " Сейчас рабочее время, ответ придёт в ближайшее время."
    at = f"{start // 60:02d}:{start % 60:02d}"
    for offset in range(8):
        day = now + timedelta(days=offset)
        if day.weekday() in days and (offset > 0 or minutes < start):
            when = "сегодня" if offset == 0 else "завтра" if offset == 1 else f"в {WEEKDAYS_ACC[day.weekday()]}"
            return head + f" Сейчас нерабочее время, ответ придёт {when} с {at}."
    return head


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


def count(name: str, user_id: int | None = None, **fields) -> None:
    stats[name] = stats.get(name, 0) + 1
    if user_id is not None and user_id not in stats["users"]:
        stats["users"].append(user_id)
    save_stats()
    mapping = EVENT_MAP.get(name)
    if mapping:
        analytics.record(mapping[0], user_id, outcome=mapping[1], **fields)


def write_feedback_log(payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(FEEDBACK_LOG) or ".", exist_ok=True)
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Не удалось записать отзыв: %s", e)


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
        is_callback = isinstance(event, CallbackQuery)
        if not is_callback and event.chat.type != "private":
            return await handler(event, data)
        if await lookup_user(user.id):
            await greet_found(event, user)
            return await handler(event, data)
        if is_callback:
            if event.data in OPEN_CALLBACKS:
                return await handler(event, data)
            await event.answer("Сначала пройдите авторизацию", show_alert=True)
            return None
        state: FSMContext = data["state"]
        text = (event.text or "").strip()
        command = text.split()[0].split("@")[0].lower() if text.startswith("/") else None
        if command in GUEST_COMMANDS:
            return await handler(event, data)
        if text == OPERATOR_BUTTON_TEXT or (command is None and is_operator_request(text)):
            await escalate(event, user)
            return None
        if command is None and text and await state.get_state() == Registration.waiting_id.state:
            return await handler(event, data)
        if user.id in access_sent:
            await event.answer(
                "Ваша заявка на доступ на рассмотрении. Как только её одобрят, я напишу вам. "
                "Если ввели данные с ошибкой, отправьте /start и введите их заново.",
                reply_markup=main_keyboard(),
            )
            return None
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


async def reset_server_dialog(user_id: int) -> None:
    try:
        r = await http.post(f"{BASE_API_URL}/reset/{user_id}", timeout=10.0)
        r.raise_for_status()
    except Exception as e:
        log.warning("Не удалось сбросить контекст диалога для %s: %s", user_id, e)


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
    register_question(user.id)
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
        count("errors", user.id, question=text)
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
        if state is not None:
            await begin_registration(message, state)
        else:
            await message.answer(
                "Ваш доступ к боту закрыт. Если это ошибка, напишите /start и пройдите авторизацию заново.",
                reply_markup=main_keyboard(is_admin(user.id)),
            )
        return

    if data.get("status") == "error":
        count("errors", user.id, question=text)
        await message.answer(
            data.get("reply") or "Не удалось обработать запрос.", reply_markup=operator_keyboard()
        )
        return

    latency = time.monotonic() - started
    stats["latency_sum"] += latency
    stats["latency_n"] += 1
    reply = data.get("reply") or "Извините, я не смог сформировать ответ."
    parse_mode = data.get("parse_mode")
    if parse_mode not in ("Markdown", "HTML"):
        parse_mode = "HTML"
    local_history[user.id].append(f"Бот: {reply}")
    if data.get("suggest_operator"):
        count("not_found", user.id, question=text, latency=latency)
        markup = not_found_keyboard(user.id)
    else:
        count("answered", user.id, question=text, latency=latency)
        markup = followup_keyboard(remember_exchange(user.id, text, reply), user.id)

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
    register_question(user.id)
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
        count("errors", user.id, question=f"[фото] {text}")
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
        if state is not None:
            await begin_registration(message, state)
        else:
            await message.answer(
                "Ваш доступ к боту закрыт. Если это ошибка, напишите /start и пройдите авторизацию заново.",
                reply_markup=main_keyboard(is_admin(user.id)),
            )
        return

    if data.get("status") == "error":
        count("errors", user.id, question=f"[фото] {text}")
        await message.answer(
            data.get("reply") or "Не удалось обработать фото.", reply_markup=operator_keyboard()
        )
        return

    latency = time.monotonic() - started
    stats["latency_sum"] += latency
    stats["latency_n"] += 1
    reply = data.get("reply") or "Извините, я не смог распознать изображение."
    parse_mode = data.get("parse_mode")
    if parse_mode not in ("Markdown", "HTML"):
        parse_mode = "HTML"
    local_history[user.id].append(f"Бот: {reply}")
    if data.get("suggest_operator"):
        count("not_found", user.id, question=f"[фото] {text}", latency=latency)
        markup = not_found_keyboard(user.id)
    else:
        count("answered", user.id, question=f"[фото] {text}", latency=latency)
        markup = followup_keyboard(remember_exchange(user.id, f"[фото] {text}", reply), user.id)
    await send_text(message, reply, markup, parse_mode)


async def begin_registration(message: Message, state: FSMContext, name: str | None = None) -> None:
    await state.set_state(Registration.waiting_id)
    if name is None:
        await message.answer(
            "Для начала работы введите, пожалуйста, ваш рабочий Email или табельный номер.",
            reply_markup=main_keyboard(),
        )
        return
    await message.answer(
        f"Здравствуйте, {html.escape(name)}! 👋\n\n"
        "Я — <b>БАРС-Помощник</b>, виртуальный ассистент компании <b>БАРС Груп</b>. "
        "Чтобы начать работу, введите, пожалуйста, ваш рабочий Email или табельный номер.\n\n"
        "Если возникнут трудности, нажмите «Связаться с оператором» внизу.",
        parse_mode=ParseMode.HTML,
        reply_markup=main_keyboard(),
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


def parse_user_row(row) -> tuple[int | None, bool]:
    if isinstance(row, dict):
        raw_id = row.get("telegram_id")
        flag = bool(row.get("is_admin") or row.get("admin") or str(row.get("role", "")).lower() == "admin")
    else:
        raw_id, flag = row, False
    try:
        return int(raw_id), flag
    except (TypeError, ValueError):
        return None, flag


async def sync_authorized() -> None:
    global last_sync_attempt
    last_sync_attempt = time.monotonic()
    try:
        r = await http.get(f"{BASE_API_URL}/users", timeout=10.0)
        r.raise_for_status()
        data = r.json()
        items = data.get("users", []) if isinstance(data, dict) else data
        parsed = [parse_user_row(x) for x in items]
    except Exception as e:
        log.warning("Не удалось синхронизировать пользователей с сервером, работаю по кэшу: %s", e)
        return
    ids = {uid for uid, _ in parsed if uid is not None}
    admins = {uid for uid, flag in parsed if uid is not None and flag}
    removed = authorized - ids
    authorized.clear()
    authorized.update(ids)
    db_admins.clear()
    db_admins.update(admins)
    save_authorized()
    log.info(
        "Синхронизировано с сервером, авторизованных: %s, админов: %s, удалено из кэша: %s",
        len(ids),
        len(admins),
        len(removed),
    )


async def lookup_user(user_id: int) -> bool:
    """Проверяет по Telegram ID, внесён ли человек в БД. Если да, пускает без авторизации."""
    if user_id in authorized:
        return True
    now = time.monotonic()
    if now - lookup_checked.get(user_id, 0.0) < LOOKUP_MIN_INTERVAL:
        return False
    lookup_checked[user_id] = now
    if len(lookup_checked) > 5000:
        for uid in [u for u, t in lookup_checked.items() if now - t > 3600]:
            lookup_checked.pop(uid, None)
    try:
        r = await http.get(f"{BASE_API_URL}/users/by_telegram/{user_id}", timeout=10.0)
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code == 404 and isinstance(body, dict) and body.get("status") == "not_found":
            return False
        r.raise_for_status()
        if not isinstance(body, dict) or body.get("status") != "success":
            raise ValueError("Неверный ответ сервера")
    except Exception as e:
        # Сервер недоступен или эндпоинта нет: проверяем по общему списку пользователей, как раньше
        log.warning("Проверка по Telegram ID не удалась (%s), сверяю со списком пользователей", e)
        async with lookup_lock:
            if user_id in authorized:
                return True
            if time.monotonic() - last_sync_attempt >= LOOKUP_MIN_INTERVAL:
                await sync_authorized()
            return user_id in authorized
    authorized.add(user_id)
    if parse_user_row(body)[1]:
        db_admins.add(user_id)
    save_authorized()
    return True


async def ensure_admin_menu(user_id: int) -> None:
    if user_id in menu_admins or not is_admin(user_id):
        return
    try:
        await bot.set_my_commands(
            BASE_COMMANDS + ADMIN_COMMANDS,
            scope=BotCommandScopeChat(chat_id=user_id),
        )
        menu_admins.add(user_id)
    except Exception:
        log.warning("Не удалось задать меню для администратора %s", user_id)


async def greet_found(event, user: User) -> None:
    if not isinstance(event, Message) or (event.text or "").startswith("/start"):
        return
    await event.answer(
        "Вы найдены в базе сотрудников ✅ Можете задавать вопросы по документации.",
        reply_markup=main_keyboard(is_admin(user.id)),
    )
    await ensure_admin_menu(user.id)


async def sync_loop() -> None:
    while True:
        await asyncio.sleep(SYNC_INTERVAL)
        await sync_authorized()


async def notify_admins(text: str, user_id: int, markup: InlineKeyboardMarkup | None = None) -> bool:
    # Сначала чат операторов; если туда не доставилось (бот не в группе, группу переименовали в супергруппу,
    # неверный ID), пишем лично администраторам. Админы должны хотя бы раз нажать /start у бота.
    groups = [[OPERATOR_CHAT_ID]] if OPERATOR_CHAT_ID else []
    groups.append(sorted(ADMIN_IDS | db_admins))
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


async def fetch_documents(admin_id: int) -> list[str]:
    r = await http.get(f"{BASE_API_URL}/documents", params={"admin_id": admin_id})
    r.raise_for_status()
    data = r.json()
    items = data.get("documents", []) if isinstance(data, dict) else data
    return [str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in items]


async def upload_document(name: str, content: bytes, mime: str, admin_id: int) -> None:
    """Отправляет файл на сервер (multipart POST /documents). Ошибку сервера поднимаем с его текстом."""
    r = await http.post(
        f"{BASE_API_URL}/documents", params={"admin_id": admin_id}, files={"file": (name, content, mime)}
    )
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail") or r.text
        except Exception:
            detail = r.text
        raise RuntimeError(f"сервер ответил {r.status_code}: {str(detail)[:200]}")


async def submit_access_request(user: User, entered: str) -> bool:
    text = (
        f"🔐 Пользователь @{user.username or 'нет username'} (ID: {user.id}) просит доступ.\n"
        f"Имя: {user.full_name}\n"
        f"Введенные данные: {entered}"
    )
    if not await notify_admins(text, user.id, access_request_keyboard(user.id)):
        return False
    access_sent[user.id] = entered
    count("access_requests", user.id)
    return True


async def show_admin_panel(message: Message) -> None:
    await message.answer("Панель управления", reply_markup=admin_keyboard())


async def show_dashboard(message: Message, period: str, edit: bool = False) -> None:
    try:
        png, caption = await asyncio.to_thread(analytics.build_dashboard, period)
    except Exception:
        log.exception("Не удалось сформировать аналитику")
        await message.answer("Не удалось сформировать отчёт. Попробуйте позже.")
        return
    keyboard = stats_keyboard(period)
    if png is None:
        if edit:
            try:
                await message.edit_text(caption, parse_mode=ParseMode.HTML, reply_markup=keyboard)
            except TelegramBadRequest:
                pass
        else:
            await message.answer(caption, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        return
    photo = BufferedInputFile(png, filename="analytics.png")
    if edit:
        try:
            await message.edit_media(
                InputMediaPhoto(media=photo, caption=caption, parse_mode=ParseMode.HTML),
                reply_markup=keyboard,
            )
        except TelegramBadRequest:
            log.warning("Не удалось обновить сообщение с аналитикой")
    else:
        try:
            await message.answer_photo(photo, caption=caption, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        except TelegramBadRequest:
            await message.answer_photo(
                BufferedInputFile(png, filename="analytics.png"),
                caption=re.sub(r"<[^>]+>", "", caption),
                reply_markup=keyboard,
            )


async def escalate(message: Message, user: User) -> None:
    if not OPERATOR_CHAT_ID:
        log.error("OPERATOR_CHAT_ID не задан")
        await message.answer("Оператор сейчас недоступен. Попробуйте, пожалуйста, позже.")
        return

    history = await fetch_dialog_history(user.id)
    guest_note = "" if is_authorized(user.id) else "🔒 Пользователь ещё не авторизован\n"
    text = (
        "🚨 Запрос на оператора!\n"
        f"Пользователь: {user.full_name} (@{user.username or 'нет username'}), ID: {user.id}\n"
        f"{guest_note}"
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
    await message.answer(
        "✅ Я передал ваш вопрос и историю диалога оператору. Специалист свяжется с вами в ближайшее время, "
        "ответ придёт сюда, в этот чат." + support_note()
    )


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
    count("operator_replies", target)
    await message.reply("✅ Ответ отправлен пользователю.")


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    user = message.from_user
    access_sent.pop(user.id, None)
    if not is_authorized(user.id):
        await begin_registration(message, state, user.first_name or "")
        return
    if is_admin(user.id):
        await ensure_admin_menu(user.id)
    user_name = html.escape(user.first_name or "")
    greeting_text = (
        f"Здравствуйте, {user_name}! 👋\n\n"
        "Я - <b>БАРС-Помощник</b>, виртуальный ассистент компании <b>БАРС Груп</b>. "
        "Я изучил всю внутреннюю документацию и готов ответить на ваши вопросы.\n\n"
        "Если понадобится живой человек, нажмите кнопку «Связаться с оператором» внизу или напишите «оператор»."
    )
    await message.answer(greeting_text, parse_mode=ParseMode.HTML, reply_markup=main_keyboard(is_admin(user.id)))


@router.message(Command("help"))
async def cmd_help(message: Message):
    text = HELP_TEXT + (ADMIN_HELP_TEXT if is_admin(message.from_user.id) else "")
    await message.answer(text, parse_mode=ParseMode.HTML)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    pending_photos.pop(message.from_user.id, None)
    reset_followups(message.from_user.id)
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
    await show_dashboard(message, STATS_DEFAULT_PERIOD)


@router.message(Command("stop"))
async def cmd_stop(message: Message, state: FSMContext):
    await state.clear()
    pending_photos.pop(message.from_user.id, None)
    reset_followups(message.from_user.id)
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
    entered = pending_access.get(user.id)
    if entered is None:
        await callback.message.answer("Отправьте /start и введите ваш рабочий Email или табельный номер.")
        return
    if await submit_access_request(user, entered):
        await callback.message.answer("Заявка отправлена администратору. Как только её одобрят, я напишу вам.")
    else:
        await callback.message.answer("Не получилось связаться с администратором. Попробуйте, пожалуйста, позже.")


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
    if target in authorized:
        await callback.answer("Этот пользователь уже в базе")
        if isinstance(callback.message, Message):
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramBadRequest:
                pass
        return
    access_sent.pop(target, None)
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
        await show_dashboard(callback.message, STATS_DEFAULT_PERIOD)
    elif action == "upload":
        await state.set_state(AdminPanel.waiting_document)
        await callback.message.answer(
            "Отправьте документ файлом (PDF или DOCX). Для отмены — /cancel."
        )
    elif action == "docs":
        try:
            names = await fetch_documents(callback.from_user.id)
        except Exception:
            log.exception("Не удалось получить список документов")
            await callback.message.answer("Не удалось получить список документов.")
            return
        await callback.message.answer("Документы в базе:\n" + "\n".join(f"• {n}" for n in names) if names else "База документов пуста.")
    elif action == "close":
        try:
            await callback.message.delete()
        except TelegramBadRequest:
            pass


@router.callback_query(F.data.startswith("st:"))
async def process_stats(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("Недоступно", show_alert=True)
        return
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    try:
        _, action, period = callback.data.split(":")
    except ValueError:
        return
    if period not in analytics.PERIODS:
        return
    message = callback.message
    try:
        if action == "view":
            await show_dashboard(message, period, edit=True)
        elif action == "weak":
            await message.answer(await asyncio.to_thread(analytics.weak_spots_text, period), parse_mode=ParseMode.HTML)
        elif action == "top":
            await message.answer(await asyncio.to_thread(analytics.top_questions_text, period), parse_mode=ParseMode.HTML)
        elif action == "csv":
            data = await asyncio.to_thread(analytics.export_csv, period)
            if data is None:
                await message.answer("За выбранный период нет данных для экспорта.")
                return
            await message.answer_document(
                BufferedInputFile(data, filename=f"bot_events_{period}_{time.strftime('%Y%m%d')}.csv"),
                caption=f"События бота: {analytics.PERIODS[period]}",
            )
    except Exception:
        log.exception("Ошибка при формировании отчёта %s", action)
        await message.answer("Не удалось сформировать отчёт. Попробуйте позже.")


async def drop_markup(callback: CallbackQuery) -> None:
    if isinstance(callback.message, Message):
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass


@router.callback_query(F.data.startswith("fu:"))
async def process_followup(callback: CallbackQuery):
    user = callback.from_user
    if followups_left(user.id) <= 0:
        await callback.answer("Лимит уточнений исчерпан, начните новую тему.", show_alert=True)
        await drop_markup(callback)
        return
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    await drop_markup(callback)
    awaiting_followup[user.id] = time.monotonic()
    number = f"Уточнение {followup_chain.get(user.id, 0) + 1} из {MAX_FOLLOWUPS}."
    if callback.data == "fu:nf":
        text = f"Переформулируйте вопрос или добавьте детали (название раздела, экрана, документа), я поищу ещё раз. {number}"
    else:
        text = f"Напишите, что именно нужно уточнить, я продолжу с того места, где мы остановились. {number}"
    await callback.message.answer(text)


@router.callback_query(F.data.startswith("nt:"))
async def process_new_topic(callback: CallbackQuery):
    try:
        exchange_id = int(callback.data.split(":")[1])
    except (ValueError, IndexError):
        await callback.answer()
        return
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    await drop_markup(callback)
    user = callback.from_user
    reset_followups(user.id)
    await reset_server_dialog(user.id)
    count("new_topics", user.id)
    exchange = exchanges.get(exchange_id)
    if exchange and exchange[0] == user.id and exchange_id not in rated:
        await callback.message.answer(RATE_PROMPT_TEXT, reply_markup=rating_keyboard(exchange_id))
    else:
        await callback.message.answer("Хорошо, переходим к новой теме. Задайте ваш вопрос.")


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
        await drop_markup(callback)
        return
    user = callback.from_user
    skipped = value == "skip"
    if not skipped:
        rated.add(exchange_id)
    await callback.answer("Спасибо за оценку!" if not skipped else None)
    if isinstance(callback.message, Message):
        try:
            if callback.message.text == RATE_PROMPT_TEXT:
                done = "Спасибо за оценку!" if not skipped else "Хорошо, без оценки."
                await callback.message.edit_text(f"{done} Задайте ваш вопрос по новой теме.", reply_markup=None)
            else:
                await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass
    if skipped:
        return
    positive = value == "1"
    question, answer = "", ""
    exchange = exchanges.get(exchange_id)
    if exchange and exchange[0] == user.id:
        question, answer = exchange[1], exchange[2]
    count("feedback_positive" if positive else "feedback_negative", user.id, question=question)
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
        if parse_user_row(employee)[1]:
            db_admins.add(user.id)
            await ensure_admin_menu(user.id)
        count("registrations", user.id)
        full_name = employee.get("full_name")
        greeting = f"Отлично, {full_name}, вы успешно авторизованы!" if full_name else "Отлично, вы успешно авторизованы!"
        await message.answer(
            f"{greeting} Чем я могу помочь?",
            reply_markup=main_keyboard(is_admin(user.id)),
        )
        return
    pending_access[user.id] = employee_id
    if access_sent.get(user.id) == employee_id:
        await message.answer(
            "Ваша заявка уже на рассмотрении. Как только её одобрят, я напишу вам.",
            reply_markup=main_keyboard(),
        )
        return
    if await submit_access_request(user, employee_id):
        await message.answer(
            "Я не нашёл вас в базе сотрудников. Заявка на доступ отправлена администратору, "
            "как только её одобрят, я напишу вам. Если ввели данные с ошибкой, отправьте /start и введите их заново.",
            reply_markup=main_keyboard(),
        )
        return
    await message.answer(
        "Я не нашёл вас в базе сотрудников, а отправить заявку администратору не получилось. "
        "Нажмите кнопку, чтобы повторить попытку.",
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
    progress = await message.answer(f"⏳ Загружаю «{name}» на сервер…")
    try:
        buf = io.BytesIO()
        await bot.download(doc, destination=buf)
        await upload_document(name, buf.getvalue(), doc.mime_type or "application/octet-stream", message.from_user.id)
    except Exception as e:
        log.exception("Не удалось загрузить документ %s", name)
        # Состояние не сбрасываем: админ может сразу отправить файл ещё раз или нажать /cancel
        await progress.edit_text(f"❌ Не удалось добавить документ «{name}»: {e}\nОтправьте файл ещё раз или /cancel.")
        return
    await state.clear()
    await progress.edit_text(f"✅ Документ «{name}» успешно принят. Индексация идёт в фоне, обычно это занимает несколько минут.")


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
        for admin_id in ADMIN_IDS | db_admins:
            await ensure_admin_menu(admin_id)
        log.info(
            "Бот запущен, сервер: %s, прокси: %s, админов: %s",
            BASE_API_URL,
            "да" if PROXY_URL else "нет",
            len(ADMIN_IDS | db_admins),
        )
        await dp.start_polling(bot)
    finally:
        if sync_task:
            sync_task.cancel()
        await http.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())