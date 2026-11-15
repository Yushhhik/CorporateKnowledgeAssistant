import asyncio
import logging
import os
import shutil
from pathlib import Path

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from llm_service import LLMError, ask_llm
from prompts import NO_ANSWER, furniture_note
import time

import features
from search import search_qdrant, warm
from database import get_db, init_db, User
from ingest import main as run_ingest

log = logging.getLogger("rag.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Corporate Knowledge Assistant API")

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
    socket_connect_timeout=3,
    socket_timeout=3,
)

SERVICE_ERROR = "Извините, сервис временно недоступен. Попробуйте позже или свяжитесь с оператором."

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

DOCS_DIR = Path("/app/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)


class TelegramMessage(BaseModel):
    user_id: int
    text: str = ""
    image_base64: str | None = None


class Feedback(BaseModel):
    user_id: int
    rating: str
    question: str = ""
    answer: str = ""


class AuthRequest(BaseModel):
    employee_id: str
    telegram_id: int
    username: str | None = None


class UserCreate(BaseModel):
    employee_id: str
    telegram_id: int
    admin_id: int
    full_name: str = "Сотрудник"


async def _warm_up():
    try:
        await asyncio.to_thread(warm)
        log.info("Модель эмбеддингов загружена, Qdrant доступен")
    except Exception:
        log.exception("Прогрев не удался: проверьте Qdrant и загрузку модели")


@app.on_event("startup")
async def startup_event():
    await init_db()
    app.state.warm_task = asyncio.create_task(_warm_up())
    log.info("Сервер запущен")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


async def save_to_history(user_id: int, text: str):
    key = f"chat:{user_id}"
    try:
        await r.lpush(key, text)
        await r.ltrim(key, 0, 9)
        await r.expire(key, 86400)
    except Exception as e:
        log.warning("Redis недоступен, история не сохранена: %s", e)


async def get_history(user_id: int) -> list:
    try:
        return await r.lrange(f"chat:{user_id}", 0, -1)
    except Exception as e:
        log.warning("Redis недоступен, работаем без истории: %s", e)
        return []


@app.get("/history/{user_id}")
async def history_endpoint(user_id: int):
    items = await get_history(user_id)
    return {"history": "\n".join(reversed(items))}


def _image_urls(payload: dict, limit: int = 2) -> list[str]:
    urls = []
    for path in [payload.get("image_path"), *(payload.get("images") or [])]:
        if path and (IMAGES_DIR / Path(path).name).is_file():
            url = f"/images/{Path(path).name}"
            if url not in urls:
                urls.append(url)
    return urls[:limit]


@app.post("/feedback")
async def feedback_endpoint(fb: Feedback):
    ok = await features.save_feedback(r, fb.model_dump() if hasattr(fb, "model_dump") else dict(fb.__dict__))
    return {"status": "success" if ok else "error"}


@app.get("/analytics")
async def analytics_endpoint():
    return await features.analytics(r)


@app.post("/auth")
async def auth_endpoint(auth: AuthRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.employee_id == auth.employee_id))
    user = result.scalar_one_or_none()

    if not user:
        return JSONResponse(status_code=404, content={"status": "not_found"})

    if user.telegram_id is not None and user.telegram_id != auth.telegram_id:
        return JSONResponse(status_code=400, content={"status": "error", "detail": "Аккаунт уже привязан"})

    user.telegram_id = auth.telegram_id
    user.is_verified = True
    await db.commit()
    return {"status": "success", "full_name": user.full_name}


@app.post("/users")
async def create_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.employee_id == user_data.employee_id))
    user = result.scalar_one_or_none()

    if not user:
        user = User(employee_id=user_data.employee_id, full_name=user_data.full_name)
        db.add(user)
    else:
        user.full_name = user_data.full_name

    user.telegram_id = user_data.telegram_id
    user.is_verified = True
    user.approved_by = user_data.admin_id

    await db.commit()
    return {"status": "success"}


@app.get("/users")
async def get_all_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(User.is_verified == True, User.telegram_id.isnot(None))
    )
    users = result.scalars().all()

    return {
        "users": [
            {"telegram_id": u.telegram_id, "employee_id": u.employee_id}
            for u in users
        ]
    }


@app.get("/documents")
async def get_documents():
    files = [f.name for f in DOCS_DIR.glob("*") if f.suffix.lower() in (".pdf", ".docx")]
    return {"documents": files}


@app.post("/documents")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = DOCS_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_ingest, str(DOCS_DIR), full=False)

    return {"status": "success", "filename": file.filename}


def _image_url(payload: dict) -> str | None:
    for path in [payload.get("image_path"), *(payload.get("images") or [])]:
        if path and (IMAGES_DIR / Path(path).name).is_file():
            return f"/images/{Path(path).name}"
    return None


@app.post("/webhook")
async def receive_message(msg: TelegramMessage, db: AsyncSession = Depends(get_db)):
    if not msg.text and not msg.image_base64:
        return {"status": "error", "reply": "Сообщение пустое"}

    # Фото без подписи: искать в базе знаний нечего (служебная фраза в документах не встречается),
    # поэтому поиск пропускаем и сразу отдаём картинку llava
    image_only = bool(msg.image_base64) and not (msg.text or "").strip()

    if image_only:
        msg.text = "Опиши это изображение"

    log.info("Сообщение от %s: %s", msg.user_id, msg.text)
    started = time.monotonic()

    try:
        result = await db.execute(select(User).where(User.telegram_id == msg.user_id, User.is_verified == True))
        user = result.scalar_one_or_none()

        if not user:
            return {"status": "unauthorized",
                    "reply": "Пожалуйста, авторизуйтесь. Введите ваш рабочий Email или табельный номер."}

        history = await get_history(msg.user_id)

        search_query = features.contextualize(msg.text, history)
        standalone = search_query == msg.text.strip()

        cached = await features.cache_get(r, msg.text) if standalone and not msg.image_base64 else None

        if cached:
            await save_to_history(msg.user_id, f"Пользователь: {msg.text}")
            await save_to_history(msg.user_id, f"Бот: {cached['reply']}")
            await features.track(r, "cached", time.monotonic() - started, source=cached.get("source"))
            return {"status": "success", "cached": True, **{k: v for k, v in cached.items() if k != "source"}}

        if image_only:
            context, payload = None, None
        else:
            context, payload = await search_qdrant(search_query, raise_errors=True)

        suggest_operator = False
        images: list[str] = []
        related: list[str] = []
        source = None

        if context is None and not msg.image_base64:
            answer = NO_ANSWER
            suggest_operator = True
        else:
            answer = await ask_llm(context if context else "", history, msg.text, image_base64=msg.image_base64)

            if answer == NO_ANSWER:
                suggest_operator = True
            else:
                note = None if msg.image_base64 else furniture_note(msg.text, context or "")
                if note and "реестра техники нет" not in answer:
                    answer = f"{note} Возможно, вы имели в виду реестр мебели?\n\n{answer}"

                if payload and payload.get("sources"):
                    answer = f"{answer}\n\n" + "\n".join(payload["sources"])
                if payload:
                    # на фото пользователя картинки из документации не прикладываем
                    images = [] if msg.image_base64 else _image_urls(payload)
                    related = payload.get("related") or []
                    source = payload.get("source")

        # обмен про фото в историю не пишем: он мешает следующим текстовым вопросам
        if not msg.image_base64:
            await save_to_history(msg.user_id, f"Пользователь: {msg.text}")
            await save_to_history(msg.user_id, f"Бот: {answer}")

        response = {
            "reply": answer,
            "suggest_operator": suggest_operator,
            "image_path": images[0] if images else None,
            "images": images,
            "related": related,
        }

        if not suggest_operator and standalone and not msg.image_base64:
            await features.cache_set(r, msg.text, {**response, "source": source})

        await features.track(
            r,
            "not_found" if suggest_operator else "found",
            time.monotonic() - started,
            question=msg.text,
            source=source,
        )

        return {"status": "success", "cached": False, **response}

    except LLMError as e:
        log.error("Ошибка LLM (Ollama): %s", e)
        await features.track(r, "error", time.monotonic() - started)
        return {"status": "error", "reply": SERVICE_ERROR}

    except Exception:
        log.exception("Ошибка обработки сообщения")
        await features.track(r, "error", time.monotonic() - started)
        return {"status": "error", "reply": SERVICE_ERROR}