import asyncio
import base64
import logging
import os
import time
from pathlib import Path

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_service import ask_llm, LLMError
from search import search_qdrant, warm

NO_ANSWER = "Я не нашел точного ответа в документации. Переключить на оператора?"

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

class TelegramMessage(BaseModel):
    user_id: int
    text: str

async def _warm_up():
    try:
        await asyncio.to_thread(warm)
        log.info("Модель эмбеддингов загружена, Qdrant доступен")
    except Exception:
        log.exception("Прогрев не удался: проверьте Qdrant и загрузку модели")

@app.on_event("startup")
async def startup_event():
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
    return {"history": "\n".join(reversed(items)) if items else "История пуста."}

def _image_url(payload: dict) -> str | None:
    for path in [payload.get("image_path"), *(payload.get("images") or [])]:
        if path and (IMAGES_DIR / Path(path).name).is_file():
            return f"/images/{Path(path).name}"
    return None

def encode_image_to_base64(image_path: str) -> str | None:
    try:
        full_path = IMAGES_DIR / Path(image_path).name
        if full_path.is_file():
            with open(full_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        log.error(f"Ошибка кодирования картинки: {e}")
    return None

@app.post("/webhook")
async def receive_message(msg: TelegramMessage):
    log.info("Пришло сообщение от %s: %s", msg.user_id, msg.text)
    total_start = time.perf_counter()

    try:
        history = await get_history(msg.user_id)

        search_start = time.perf_counter()
        context, payload = await search_qdrant(msg.text, raise_errors=True)
        search_time = time.perf_counter() - search_start

        suggest_operator = False
        image_url = None
        image_base64 = None

        if context is None:
            answer = NO_ANSWER
            suggest_operator = True
        else:
            raw_image_path = payload.get("image_path")
            if raw_image_path:
                image_base64 = encode_image_to_base64(raw_image_path)

            llm_start = time.perf_counter()
            answer = await ask_llm(context, history, msg.text, image_base64)
            llm_time = time.perf_counter() - llm_start

            if answer == NO_ANSWER:
                suggest_operator = True
            else:
                sources = payload.get("sources", [])
                if sources:
                    answer = f"{answer}\n\n" + "\n".join(sources)
                image_url = _image_url(payload)

            log.info(f"Поиск: {search_time:.2f} сек, Генерация: {llm_time:.2f} сек")

        await save_to_history(msg.user_id, f"Пользователь: {msg.text}")
        await save_to_history(msg.user_id, f"Бот: {answer}")

        total_time = time.perf_counter() - total_start
        log.info(f"Общее время ответа: {total_time:.2f} сек")

        return {
            "status": "success",
            "reply": answer,
            "suggest_operator": suggest_operator,
            "image_path": image_url,
        }

    except LLMError as e:
        log.error("Ошибка LLM: %s", e)
        return {"status": "error", "reply": SERVICE_ERROR}

    except Exception:
        log.exception("Ошибка обработки сообщения")
        return {"status": "error", "reply": SERVICE_ERROR}