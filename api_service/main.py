import asyncio
import logging
import os
from pathlib import Path

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_service import LLMError, ask_llm
from prompts import NO_ANSWER
import time

import features
from search import search_qdrant, warm

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
DOCS_DIR = Path("docs")
if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")

class TelegramMessage(BaseModel):
    user_id: int
    text: str


class Feedback(BaseModel):
    user_id: int
    rating: str
    question: str = ""
    answer: str = ""


async def _warm_up():
    try:
        await asyncio.to_thread(warm)
        log.info("Модель эмбеддингов загружена, Qdrant доступен")
    except Exception:
        log.exception("Прогрев не удался: проверьте Qdrant и загрузку модели (первый запрос будет медленным)")


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
    return {"history": "\n".join(reversed(items))}


def _image_urls(payload: dict, limit: int = 4) -> list[str]:
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


def _image_url(payload: dict) -> str | None:
    for path in [payload.get("image_path"), *(payload.get("images") or [])]:
        if path and (IMAGES_DIR / Path(path).name).is_file():
            return f"/images/{Path(path).name}"
    return None


@app.post("/webhook")
async def receive_message(msg: TelegramMessage):
    log.info("Сообщение от %s: %s", msg.user_id, msg.text)
    started = time.monotonic()

    try:
        history = await get_history(msg.user_id)

        search_query = features.contextualize(msg.text, history)
        standalone = search_query == msg.text.strip()

        cached = await features.cache_get(r, msg.text) if standalone else None

        if cached:
            await save_to_history(msg.user_id, f"Пользователь: {msg.text}")
            await save_to_history(msg.user_id, f"Бот: {cached['reply']}")
            await features.track(r, "cached", time.monotonic() - started, source=cached.get("source"))
            return {"status": "success", "cached": True, **{k: v for k, v in cached.items() if k != "source"}}

        context, payload = await search_qdrant(search_query, raise_errors=True)

        suggest_operator = False
        images: list[str] = []
        related: list[str] = []
        source = None

        if context is None:
            answer = NO_ANSWER
            suggest_operator = True
        else:
            answer = await ask_llm(context, history, msg.text)

            if answer == NO_ANSWER:
                suggest_operator = True
            else:
                sources_md = "\n".join(f"📎 {s}" for s in payload["sources"])
                answer = f"{answer}\n\n{sources_md}"
                images = _image_urls(payload)
                related = payload.get("related") or []
                source = payload.get("source")

        await save_to_history(msg.user_id, f"Пользователь: {msg.text}")
        await save_to_history(msg.user_id, f"Бот: {answer}")

        response = {
            "reply": answer,
            "parse_mode": "Markdown",
            "suggest_operator": suggest_operator,
            "image_path": images[0] if images else None,
            "images": images,
            "related": related,
        }

        if not suggest_operator and standalone:
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