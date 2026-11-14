from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import redis.asyncio as redis
import os
import openai
from search import search_async, is_confident

app = FastAPI(title="Corporate Knowledge Assistant API")

r = redis.Redis(host='redis', port=6379, decode_responses=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
ollama_client = openai.OpenAI(
    base_url=f"{OLLAMA_BASE_URL}/v1",
    api_key="ollama"
)


class TelegramMessage(BaseModel):
    user_id: int
    text: str


@app.on_event("startup")
async def startup_event():
    print("Сервер запущен и готов к поиску!")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


async def save_to_history(user_id: int, text: str):
    key = f"chat:{user_id}"
    await r.lpush(key, text)
    await r.ltrim(key, 0, 9)
    await r.expire(key, 86400)


async def get_history(user_id: int) -> list:
    key = f"chat:{user_id}"
    return await r.lrange(key, 0, -1)


async def ask_llm(context: str, question: str, history: list, payload: dict):
    from prompts import build_prompt
    messages = build_prompt(context, history, question)

    try:
        response = await asyncio.to_thread(
            ollama_client.chat.completions.create,
            model="llama3.2",
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка Ollama: {e}")
        return "Извините, сервис временно недоступен. Попробуйте позже."


@app.post("/webhook")
async def receive_message(msg: TelegramMessage):
    print(f"\nПришло сообщение от {msg.user_id}: {msg.text}")

    try:
        history = await get_history(msg.user_id)
        await save_to_history(msg.user_id, f"Пользователь: {msg.text}")

        hits = await search_async(msg.text)

        suggest_operator = False

        if not is_confident(hits):
            answer = "Я не нашел точного ответа в корпоративной документации. Возможно, вам нужен оператор поддержки?"
            suggest_operator = True
        else:
            best_hit = hits[0]
            context = best_hit["text"]
            payload = {"source": best_hit["source"], "page": best_hit["page"]}

            answer = await ask_llm(context, msg.text, history, payload)

            source = payload.get("source", "Неизвестный файл")
            page = payload.get("page", "?")
            answer = f"{answer}\n\nИсточник: {source}, стр. {page}"

        await save_to_history(msg.user_id, f"Бот: {answer}")

        return {"status": "success", "reply": answer, "suggest_operator": suggest_operator}

    except Exception as e:
        print(f"ОШИБКА: {e}")
        return {"status": "error", "reply": "Упс, сервис временно недоступен."}