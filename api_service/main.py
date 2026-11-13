from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import redis.asyncio as redis
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

app = FastAPI(title="Corporate Knowledge Assistant API")

r = redis.Redis(host='redis', port=6379, decode_responses=True)

qdrant = QdrantClient(url="http://qdrant:6333", trust_env=False)
COLLECTION_NAME = "corporate_docs"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TelegramMessage(BaseModel):
    user_id: int
    text: str


@app.on_event("startup")
async def startup_event():
    print("⏳ Инициализация Qdrant и загрузка документов...")

    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        print("🗑️ Старая коллекция удалена.")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=qdrant.get_embedding_size(MODEL_NAME),
            distance=Distance.COSINE
        ),
    )

    docs = [
        {
            "text": "Чтобы сбросить пароль от учетной записи, необходимо зайти в настройки профиля и нажать 'Сбросить пароль'.",
            "source": "Инструкция_по_безопасности.pdf", "page": 4},
        {"text": "Для получения доступа к VPN необходимо написать заявку в отдел кадров на почту hr@company.ru.",
         "source": "БП_Командировка.pdf", "page": 12},
        {"text": "Корпоративная столовая работает с 12:00 до 14:00. Оплата только по пропуску.",
         "source": "FAQ_сотрудника.pdf", "page": 1}
    ]

    qdrant.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=[models.Document(text=doc["text"], model=MODEL_NAME) for doc in docs],
        payload=[{"source": doc["source"], "page": doc["page"], "document": doc["text"]} for doc in docs],
        ids=list(range(len(docs)))
    )
    print("✅ База знаний успешно загружена!")


async def save_to_history(user_id: int, text: str):
    key = f"chat:{user_id}"
    await r.lpush(key, text)
    await r.ltrim(key, 0, 9)
    await r.expire(key, 86400)


async def get_history(user_id: int) -> list:
    key = f"chat:{user_id}"
    return await r.lrange(key, 0, -1)


async def search_qdrant(query: str):
    print(f"🔍 Ищем в Qdrant по запросу: {query}")

    try:
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=models.Document(text=query, model=MODEL_NAME),
            limit=1
        )

        points = search_result.points
        if points:
            score = points[0].score
            payload = points[0].payload

            if score < 0.65:
                print(f"⚠️ Низкая уверенность совпадения ({score:.2f}). Ответ не найден.")
                return None, None

            print(f"📄 Найдено в базе (Уверенность: {score:.2f}): {payload.get('document')}")
            return payload.get("document"), payload

    except Exception as e:
        print(f"⚠️ Ошибка поиска в Qdrant: {e}")

    return None, None


async def mock_ask_openai(context: str, question: str, history: list, payload: dict):
    await asyncio.sleep(0.5)

    source = payload.get("source", "Неизвестный файл")
    page = payload.get("page", "?")

    return f"Ответ: {context}\n\n📚 Источник: {source}, стр. {page}"


@app.post("/webhook")
async def receive_message(msg: TelegramMessage):
    print(f"\n📩 Пришло сообщение от {msg.user_id}: {msg.text}")

    try:
        history = await get_history(msg.user_id)
        await save_to_history(msg.user_id, f"Пользователь: {msg.text}")

        context, payload = await search_qdrant(msg.text)

        if context is None:
            answer = "Я не нашел точного ответа в корпоративной документации. Возможно, вам нужен оператор поддержки?"
        else:
            answer = await mock_ask_openai(context, msg.text, history, payload)

        await save_to_history(msg.user_id, f"Бот: {answer}")
        return {"status": "success", "reply": answer}

    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return {"status": "error", "reply": "Упс, сервис временно недоступен."}