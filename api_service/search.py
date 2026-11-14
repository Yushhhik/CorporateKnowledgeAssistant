"""Поиск по базе знаний. Эту функцию вызывает интегратор в основном пайплайне.

Пример:
    from search import search, search_async, is_confident, format_source
    hits = await search_async("Как сбросить пароль?")
    if is_confident(hits):
        ...   # собираем промпт из hits, источник берём через format_source(hit)

Если hit["type"] == "image", у него есть hit["image_path"]: файл можно отправить
пользователю в Telegram (send_photo), это хорошо для вопросов "как выглядит кнопка".

Ручная проверка:
    python search.py "Мне не отвечает офис-менеджер"
"""
import asyncio
import sys
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue

from qdrant_store import COLLECTION_NAME, embed_query, get_client

TOP_K = 5
# Порог уверенности: если лучший результат ниже, бот должен сказать
# "Я не нашел точного ответа. Переключить на оператора?".
# Подбирается на тест-кейсах (python test_search.py подскажет значение).
CONFIDENCE_THRESHOLD = 0.80

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def search(query: str, top_k: int = TOP_K, chunk_type: str | None = None) -> list[dict]:
    """Топ-k ближайших чанков, лучшие первыми.

    chunk_type: None (и текст, и картинки), "text" или "image".
    Каждый результат: {type, source, section, page, text, image_path, score}.
    """
    query = (query or "").strip()
    if not query:
        return []

    flt = None
    if chunk_type:
        flt = Filter(must=[FieldCondition(key="type", match=MatchValue(value=chunk_type))])

    response = _get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=embed_query(query),
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "type": p.payload.get("type", "text"),
            "source": p.payload["source"],
            "section": p.payload.get("section"),
            "page": p.payload.get("page"),
            "text": p.payload["text"],
            "image_path": p.payload.get("image_path"),
            "score": round(p.score, 4),
        }
        for p in response.points
    ]


async def search_async(query: str, top_k: int = TOP_K, chunk_type: str | None = None) -> list[dict]:
    """Та же функция, но не блокирует event loop (эмбеддинг считается в отдельном потоке)."""
    return await asyncio.to_thread(search, query, top_k, chunk_type)


def is_confident(hits: list[dict], threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """True, если лучший результат достаточно близок к вопросу, чтобы отвечать по нему."""
    return bool(hits) and hits[0]["score"] >= threshold


def format_source(hit: dict) -> str:
    """Строка для ответа бота: 'Источник: БП Командировка, раздел «5 Инструкция»' или '..., стр. 9'."""
    name = Path(hit["source"]).stem
    if hit.get("page"):
        return f"Источник: {name}, стр. {hit['page']}"
    return f"Источник: {name}, раздел «{hit['section']}»"


CONTEXT_CHUNKS = 3  # сколько лучших чанков склеиваем в контекст для LLM


async def search_qdrant(query: str) -> tuple[str | None, dict | None]:
    """Интерфейс для интегратора (контракт с Core).

    Использование: context, payload = await search_qdrant(msg.text)

    Возвращает (контекст, метаданные) или (None, None), если ответа нет / уверенность низкая.
      контекст: текст лучших фрагментов (те из топ-3, что выше порога), для промпта LLM;
      метаданные: {"source": имя файла, "page": номер страницы (у docx вместо номера
                   название раздела), "score": ..., "section", "citation",
                   "image_path", "sources"}.
    """
    hits = await search_async(query, top_k=TOP_K)
    if not is_confident(hits):
        return None, None

    good = [h for h in hits[:CONTEXT_CHUNKS] if h["score"] >= CONFIDENCE_THRESHOLD]
    context = "\n\n".join(f"[{format_source(h)}]\n{h['text']}" for h in good)

    best = hits[0]
    payload = {
        "source": best["source"],
        # у docx нет страниц, поэтому вместо номера отдаём раздел: ключ page есть всегда
        "page": best["page"] if best["page"] is not None else best["section"],
        "score": best["score"],
        "section": best["section"],
        "citation": format_source(best),    # готовая строка "Источник: ..."
        "image_path": best["image_path"],   # путь к картинке, если лучший результат это скриншот
        "sources": list(dict.fromkeys(format_source(h) for h in good)),
    }
    return context, payload


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Как сбросить пароль?"
    results = search(q)
    print(f"Вопрос: {q}")
    print(f"Уверенность: {is_confident(results)}\n")
    for i, h in enumerate(results, 1):
        print(f"{i}. [{h['score']}] ({h['type']}) {format_source(h)}")
        print(f"   {h['text'][:200]}\n")
