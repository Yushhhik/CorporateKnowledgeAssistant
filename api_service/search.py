"""Поиск по базе знаний."""

import asyncio
import logging
import re
import sys
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue

from qdrant_store import (
    COLLECTION_NAME,
    embed_query,
    get_client,
    get_neighbors,
    get_page_images,
    warm_up,
)

from features import related_titles

log = logging.getLogger("rag.search")

TOP_K = 5
# Порог по косинусной близости. Подставьте значение, которое подсказал test_search.py
CONFIDENCE_THRESHOLD = 0.8359
CONTEXT_CHUNKS = 3

CANDIDATES = 15          # сколько кандидатов берём из Qdrant перед пересортировкой
LEXICAL_WEIGHT = 0.05    # максимальная добавка за совпадение слов; score для порога остаётся чистым
MAX_CONTEXT_CHARS = 3500

_client = None


def _get_client():
    """Переиспользуем подключение к Qdrant."""
    global _client

    if _client is None:
        _client = get_client()

    return _client


def warm() -> None:
    """Вызвать один раз при старте бота: грузит модель и проверяет Qdrant."""
    warm_up()
    _get_client().get_collections()


def detect_system(query: str) -> str | None:
    """Определяет систему, если пользователь явно назвал её."""

    query = (query or "").lower()

    if "sap" in query:
        return "sap"

    if re.search(r"(?<![а-яa-z0-9])(1с|1c)(?![а-яa-z0-9])", query):
        return "1c"

    if "docsvision" in query or "доксвиж" in query or "докс вижн" in query:
        return "docsvision"

    if (
        "garantexpress" in query
        or "гарант экспресс" in query
        or "гарантэкспресс" in query
        or "экспресс согласован" in query
    ):
        return "garantexpress"

    if "барс" in query or "bars" in query:
        return "bars"

    return None


_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _stem(word: str) -> str:
    """Грубое усечение окончаний: «командировку» и «командировка» совпадут."""
    return word[:5] if len(word) > 5 else word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD_RE.findall(text.lower()) if len(w) > 3}


def _lexical_bonus(query_words: set[str], text: str) -> float:
    if not query_words:
        return 0.0

    return LEXICAL_WEIGHT * len(query_words & _words(text)) / len(query_words)


def search(
    query: str,
    top_k: int = TOP_K,
    chunk_type: str | None = None,
    system: str | None = None
) -> list[dict]:
    """Ищет ближайшие чанки с возможной фильтрацией по системе.

    Порядок результатов уточняется совпадением слов, но поле score — это чистая
    косинусная близость: именно по ней калибруется порог.
    """

    query = (query or "").strip()

    if not query:
        return []

    # Если система не передана явно, определяем её по вопросу
    if system is None:
        system = detect_system(query)

    conditions = []

    if chunk_type:
        conditions.append(
            FieldCondition(key="type", match=MatchValue(value=chunk_type))
        )

    if system:
        conditions.append(
            FieldCondition(key="system", match=MatchValue(value=system))
        )

    flt = Filter(must=conditions) if conditions else None

    response = _get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=embed_query(query),
        query_filter=flt,
        limit=max(top_k, CANDIDATES),
        with_payload=True,
    )

    qwords = _words(query)

    ranked = sorted(
        response.points,
        key=lambda p: p.score + _lexical_bonus(
            qwords,
            f"{p.payload.get('section') or ''} {p.payload['text']}"
        ),
        reverse=True,
    )[:top_k]

    return [
        {
            "type": p.payload.get("type", "text"),
            "source": p.payload["source"],
            "section": p.payload.get("section"),
            "page": p.payload.get("page"),
            "text": p.payload["text"],
            "image_path": p.payload.get("image_path"),
            "system": p.payload.get("system", "unknown"),
            "chunk_index": p.payload.get("chunk_index"),
            "score": round(p.score, 4),
        }
        for p in ranked
    ]


async def search_async(
    query: str,
    top_k: int = TOP_K,
    chunk_type: str | None = None,
    system: str | None = None
) -> list[dict]:
    """Асинхронная версия поиска."""

    return await asyncio.to_thread(search, query, top_k, chunk_type, system)


def is_confident(
    hits: list[dict],
    threshold: float | None = None
) -> bool:
    """Проверяет score лучшего результата."""

    if threshold is None:
        threshold = CONFIDENCE_THRESHOLD

    return bool(hits) and max(h["score"] for h in hits) >= threshold


def format_source(hit: dict) -> str:
    name = Path(hit["source"]).stem
    filename = hit["source"]
    if hit.get("page"):
        return f"[{name}, стр. {hit['page']}](/docs/{filename}#page={hit['page']})"
    return f"[{name}](/docs/{filename})"


def _expand(best: dict) -> str:
    """Текст лучшего чанка + соседние текстовые чанки: ответ не обрывается на середине."""

    parts = [best["text"]]

    if best.get("chunk_index") is None or best["type"] != "text":
        return best["text"]

    try:
        neighbors = get_neighbors(
            _get_client(), best["source"], best["chunk_index"]
        )
    except Exception:
        log.exception("neighbors failed")
        return best["text"]

    for n in neighbors:
        if n.get("type") != "text" or n.get("section") != best.get("section"):
            continue

        if n.get("chunk_index", 0) < best["chunk_index"]:
            parts.insert(0, n["text"])
        else:
            parts.append(n["text"])

    return "\n".join(parts)


def _collect_images(best: dict, good: list[dict]) -> list[str]:
    """Пути картинок: из найденных чанков и со страницы лучшего чанка."""

    paths = [h["image_path"] for h in good if h.get("image_path")]

    if best.get("page") is not None:
        try:
            for p in get_page_images(_get_client(), best["source"], best["page"]):
                if p.get("image_path"):
                    paths.append(p["image_path"])
        except Exception:
            log.exception("page images failed")

    return list(dict.fromkeys(paths))[:5]


def _search_qdrant_sync(query: str) -> tuple[str | None, dict | None]:
    hits = search(query, top_k=TOP_K)

    if not is_confident(hits):
        return None, None

    good = [
        h for h in hits
        if h["score"] >= CONFIDENCE_THRESHOLD
    ][:CONTEXT_CHUNKS]

    if not good:
        return None, None

    best = good[0]

    blocks = []

    for i, h in enumerate(good):
        text = _expand(h) if i == 0 else h["text"]
        blocks.append(f"[{format_source(h)}]\n{text}")

    context = "\n\n".join(blocks)[:MAX_CONTEXT_CHARS]

    payload = {
        "source": best["source"],
        "page": (
            best["page"]
            if best["page"] is not None
            else best["section"]
        ),
        "score": best["score"],
        "section": best["section"],
        "citation": format_source(best),
        "image_path": best["image_path"],
        "images": _collect_images(best, good),
        "related": related_titles(hits, best, CONFIDENCE_THRESHOLD),
        "system": best["system"],
        "sources": list(dict.fromkeys(format_source(h) for h in good)),
    }

    return context, payload


async def search_qdrant(
    query: str,
    raise_errors: bool = False
) -> tuple[str | None, dict | None]:
    """Интерфейс для интегратора: context, payload = await search_qdrant(msg.text).

    По умолчанию не бросает исключений: при сбое Qdrant/модели возвращает (None, None),
    и бот отвечает как на вопрос без ответа в базе. API передаёт raise_errors=True,
    чтобы недоступность Qdrant не выдавалась за «ответа нет в документации».
    """

    try:
        return await asyncio.to_thread(_search_qdrant_sync, query)
    except Exception:
        log.exception("search_qdrant failed")

        if raise_errors:
            raise

        return None, None


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Как сбросить пароль?"

    results = search(q)

    print(f"Вопрос: {q}")
    print(f"Определённая система: {detect_system(q)}")
    print(f"Уверенность: {is_confident(results)}\n")

    for i, h in enumerate(results, 1):
        print(
            f"{i}. [{h['score']}] "
            f"({h['type']}, {h['system']}) "
            f"{format_source(h)}"
        )
        print(f"   {h['text'][:200]}\n")
