"""Дополнительные возможности API, не требующие изменений в боте.

Всё здесь чистые функции или обёртки над Redis, которые никогда не бросают исключений:
если Redis недоступен, бот работает как раньше, просто без кэша и статистики.
"""

import hashlib
import json
import logging
import re
import time

log = logging.getLogger("rag.features")

CACHE_TTL = 3600
GAPS_LIMIT = 200
FEEDBACK_LIMIT = 1000

_FOLLOWUP_START = re.compile(
    r"^(а|и|ну|тогда|то есть|а если|а как|а где|а можно|а что|а почему|а зачем|а какой|а какая|а какие|"
    r"подробнее|поподробнее|ещё|еще|дальше|что дальше|что потом|что значит|как это|почему|зачем|пример|"
    r"например|покажи|объясни|уточни|повтори|расскажи подробнее|а потом|а после|после этого)\b",
    re.IGNORECASE,
)
_PRONOUNS = re.compile(
    r"\b(его|её|ее|их|им|ему|ей|это|этого|этом|этому|этот|эта|эти|этих|этой|там|тут|такой|такая|такое|"
    r"такие|оно|они|он|она|тот|та|те|то|данный|данная|данное|данные|предыдущ\w+|прошл\w+|выше|"
    r"тоже|также|вышеуказанн\w+|первый|второй|третий|последний|следующий|шаг|пункт)\b",
    re.IGNORECASE,
)
# Самодостаточные вопросы не склеиваем с прошлым, даже если в них есть «это»
_SELF_CONTAINED_START = re.compile(r"^(как|что|где|когда|почему|зачем|какой|какая|какие|можно ли|есть ли)\b.{25,}", re.IGNORECASE)
_REFUSAL_MARK = "не нашел точного ответа"
FOLLOWUP_MAX_WORDS = 8


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (question or "").lower())).strip()


def _turns(history: list) -> list[tuple[str, str | None]]:
    """Пары (вопрос, ответ) от старых к новым. История в Redis хранится от новых к старым."""
    items = list(reversed(history or []))
    turns: list[tuple[str, str | None]] = []

    for item in items:
        if item.startswith("Пользователь:"):
            turns.append((item.split(":", 1)[1].strip(), None))
        elif item.startswith("Бот:") and turns and turns[-1][1] is None:
            turns[-1] = (turns[-1][0], item.split(":", 1)[1].strip())

    return turns


def last_user_question(history: list) -> str | None:
    """Последний вопрос пользователя, на который бот реально ответил (отказы пропускаем)."""
    for question, answer in reversed(_turns(history)):
        if answer is not None and _REFUSAL_MARK in answer.lower():
            continue
        if question:
            return question
    return None


def is_followup(question: str, history: list) -> bool:
    """Уточняющий ли это вопрос: короткий и опирается на прошлую реплику."""
    q = (question or "").strip()
    words = q.split()

    if not words or not last_user_question(history):
        return False

    if len(words) <= 2:
        return True

    if len(words) > FOLLOWUP_MAX_WORDS:
        return False

    if _SELF_CONTAINED_START.match(q) and not _PRONOUNS.search(q):
        return False

    return bool(_FOLLOWUP_START.match(q)) or bool(_PRONOUNS.search(q))


def contextualize(question: str, history: list, topic: str | None = None) -> str:
    """Запрос для поиска: уточнение дополняем темой диалога (или прошлым вопросом).

    topic — самостоятельная формулировка последнего успешного вопроса. Благодаря ей цепочка
    уточнений не размывается: на пятом «а подробнее?» тема всё ещё исходная.
    """
    q = (question or "").strip()
    previous = topic or last_user_question(history)

    if previous and is_followup(q, history):
        # Если прошлый вопрос уже содержит тему уточнения, повторно не добавляем
        if normalize(previous) in normalize(q):
            return q
        return f"{previous}. {q}"

    return q


def cache_key(question: str) -> str:
    return "ans:" + hashlib.md5(normalize(question).encode("utf-8")).hexdigest()


def related_titles(hits: list[dict], best: dict, threshold: float, limit: int = 3) -> list[str]:
    seen = {best.get("section")}
    result = []

    for h in hits:
        section = h.get("section")

        if section in seen or h["score"] < threshold - 0.04:
            continue

        seen.add(section)
        title = str(section).split(">")[-1].strip()

        if title and not title.lower().startswith("стр."):
            result.append(title)

        if len(result) >= limit:
            break

    return result


async def cache_get(r, question: str) -> dict | None:
    try:
        raw = await r.get(cache_key(question))
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("кэш недоступен: %s", e)
        return None


async def cache_set(r, question: str, value: dict) -> None:
    try:
        await r.set(cache_key(question), json.dumps(value, ensure_ascii=False), ex=CACHE_TTL)
    except Exception as e:
        log.warning("кэш недоступен: %s", e)


LAST_CTX_TTL = 1800
LAST_CTX_CHARS = 3000


async def ctx_save(r, user_id: int, context: str, payload: dict | None) -> None:
    """Запоминаем найденные фрагменты, чтобы уточнение («а подробнее?») могло на них опереться."""
    try:
        value = {"context": (context or "")[:LAST_CTX_CHARS], "payload": payload or {}}
        await r.set(f"lastctx:{user_id}", json.dumps(value, ensure_ascii=False, default=str), ex=LAST_CTX_TTL)
    except Exception as e:
        log.warning("контекст не сохранён: %s", e)


async def ctx_get(r, user_id: int) -> dict | None:
    try:
        raw = await r.get(f"lastctx:{user_id}")
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("контекст недоступен: %s", e)
        return None


async def topic_save(r, user_id: int, topic: str) -> None:
    try:
        await r.set(f"topic:{user_id}", topic[:300], ex=LAST_CTX_TTL)
    except Exception as e:
        log.warning("тема не сохранена: %s", e)


async def topic_get(r, user_id: int) -> str | None:
    try:
        raw = await r.get(f"topic:{user_id}")
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except Exception as e:
        log.warning("тема недоступна: %s", e)
        return None


async def track(r, outcome: str, seconds: float, question: str | None = None, source: str | None = None) -> None:
    try:
        await r.hincrby("stats", "questions", 1)
        await r.hincrby("stats", outcome, 1)
        await r.hincrbyfloat("stats", "latency_sum", round(seconds, 3))

        if outcome == "not_found" and question:
            await r.lpush("gaps", json.dumps({"q": question, "t": int(time.time())}, ensure_ascii=False))
            await r.ltrim("gaps", 0, GAPS_LIMIT - 1)

        if source:
            await r.hincrby("sources", source, 1)
    except Exception as e:
        log.warning("статистика недоступна: %s", e)


async def save_feedback(r, item: dict) -> bool:
    try:
        rating = "up" if item.get("rating") in ("up", "👍", 1, "1") else "down"
        await r.hincrby("stats", f"feedback_{rating}", 1)
        item = {**item, "rating": rating, "t": int(time.time())}
        await r.lpush("feedback", json.dumps(item, ensure_ascii=False))
        await r.ltrim("feedback", 0, FEEDBACK_LIMIT - 1)
        return True
    except Exception as e:
        log.warning("отзыв не сохранён: %s", e)
        return False


async def analytics(r, top: int = 10) -> dict:
    try:
        stats = await r.hgetall("stats")
        sources = await r.hgetall("sources")
        gaps_raw = await r.lrange("gaps", 0, 99)
    except Exception as e:
        log.warning("аналитика недоступна: %s", e)
        return {"available": False}

    n = lambda k: int(float(stats.get(k, 0)))
    total = n("questions")

    gap_count: dict[str, int] = {}
    for raw in gaps_raw:
        try:
            q = json.loads(raw)["q"]
        except Exception:
            continue
        key = normalize(q)
        gap_count[key] = gap_count.get(key, 0) + 1

    up, down = n("feedback_up"), n("feedback_down")

    return {
        "available": True,
        "questions": total,
        "found": n("found") + n("cached"),
        "cached": n("cached"),
        "not_found": n("not_found"),
        "errors": n("error"),
        "answer_rate": round((n("found") + n("cached")) / total, 3) if total else None,
        "avg_seconds": round(float(stats.get("latency_sum", 0)) / total, 2) if total else None,
        "feedback": {"up": up, "down": down,
                     "satisfaction": round(up / (up + down), 3) if up + down else None},
        "top_sources": sorted(((k, int(v)) for k, v in sources.items()), key=lambda x: -x[1])[:top],
        "knowledge_gaps": sorted(gap_count.items(), key=lambda x: -x[1])[:top],
    }