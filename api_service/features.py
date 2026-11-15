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

_FOLLOWUP_START = re.compile(r"^(а|и|ну|тогда|то есть|а если|а как|а где|а можно)\b", re.IGNORECASE)
_PRONOUNS = re.compile(r"\b(его|её|ее|их|это|этого|этом|этот|эта|там|тут|такой|такая|оно|они|он|она)\b", re.IGNORECASE)


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (question or "").lower())).strip()


def last_user_question(history: list) -> str | None:
    for item in history or []:
        if item.startswith("Пользователь:"):
            return item.split(":", 1)[1].strip()
    return None


def contextualize(question: str, history: list) -> str:
    q = (question or "").strip()
    words = q.split()
    previous = last_user_question(history)

    if not previous or not words:
        return q

    short = len(words) <= 4
    followup = bool(_FOLLOWUP_START.match(q)) or bool(_PRONOUNS.search(q))

    if (short and followup) or (len(words) <= 2):
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