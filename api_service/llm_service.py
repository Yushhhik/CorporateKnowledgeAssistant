"""Генерация ответа через Ollama (OpenAI-совместимый API)."""

import asyncio
import hashlib
import logging
import os
import re
import time

import openai

from prompts import NO_ANSWER, build_prompt

log = logging.getLogger("rag.llm")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")
# Картинки всегда разбирает llava, текстовые вопросы — OLLAMA_MODEL
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))

_client: openai.OpenAI | None = None

# Источники добавляет код (из результатов поиска), а не модель: небольшие модели выдумывают страницы.
_SOURCE_LINE_RE = re.compile(r"(?im)^\s*источник(?:и)?\s*[:—-].*$")
_REFUSAL_RE = re.compile(r"не нашел точного ответа|не нашла точного ответа|__NO_ANSWER__", re.IGNORECASE)
# Длинный ответ, в котором лишь упомянута оговорка, отказом не считается
_REFUSAL_MAX_LEN = 250

_cache: dict[str, str] = {}
_CACHE_MAX_SIZE = 1000

_stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "successful": 0,
    "errors": 0,
    "total_time": 0.0,
    "max_time": 0.0,
    "min_time": float("inf"),
}


class LLMError(Exception):
    """Модель недоступна или вернула ошибку."""


def _get_client() -> openai.OpenAI:
    global _client

    if _client is None:
        _client = openai.OpenAI(
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key="ollama",
            timeout=LLM_TIMEOUT,
            max_retries=0,
        )

    return _client


def _make_cache_key(context: str, question: str) -> str:
    raw = f"{context.strip().lower()}|{question.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cleanup_cache() -> None:
    if len(_cache) > _CACHE_MAX_SIZE:
        for key in list(_cache.keys())[: _CACHE_MAX_SIZE // 2]:
            del _cache[key]


def get_stats() -> dict:
    total = _stats["total_requests"]
    return {
        **_stats,
        "avg_time": round(_stats["total_time"] / total, 2) if total else 0.0,
        "min_time": round(_stats["min_time"], 2) if _stats["min_time"] != float("inf") else 0.0,
    }


def clear_cache() -> None:
    _cache.clear()


def reset_stats() -> None:
    _stats.update(total_requests=0, cache_hits=0, successful=0, errors=0,
                  total_time=0.0, max_time=0.0, min_time=float("inf"))


def clean_answer(text: str | None, context: str = "", question: str = "") -> str:
    """Убирает строки «Источник: ...», написанные моделью, и нормализует отказ.

    Отказом считается только короткий ответ с оговоркой: если модель дала инструкцию
    и в конце добавила «не нашел точного ответа», полезный текст не выбрасываем.
    """

    text = (text or "").strip()

    if not text:
        return NO_ANSWER

    if _REFUSAL_RE.search(text) and len(text) <= _REFUSAL_MAX_LEN:
        return NO_ANSWER

    # Небольшие модели «дописывают диалог»: после ответа ставят «Источник» и придумывают новый вопрос-ответ.
    # Всё после первой строки «Источник» и после повтора вопроса отбрасываем.
    cut = _SOURCE_LINE_RE.search(text)
    if cut and cut.start() >= 40:
        text = text[:cut.start()].strip()

    q = (question or "").strip().lower()
    if len(q) >= 8:
        pos = text.lower().find(q, 1)
        if pos >= 40:
            text = text[:pos].strip()

    text = _SOURCE_LINE_RE.sub("", text).strip()
    text = _REFUSAL_RE.sub("", text).strip() if _REFUSAL_RE.search(text) else text

    return text or NO_ANSWER


CONDENSE_PROMPT = (
    "Перепиши последний вопрос пользователя так, чтобы он был понятен без истории диалога. "
    "Подставь вместо местоимений и пропущенных слов конкретную тему из предыдущих реплик. "
    "Ничего не отвечай и не добавляй. Выведи ТОЛЬКО переписанный вопрос одной строкой. "
    "Если вопрос и так понятен сам по себе, выведи его без изменений."
)
CONDENSE_TIMEOUT = float(os.getenv("CONDENSE_TIMEOUT", "20"))


def _short(text: str, limit: int) -> str:
    text = _SOURCE_LINE_RE.sub("", text or "").strip().replace("\n", " ")
    return text[:limit]


async def condense_question(history: list, question: str, topic: str | None = None) -> str | None:
    """Самостоятельная формулировка уточняющего вопроса для поиска («а как его удалить?» -> «Как удалить фильтр?»).

    Любой сбой или странный ответ даёт None: тогда остаётся простая склейка с прошлым вопросом.
    """
    turns = []
    items = list(reversed(history or []))[-4:]      # последние две пары, от старых к новым

    for item in items:
        if item.startswith("Пользователь:"):
            turns.append(f"Пользователь: {_short(item.split(':', 1)[1], 200)}")
        elif item.startswith("Бот:"):
            turns.append(f"Ассистент: {_short(item.split(':', 1)[1], 250)}")

    if topic:
        turns.insert(0, f"Тема диалога: {_short(topic, 200)}")

    if not turns:
        return None

    messages = [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": "Диалог:\n" + "\n".join(turns) + f"\n\nПоследний вопрос: {question}\nПереписанный вопрос:"},
    ]

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                _get_client().chat.completions.create,
                model=OLLAMA_MODEL, messages=messages, temperature=0, max_tokens=80,
            ),
            timeout=CONDENSE_TIMEOUT,
        )
        text = (response.choices[0].message.content or "").strip().split("\n")[0].strip(" \"'«»")
    except Exception as e:
        log.warning("не удалось переформулировать вопрос: %s", e)
        return None

    if not (5 <= len(text) <= 200) or _REFUSAL_RE.search(text):
        return None

    return text


async def ask_llm(context: str, history: list, question: str, image_base64: str | None = None) -> str:
    """Возвращает ответ по контексту или дословный NO_ANSWER. При сбое модели бросает LLMError."""

    # Без фрагментов отвечаем отказом, но вопрос по картинке можно обработать и без текста документов
    if (not context or not context.strip()) and not image_base64:
        return NO_ANSWER

    _stats["total_requests"] += 1
    started = time.time()

    cache_key = _make_cache_key(context, question)

    # В кэше только ответы без истории-зависимости: отказ и уточняющие вопросы не кэшируем
    if cache_key in _cache and not history and not image_base64:
        _stats["cache_hits"] += 1
        return _cache[cache_key]

    messages = build_prompt(context, history, question, has_image=bool(image_base64))

    if image_base64:
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["content"] = [
                    {"type": "text", "text": msg["content"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ]
                break

    try:
        response = await asyncio.to_thread(
            _get_client().chat.completions.create,
            model=VISION_MODEL if image_base64 else OLLAMA_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=LLM_MAX_TOKENS,
            stop=["\nИсточник", "\n\nВопрос:"],
        )
        answer = clean_answer(response.choices[0].message.content, context, question)

        if answer != NO_ANSWER and not history and not image_base64:
            _cache[cache_key] = answer
            _cleanup_cache()

        _stats["successful"] += 1
        return answer

    except Exception as e:
        _stats["errors"] += 1
        log.error("Ошибка LLM: %s", e)
        raise LLMError(str(e)) from e

    finally:
        elapsed = time.time() - started
        _stats["total_time"] += elapsed
        _stats["max_time"] = max(_stats["max_time"], elapsed)
        _stats["min_time"] = min(_stats["min_time"], elapsed)
        log.info("LLM: %.2f с, успехов %s, ошибок %s", elapsed, _stats["successful"], _stats["errors"])
