import asyncio
import os
import re
import hashlib
import time
import openai
from prompts import NO_ANSWER, build_prompt

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://sprashivaem_u_natashi:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "sprashivaem_u_natashi")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

_client: openai.OpenAI | None = None
_SOURCE_LINE_RE = re.compile(r"(?im)^\s*источник(?:и)?\s*[:—-].*$")
_REFUSAL_RE = re.compile(r"не нашел точного ответа", re.IGNORECASE)

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
    pass


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
    context_norm = context.strip().lower()
    question_norm = question.strip().lower()
    raw_key = f"{context_norm}|{question_norm}"
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _cleanup_cache():
    if len(_cache) > _CACHE_MAX_SIZE:
        keys_to_remove = list(_cache.keys())[:_CACHE_MAX_SIZE // 2]
        for key in keys_to_remove:
            del _cache[key]


def _update_stats_time(elapsed: float):
    _stats["total_time"] += elapsed
    if elapsed > _stats["max_time"]:
        _stats["max_time"] = elapsed
    if elapsed < _stats["min_time"]:
        _stats["min_time"] = elapsed


def get_stats() -> dict:
    total = _stats["total_requests"]
    avg_time = round(_stats["total_time"] / total, 2) if total > 0 else 0.0
    return {**_stats, "avg_time": avg_time,
            "min_time": round(_stats["min_time"], 2) if _stats["min_time"] != float("inf") else 0.0}


def print_stats():
    s = get_stats()
    print("Статистика LLM сервиса")
    print(f"Всего запросов:      {s['total_requests']}")
    print(f"Попаданий в кэш:     {s['cache_hits']}")
    print(f"Успешных ответов:    {s['successful']}")
    print(f"Ошибок:              {s['errors']}")
    print(f"Среднее время:       {s['avg_time']} сек")
    print(f"Макс. время ответа:  {s['max_time']:.2f} сек")
    print(f"Мин. время ответа:   {s['min_time']:.2f} сек")


def clean_answer(text: str | None, context: str = "") -> str:
    text = (text or "").strip()
    if not text or _REFUSAL_RE.search(text):
        return NO_ANSWER
    source_match = _SOURCE_LINE_RE.search(text)
    source_text = source_match.group(0).strip() if source_match else None
    text = _SOURCE_LINE_RE.sub("", text).strip()
    if not source_text:
        source_text = "Источник: документация БАРС Груп"
    return f"{text}\n\n{source_text}" if text else NO_ANSWER


async def ask_llm(context: str, history: list, question: str, image_base64: str | None = None) -> str:
    _stats["total_requests"] += 1
    start_time = time.time()

    cache_key = _make_cache_key(context, question)

    if cache_key in _cache:
        _stats["cache_hits"] += 1
        print(f"[LLM] Кэш-попадание. Мгновенный ответ.")
        return _cache[cache_key]

    if not context or not context.strip():
        print("[LLM] Контекст пустой, LLM не вызывается")
        return NO_ANSWER

    messages = build_prompt(context, history, question)

    if image_base64:
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["content"] = [
                    {"type": "text", "text": msg["content"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
                break

    try:
        response = await asyncio.to_thread(
            _get_client().chat.completions.create,
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        answer = clean_answer(response.choices[0].message.content, context)

        _cache[cache_key] = answer
        _cleanup_cache()
        _stats["successful"] += 1
        print(f"[LLM] Кэш-промах. Новый ответ сохранен. Размер кэша: {len(_cache)}")
        return answer

    except Exception as e:
        _stats["errors"] += 1
        print(f"[LLM] Ошибка LLM: {e}")
        raise LLMError(str(e)) from e

    finally:
        elapsed = time.time() - start_time
        _stats["total_time"] += elapsed
        if elapsed > _stats["max_time"]:
            _stats["max_time"] = elapsed
        if elapsed < _stats["min_time"]:
            _stats["min_time"] = elapsed
        print(
            f"[LLM] Запрос #{_stats['total_requests']} | время: {elapsed:.2f} сек | успехов: {_stats['successful']} | ошибок: {_stats['errors']}")


def clear_cache():
    _cache.clear()
    print("[LLM] Кэш очищен.")


def reset_stats():
    global _stats
    _stats = {
        "total_requests": 0, "cache_hits": 0, "successful": 0, "errors": 0,
        "total_time": 0.0, "max_time": 0.0, "min_time": float("inf"),
    }
    print("[LLM] Статистика сброшена.")


if __name__ == "__main__":
    async def run_demo():
        print("Запуск демо-теста LLM сервиса с кэшем и метриками...\n")
        ctx = "Реестр мебели: кнопка Добавить, поля Модель, Подразделение, Кабинет. Источник: Реестр.pdf, стр. 5"
        test_questions = [
            "Как добавить запись в реестр мебели?",
            "Как добавить запись в реестр мебели?",
            "Как приготовить борщ?",
        ]

        for q in test_questions:
            print(f"\nВопрос: {q}")
            try:
                answer = await ask_llm(ctx, [], q)
                print(f"Ответ бота: {answer[:100]}...")
            except LLMError:
                print("LLM недоступна")

        print_stats()


    asyncio.run(run_demo())