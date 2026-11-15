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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "500"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))

_client: openai.OpenAI | None = None

_SOURCE_LINE_RE = re.compile(r"(?im)^\s*источник(?:и)?\s*[:—-].*$")
_REFUSAL_RE = re.compile(r"не нашел точного ответа|не нашла точного ответа|__NO_ANSWER__", re.IGNORECASE)
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
    raw = f"{context.strip().lower()}|{question.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cleanup_cache() -> None:
    if len(_cache) > _CACHE_MAX_SIZE:
        for key in list(_cache.keys())[: _CACHE_MAX_SIZE // 2]:
            del _cache[key]


def clean_answer(text: str | None, context: str = "", question: str = "") -> str:
    text = (text or "").strip()
    if not text:
        return NO_ANSWER

    if _REFUSAL_RE.search(text) and len(text) <= _REFUSAL_MAX_LEN:
        return NO_ANSWER

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


async def ask_llm(context: str, history: list, question: str, image_base64: str | None = None) -> str:
    if not context or not context.strip():
        return NO_ANSWER

    _stats["total_requests"] += 1
    started = time.time()

    cache_key = _make_cache_key(context, question)

    if cache_key in _cache and not history:
        _stats["cache_hits"] += 1
        return _cache[cache_key]

    messages = build_prompt(context, history, question)

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
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=LLM_MAX_TOKENS,
            stop=["\nИсточник", "\n\nВопрос:"],
        )
        answer = clean_answer(response.choices[0].message.content, context, question)

        if answer != NO_ANSWER and not history:
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