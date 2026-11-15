import asyncio
import os
import re
import hashlib
import openai
from prompts import NO_ANSWER, build_prompt

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://sprashivaem_u_natashi:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "sprashivaem_u_natashi")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

_client: openai.OpenAI | None = None
_SOURCE_LINE_RE = re.compile(r"(?im)^\s*источник(?:и)?\s*:.*$")
_REFUSAL_RE = re.compile(r"не нашел точного ответа", re.IGNORECASE)

_cache: dict[str, str] = {}
_CACHE_MAX_SIZE = 1000


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


def clean_answer(text: str | None) -> str:
    text = (text or "").strip()
    if not text or _REFUSAL_RE.search(text):
        return NO_ANSWER
    text = _SOURCE_LINE_RE.sub("", text).strip()
    return text or NO_ANSWER


async def ask_llm(context: str, history: list, question: str, image_base64: str | None = None) -> str:
    cache_key = _make_cache_key(context, question)

    if cache_key in _cache:
        print(f"[LLM] Кэш-попадание! Возвращаю сохранённый ответ.")
        return _cache[cache_key]

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
    except Exception as e:
        raise LLMError(str(e)) from e

    answer = clean_answer(response.choices[0].message.content)
    _cache[cache_key] = answer
    _cleanup_cache()
    print(f"[LLM] Кэш-промах. Новый ответ сохранен. Размер кэша: {len(_cache)}")
    return answer


def clear_cache():
    _cache.clear()
    print(f"[LLM] Кэш очищен.")