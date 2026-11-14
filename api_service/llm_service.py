import asyncio
import os
import re
import openai
from prompts import NO_ANSWER, build_prompt

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://sprashivaem_u_natashi:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "sprashivaem_u_natashi")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

_client: openai.OpenAI | None = None

_SOURCE_LINE_RE = re.compile(r"(?im)^\s*источник(?:и)?\s*:.*$")
_REFUSAL_RE = re.compile(r"не нашел точного ответа", re.IGNORECASE)


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


def clean_answer(text: str | None) -> str:
    text = (text or "").strip()
    if not text or _REFUSAL_RE.search(text):
        return NO_ANSWER
    text = _SOURCE_LINE_RE.sub("", text).strip()
    return text or NO_ANSWER


async def ask_llm(context: str, history: list, question: str, image_base64: str | None = None) -> str:
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
    return clean_answer(response.choices[0].message.content)