import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

import llm_service
from prompts import build_prompt
from search import search_qdrant


def read_query() -> str:
    raw = open("q.txt", "rb").read()
    try:
        return raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return raw.decode("cp1251").strip()


async def main():
    question = read_query()
    print("question :", question)
    print("model    :", llm_service.OLLAMA_MODEL, "| url:", llm_service.OLLAMA_BASE_URL)

    context, payload = await search_qdrant(question, raise_errors=True)
    if not context:
        print("\nSEARCH: nothing accepted -> bot refuses before calling the model")
        return
    print(f"\nSEARCH: ok, context {len(context)} chars, best source: {payload['source']}, score {payload['score']}")
    print("images:", payload.get("images"))

    print("\nCONTEXT SENT TO MODEL (first 1800 chars):\n" + context[:1800])
    messages = build_prompt(context, [], question)
    print("prompt chars:", sum(len(m["content"]) for m in messages))

    resp = llm_service._get_client().chat.completions.create(
        model=llm_service.OLLAMA_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=llm_service.LLM_MAX_TOKENS,
        stop=["\nИсточник", "\n\nВопрос:"],
    )
    raw = resp.choices[0].message.content
    print("\nRAW MODEL OUTPUT (finish_reason=%s):\n%s" % (resp.choices[0].finish_reason, raw))
    print("\nAFTER clean_answer:\n%s" % llm_service.clean_answer(raw, context, question))


asyncio.run(main())
