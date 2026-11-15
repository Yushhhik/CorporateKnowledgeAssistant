import re
from typing import List, Dict

NO_ANSWER = "Я не нашел точного ответа в документации. Переключить на оператора?"

SYSTEM_PROMPT = """Ты — ассистент первой линии поддержки БАРС Груп. Отвечай на вопрос сотрудника, используя ТОЛЬКО фрагменты документации ниже.
Правила:
1. Отвечай строго на заданный вопрос и больше ни на что. Не предлагай и не придумывай другие вопросы и темы.
2. Если во фрагментах есть ответ или его часть — ответь по ним: перескажи шаги, названия кнопок, условия и цифры так, как они написаны. Если ответ неполный, дай то, что есть, и коротко скажи, чего не хватает.
3. Отказывайся ("__NO_ANSWER__", дословно и ничего больше) ТОЛЬКО если фрагменты вообще не про тему вопроса. Если фрагменты по теме, но точного ответа нет, перескажи самое близкое из них (разделы, экраны, кнопки, функции) и коротко скажи, что подробностей нет.
4. Ничего не выдумывай: названия кнопок, шаги, числа и ссылки бери только из фрагментов. Знания о других системах не используй.
5. Не пиши строку "Источник" и номера страниц: источники добавляются автоматически.
6. Отвечай по-русски, на "Вы", кратко и по делу; шаги оформляй нумерованным списком.
7. Опечатки и разговорные формулировки вопроса не мешают: опирайся на смысл.
Фрагменты документации:
__CONTEXT__"""

FURNITURE_NOTE = "В документации описан только реестр мебели, реестра техники нет."
_TECH_REGISTRY = re.compile(r"реестр\w*\s+(?:\w+\s+)?техник", re.IGNORECASE)


def furniture_note(question: str, context: str) -> str | None:
    if _TECH_REGISTRY.search(question or "") and "мебел" in (context or "").lower():
        return FURNITURE_NOTE
    return None


HISTORY_MESSAGES = 6
HISTORY_ITEM_CHARS = 300
_SOURCE_LINE = re.compile(r"(?im)^\s*источник(?:и)?\s*[:—-].*$")


def _is_refusal(text: str) -> bool:
    return text.startswith(NO_ANSWER[:30])


def _clean_history(history: List[str]) -> List[Dict]:
    items: List[Dict] = []
    for msg in reversed(history):
        if msg.startswith("Пользователь:"):
            role, content = "user", msg[len("Пользователь:"):].strip()
        elif msg.startswith("Бот:"):
            role, content = "assistant", msg[len("Бот:"):].strip()
        else:
            role, content = "user", msg.strip()

        if role == "assistant":
            content = _SOURCE_LINE.sub("", content).strip()
            if _is_refusal(content):
                if items and items[-1]["role"] == "user":
                    items.pop()
                continue

        if content:
            items.append({"role": role, "content": content[:HISTORY_ITEM_CHARS]})

    return items[-HISTORY_MESSAGES:]


def build_prompt(context: str, history: List[str], question: str) -> List[Dict]:
    if not context or context.strip() == "":
        system_content = (
            "Ты — корпоративный ассистент БАРС Груп. "
            "Контекст пустой, поэтому на любой вопрос отвечай дословно: "
            f"'{NO_ANSWER}'"
        )
    else:
        system_content = (
            SYSTEM_PROMPT
            .replace("__NO_ANSWER__", NO_ANSWER)
            .replace("__CONTEXT__", context)
        )

    system_message = {
        "role": "system",
        "content": system_content
    }

    history_messages = _clean_history(history)
    user_message = {"role": "user", "content": question}
    return [system_message] + history_messages + [user_message]