
"""Диагностика полного RAG pipeline."""

import sys
import asyncio
import inspect
from pathlib import Path
from collections import Counter

# Добавляем api_service в путь импорта
ROOT = Path(__file__).resolve().parent
API_DIR = ROOT / "api_service"
sys.path.insert(0, str(API_DIR))

from parser import list_document_files, parse_file
from qdrant_store import (
    get_client,
    COLLECTION_NAME,
    stats,
    count_document,
)
from search import search, search_qdrant
from llm_service import ask_llm


DOCS_DIR = ROOT / "api_service" / "docs"

# Контрольные вопросы.
# При необходимости добавь сюда вопросы по своим документам.
QUESTIONS = [
    "Что нужно указывать в поле Модель?",
    "Как выполнить массовое добавление записи?",
    "Какая сейчас цена доллара к евро?",
]


def print_line():
    print("-" * 70)


def check_parsing():
    """Проверяем парсинг и качество чанков."""

    print("\n1. ПРОВЕРКА ПАРСИНГА")
    print_line()

    if not DOCS_DIR.exists():
        print(f"Папка не найдена: {DOCS_DIR}")
        return {}

    files = list_document_files(str(DOCS_DIR))
    result = {}

    if not files:
        print("Документы не найдены.")
        return result

    for path in files:
        print(f"\nДокумент: {path.name}")

        try:
            chunks = parse_file(path)
        except Exception as e:
            print(f"ОШИБКА ПАРСИНГА: {e}")
            continue

        if not chunks:
            print("ВНИМАНИЕ: парсер не вернул чанки.")
            result[path.name] = []
            continue

        types = Counter(c.get("type", "unknown") for c in chunks)
        lengths = [
            len(c.get("text", ""))
            for c in chunks
            if c.get("type") == "text"
        ]

        print(f"Всего чанков: {len(chunks)}")
        print(f"Текстовых: {types.get('text', 0)}")
        print(f"Изображений: {types.get('image', 0)}")

        if lengths:
            print(f"Минимальная длина текста: {min(lengths)}")
            print(f"Максимальная длина текста: {max(lengths)}")
            print(f"Средняя длина текста: {sum(lengths) / len(lengths):.1f}")

        empty = [
            i for i, c in enumerate(chunks)
            if not c.get("text", "").strip()
        ]

        if empty:
            print(f"ВНИМАНИЕ: пустые чанки: {empty}")

        missing_metadata = [
            i for i, c in enumerate(chunks)
            if not c.get("source") or c.get("page") is None
        ]

        if missing_metadata:
            print(
                "ВНИМАНИЕ: чанки без source или page:",
                missing_metadata[:10]
            )

        images = [c for c in chunks if c.get("type") == "image"]

        if images:
            methods = Counter(
                c.get("method", "unknown") for c in images
            )
            print(f"Способы обработки изображений: {dict(methods)}")

        print("\nПримеры чанков:")

        for i, chunk in enumerate(chunks[:3]):
            print_line()
            print(f"Чанк {i}")
            print("Источник:", chunk.get("source"))
            print("Страница:", chunk.get("page"))
            print("Раздел:", chunk.get("section"))
            print("Тип:", chunk.get("type"))
            print("Текст:", chunk.get("text", "")[:400])

        result[path.name] = chunks

    return result


def check_qdrant(parsed):
    """Сравниваем локальный парсинг с содержимым Qdrant."""

    print("\n\n2. ПРОВЕРКА QDRANT")
    print_line()

    client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Коллекция {COLLECTION_NAME} не существует.")
        return

    database = stats(client)

    total = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    ).count

    print(f"Коллекция: {COLLECTION_NAME}")
    print(f"Всего точек в Qdrant: {total}")

    print("\nСравнение парсинга и базы:")

    for name, chunks in parsed.items():
        parsed_count = len(chunks)
        database_count = count_document(client, name)

        print_line()
        print(name)
        print(f"После парсинга: {parsed_count}")
        print(f"В Qdrant: {database_count}")

        if parsed_count == database_count:
            print("Количество совпадает.")
        else:
            print("ВНИМАНИЕ: количество не совпадает!")

    print("\nФактический состав базы:")

    for name, values in sorted(database.items()):
        print(
            f"{name}: "
            f"текст={values.get('text', 0)}, "
            f"изображения={values.get('image', 0)}"
        )


async def check_search_and_llm():
    """Проверяем поиск, контекст и ответ LLM."""

    print("\n\n3. ПРОВЕРКА ПОИСКА И LLM")
    print_line()

    for question in QUESTIONS:
        print(f"\nВОПРОС: {question}")
        print_line()

        # 1. Проверяем результаты поиска
        try:
            hits = search(question, top_k=5)

            if inspect.isawaitable(hits):
                hits = await hits

            print("\nРЕЗУЛЬТАТЫ ПОИСКА:")

            if not hits:
                print("Поиск ничего не вернул.")
            else:
                for i, hit in enumerate(hits, 1):
                    print_line()
                    print(f"Результат {i}")
                    print("Score:", hit.get("score"))
                    print("Источник:", hit.get("source"))
                    print("Страница:", hit.get("page"))
                    print("Тип:", hit.get("type"))
                    print("Текст:", hit.get("text", "")[:500])

        except Exception as e:
            print(f"ОШИБКА search(): {e}")

        # 2. Получаем контекст штатной функции поиска
        try:
            context, payload = await search_qdrant(question)

            print("\nКОНТЕКСТ ДЛЯ LLM:")
            print_line()

            if context is None:
                print("Контекст отсутствует.")
                print("LLM не вызывается.")
                continue

            print(context)

            print("\nМЕТАДАННЫЕ:")
            print(payload)

        except Exception as e:
            print(f"ОШИБКА search_qdrant(): {e}")
            continue

        # 3. Передаём именно этот контекст в LLM
        try:
            print("\nОТВЕТ LLM:")
            print_line()

            answer = ask_llm(
                context=context,
                history=[],
                question=question
            )

            if inspect.isawaitable(answer):
                answer = await answer

            print(answer)

        except Exception as e:
            print(f"ОШИБКА ask_llm(): {e}")

        print_line()


async def main():
    print("=" * 70)
    print("ДИАГНОСТИКА CORPORATE KNOWLEDGE ASSISTANT")
    print("=" * 70)

    # Этап 1: парсинг
    parsed = check_parsing()

    # Этап 2: Qdrant
    try:
        check_qdrant(parsed)
    except Exception as e:
        print(f"\nОШИБКА ПРОВЕРКИ QDRANT: {e}")

    # Этапы 3–5: поиск, контекст, LLM
    await check_search_and_llm()

    print("\n\nДИАГНОСТИКА ЗАВЕРШЕНА.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
