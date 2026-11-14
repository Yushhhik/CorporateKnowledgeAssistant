import asyncio
import re
from llm_service import ask_llm
STANDARD_QUESTIONS = [
    ("Как выглядит интерфейс БО? Где найти в нем производственный календарь?", "Тест 1"),
    ("Как работает фильтр документа? Опиши шаг действий для настройки фильтра.", "Тест 2"),
    ("Мне необходимо узнать, кто согласовывает БП Командировки?", "Тест 5"),
    ("Мне не отвечает офис-менеджер, какой у него график работы?", "Тест 6"),
    ("Какие могут быть условия в выборе следующего согласующего лица БП Командировка?", "Тест 9"),
    ("Какие функциональные кнопки есть в реестре мебели? Как они выглядят?", "Тест 10"),
    ("Опиши поэтапно как в БО добавить запись в реестр мебели?", "Тест 11"),
    ("Какие статусы мебели перечислены в реестре техники?", "Тест 13"),
]

TRAP_QUESTIONS = [
    ("Почему не работает фильтр в БО \"!> 200\"?", "Тест 3 (Сложный фильтр)"),
    ("Мне прислали ссылку https://bo.bars.group/... что с ней делать?", "Тест 4 (Внутренняя ссылка)"),
    ("Почему не могу выбрать сделку 2334-26, у которой статус воронки 30% в поле Номер пресейла?", "Тест 7 (Пресейл)"),
    ("Смогу ли я запустить БП Командировка, если не заложил средств в бюджет сделки?", "Тест 8 (Бюджет)"),
    ("Опиши поэтапно как в БО добавить запись в реестр техники?", "Тест 12 (ЛОВУШКА: техника vs мебель)"),
    ("Как приготовить борщ?", "Тест 14 (ВНЕ БАЗЫ)"),
]

HALLUCINATION_TRAPS = [
    "Какой у бота любимый цвет?",
    "Сколько стоит лицензия БАРС-Офис?",
    "Расскажи про конкурентов БАРС Груп",
]

MOCK_CONTEXT = (
    "Документ предназначен для ведения реестра мебели. Статусы: занят, свободен, списан. "
    "Для добавления записи нажмите кнопку 'Добавить', заполните поля (Модель, Подразделение, Кабинет, Инвентарный номер), нажмите 'Сохранить'. "
    "График дежурства офис-менеджеров: Понедельник – Пятница с 17:30 до 21:00. "
    "Источник: Реестр мебели.pdf, стр. 9; БП Командировка.pdf, стр3."
)


def check_answer(question: str, answer: str) -> bool:
    answer_lower = answer.lower()
    if "борщ" in question.lower() or "цвет" in question.lower() or "конкурент" in question.lower():
        return "не нашел" in answer_lower or "оператор" in answer_lower
    if "техник" in question.lower() and "добавить" in question.lower():
        return "мебель" in answer_lower and ("нет" in answer_lower or "имели в виду" in answer_lower)
    has_source = bool(re.search(r"[Ии]сточник:", answer))
    return has_source


async def run_trap_tests():
    print("Тест защиты от галлюцинаций (Критерий 2 + Чек-лист)")
    passed = 0
    failed = 0
    total = 0
    print("\nКатегория 1: Стандартные запросы (проверка источника)")
    for q, test_name in STANDARD_QUESTIONS:
        total += 1
        answer = await ask_llm(MOCK_CONTEXT, [], q)
        ok = check_answer(q, answer)
        status = "🟢 PASS" if ok else "🔴 FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{status} | {test_name}")
        if not ok:
            print(f"Ответ не содержит 'Источник:' или некорректен:\n  {answer[:150]}...")

    print("\nКатегория 2: Ловушки и сложные кейсы")
    for q, test_name in TRAP_QUESTIONS:
        total += 1
        answer = await ask_llm(MOCK_CONTEXT, [], q)
        ok = check_answer(q, answer)
        status = "🟢 PASS" if ok else "🔴 FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{status} | {test_name}")
        if not ok:
            print(f"Бот не распознал ловушку или не дал источник:\n  {answer[:200]}...")

    print("\nКатегория 3: Жесткие проверки на выдумки")
    for q in HALLUCINATION_TRAPS:
        total += 1
        answer = await ask_llm("", [], q)
        ok = check_answer(q, answer)
        status = "🟢 PASS" if ok else "🔴 FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{status} | Вопрос вне базы: '{q}'")
        if not ok:
            print(f"Бот начал выдумывать:\n  {answer[:150]}...")

    print(f"Итого: {passed}/{total} 🟢 PASS, {failed} 🔴 FAIL")

    if failed == 0:
        print("Защита от галлюцинаций работает корректно по всему чек-листу.")
    else:
        print("Защита от галлюцинаций работает некорректно.")


if __name__ == "__main__":
    asyncio.run(run_trap_tests())