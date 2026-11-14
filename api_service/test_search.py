
"""
Расширенное тестирование RAG-поиска.

Проверяем:
1. Находит ли поиск правильный документ.
2. Попадает ли он в топ-5 и на какую позицию.
3. Как поиск справляется с разговорными формулировками и опечатками.
4. Какие score получают релевантные и нерелевантные запросы.
5. Можно ли подобрать порог уверенности.

Запуск:
    python test_search.py
"""

import asyncio

from search import format_source, search, search_qdrant


KW_FUNC = "функциональные"   # Функциональные возможности с БАРС-Офисом
KW_BP = "командировка"       # БП Командировка
KW_FURN = "мебели"           # Реестр мебели

TOP_N = 5


# ============================================================
# 1. БАЗОВЫЕ ПОЛОЖИТЕЛЬНЫЕ ТЕСТЫ
# ============================================================

TEST_CASES = [
    # 12 тест-кейсов из брифа: (группа, вопрос, ключевое слово документа)
    ("БАРС Офис", "Как выглядит интерфейс БО? Где найти в нем производственный календарь?", KW_FUNC),
    ("БАРС Офис", "Как работает фильтр документа? Опиши шаг действий для настройки фильтра", KW_FUNC),
    ("БАРС Офис", 'Почему не работает фильтр в БО "!> 200"?', KW_FUNC),
    ("БАРС Офис", "Мне прислали ссылку https://bo.bars.group//#mainmenuitemclick&bars_office&byudjeetsdelki что с ней делать?", KW_FUNC),
    ("БП Командировка", "Мне необходимо узнать, кто согласовывает БП Командировки?", KW_BP),
    ("БП Командировка", "Мне не отвечает офис-менеджер, какой у него график работы?", KW_BP),
    ("БП Командировка", "Почему не могу выбрать сделку 2334-26, у которой статус воронки 30% в поле Номер пресейла?", KW_BP),
    ("БП Командировка", "Смогу ли я запустить БП Командировка, если не заложил средств в бюджет сделки?", KW_BP),
    ("БП Командировка", "Какие могут быть условия в выборе следующего согласующего лица БП Командировка?", KW_BP),
    ("Реестр мебели", "Какие функциональные кнопки есть в реестре мебели? Как они выглядят?", KW_FURN),
    # В документах есть только реестр МЕБЕЛИ: вероятно опечатка в брифе или ловушка.
    ("Реестр мебели", "Опиши поэтапно как в БО добавить запись в реестр техники?", KW_FURN),
    ("Реестр мебели", "Какие статусы мебели перечислены в реестре техники?", KW_FURN),
]


# ============================================================
# 2. СЛОЖНЫЕ ПОЛОЖИТЕЛЬНЫЕ ТЕСТЫ
# Другие формулировки уже известных вопросов.
# ============================================================

HARD_TEST_CASES = [
    # БАРС Офис — разговорные формулировки
    ("БАРС Офис", "Где в системе найти основное меню?", KW_FUNC),
    ("БАРС Офис", "Как мне перейти к производственному календарю?", KW_FUNC),
    ("БАРС Офис", "Где посмотреть свой профиль в БАРС?", KW_FUNC),
    ("БАРС Офис", "Что находится в панели быстрого доступа?", KW_FUNC),

    # БАРС Офис — фильтры и работа с документами
    ("БАРС Офис", "Как отфильтровать документы по нужному условию?", KW_FUNC),
    ("БАРС Офис", "Можно ли оставить настройки фильтра для следующего открытия?", KW_FUNC),
    ("БАРС Офис", "Как убрать фильтр, который я раньше создал?", KW_FUNC),
    ("БАРС Офис", "Почему фильтр с условием !> 200 не срабатывает?", KW_FUNC),
    ("БАРС Офис", "Как вывести записи в группы по колонке?", KW_FUNC),
    ("БАРС Офис", "Можно ли посмотреть документ в отдельном окне?", KW_FUNC),
    ("БАРС Офис", "Как выгрузить данные документа в Excel?", KW_FUNC),

    # Опечатки и сокращения
    ("БАРС Офис", "Где находится главноe меню БАРСа?", KW_FUNC),
    ("БАРС Офис", "Как настроить фильтр для докумнта?", KW_FUNC),
    ("БАРС Офис", "Где в БО находится профиль пользователя?", KW_FUNC),

    # Реестр мебели — разговорные формулировки
    ("Реестр мебели", "Зачем вообще нужен реестр мебели?", KW_FURN),
    ("Реестр мебели", "Кто может работать с этим реестром?", KW_FURN),
    ("Реестр мебели", "Что записывают в реестр по каждому предмету мебели?", KW_FURN),
    ("Реестр мебели", "Что нужно указывать в поле Модель?", KW_FURN),
    ("Реестр мебели", "В каком случае предмет мебели считается занятым?", KW_FURN),
    ("Реестр мебели", "Как оформить списание предмета мебели?", KW_FURN),
]


# ============================================================
# 3. ОТРИЦАТЕЛЬНЫЕ ТЕСТЫ
# Ответа на эти вопросы нет в текущей документации.
# ============================================================

TRICK_CASES = [
    # Совершенно посторонние вопросы
    "Какая завтра погода в Москве?",
    "Как приготовить борщ?",
    "Кто выиграл чемпионат мира по футболу?",
    "Как настроить принтер дома?",
    "Как подключить домашний Wi-Fi роутер?",
    "Какая сейчас цена доллара к евро?",
    "Как установить Python на Windows?",
    "Как приготовить шоколадный торт?",
    "Как оформить загранпаспорт?",
    "Какие фильмы сейчас идут в кинотеатрах?",
    "Как заменить аккумулятор в телефоне?",
    "Как выбрать туристическую палатку?",

    # Тематически близкие вопросы.
    # Перед использованием нужно убедиться, что ответов на них нет в PDF.
    "Как восстановить забытый пароль от БАРС Офиса?",
    "Как создать новую учетную запись сотрудника в БАРС Офисе?",
    "Как изменить пароль другого пользователя?",
    "Как прикрепить фотографию предмета мебели к записи?",
    "Как заказать новую мебель для подразделения?",
    "Как перенести мебель из одного кабинета в другой?",
    "Как рассчитать стоимость мебели по данным реестра?",
]


# ============================================================
# 4. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================

def show(hits: list[dict]) -> None:
    for i, h in enumerate(hits, 1):
        print(
            f"  {i}. [{h['score']:.4f}] "
            f"({h['type']}) {format_source(h)}"
        )
        print(f"     {h['text'][:150]}...")


# ============================================================
# 5. ПОЛОЖИТЕЛЬНЫЕ ТЕСТЫ
# ============================================================

def run_normal() -> tuple[list[float], int, int, float]:
    print(f"\n{'=' * 75}")
    print("ПОЛОЖИТЕЛЬНЫЕ ТЕСТЫ")
    print(f"{'=' * 75}")

    all_cases = TEST_CASES + HARD_TEST_CASES

    relevant_scores = []
    group_results = {}

    hit1 = 0
    hit5 = 0
    reciprocal_rank_sum = 0.0

    for group, question, keyword in all_cases:
        hits = search(question, top_k=TOP_N)

        matching = [
            (i, h)
            for i, h in enumerate(hits, 1)
            if keyword in h["source"].lower()
        ]

        ok = bool(matching)
        rank = matching[0][0] if matching else None

        best_score = hits[0]["score"] if hits else 0.0

        # Сохраняем score первого результата поиска
        # для сравнения с отрицательными тестами
        relevant_scores.append(best_score)

        if rank is not None:
            hit5 += 1
            reciprocal_rank_sum += 1 / rank

            if rank == 1:
                hit1 += 1

        if group not in group_results:
            group_results[group] = [0, 0]

        group_results[group][1] += 1
        group_results[group][0] += int(ok)

        status = "OK" if ok else "FAIL"

        print(f"\n[{status}] {group}: {question}")
        print(f"Лучший score: {best_score:.4f}")

        if rank is not None:
            print(f"Нужный документ найден на позиции: {rank}")
        else:
            print(f"Нужный документ не найден в топ-{TOP_N}")

        show(hits)

    total = len(all_cases)

    print(f"\n{'-' * 75}")
    print("РЕЗУЛЬТАТЫ ПО ГРУППАМ")

    for group, (success, count) in group_results.items():
        print(f"{group}: {success}/{count}")

    print(f"\nВсего найдено нужных документов: {hit5}/{total}")
    print(f"Hit@1: {hit1 / total:.2%}")
    print(f"Hit@{TOP_N}: {hit5 / total:.2%}")
    print(f"MRR@{TOP_N}: {reciprocal_rank_sum / total:.4f}")

    return relevant_scores, hit1, hit5, reciprocal_rank_sum / total


# ============================================================
# 6. ОТРИЦАТЕЛЬНЫЕ ТЕСТЫ
# ============================================================

def run_trick() -> list[float]:
    print(f"\n{'=' * 75}")
    print("НЕГАТИВНЫЕ ТЕСТЫ — ОТВЕТА В ДОКУМЕНТАХ НЕТ")
    print(f"{'=' * 75}")

    scores = []

    for question in TRICK_CASES:
        hits = search(question, top_k=TOP_N)

        best_score = hits[0]["score"] if hits else 0.0
        scores.append(best_score)

        print(f"\n{question}")
        print(f"Лучший score: {best_score:.4f}")

        show(hits)

    return scores


# ============================================================
# 7. АНАЛИЗ ПОРОГА УВЕРЕННОСТИ
# ============================================================

def show_threshold(
    normal: list[float],
    trick: list[float]
) -> None:
    print(f"\n{'=' * 75}")
    print("АНАЛИЗ ПОРОГА УВЕРЕННОСТИ")
    print(f"{'=' * 75}")

    if not normal or not trick:
        print("Недостаточно данных для анализа порога.")
        return

    min_normal = min(normal)
    max_normal = max(normal)

    min_trick = min(trick)
    max_trick = max(trick)

    print(
        f"Вопросы с ответом: "
        f"min={min_normal:.4f}, max={max_normal:.4f}"
    )

    print(
        f"Вопросы без ответа: "
        f"min={min_trick:.4f}, max={max_trick:.4f}"
    )

    if min_normal > max_trick:
        threshold = (min_normal + max_trick) / 2

        print("\nГруппы полностью разделяются.")
        print(f"Предварительный порог: {threshold:.4f}")

    else:
        print("\nГруппы пересекаются.")
        print("Одним порогом полностью разделить запросы не получится.")

    # Проверяем несколько возможных порогов
    candidates = sorted(set(normal + trick + [1.0]))

    best_threshold = None
    best_accuracy = 0

    print("\nПроверка возможных порогов:")

    for threshold in candidates:
        accepted = sum(s >= threshold for s in normal)
        rejected = sum(s < threshold for s in trick)

        recall = accepted / len(normal)
        specificity = rejected / len(trick)

        balanced_accuracy = (recall + specificity) / 2

        if balanced_accuracy > best_accuracy:
            best_accuracy = balanced_accuracy
            best_threshold = threshold

    if best_threshold is not None:
        accepted = sum(s >= best_threshold for s in normal)
        rejected = sum(s < best_threshold for s in trick)

        print(f"\nЛучший проверенный порог: {best_threshold:.4f}")
        print(
            f"Вопросы с ответом, принятые системой: "
            f"{accepted}/{len(normal)}"
        )
        print(
            f"Вопросы без ответа, отклоненные системой: "
            f"{rejected}/{len(trick)}"
        )
        print(
            f"Balanced Accuracy: {best_accuracy:.2%}"
        )


# ============================================================
# 8. ПРОВЕРКА КОНТРАКТА search_qdrant
# ============================================================

async def check_contract() -> None:
    print(f"\n{'=' * 75}")
    print("ПРОВЕРКА КОНТРАКТА search_qdrant")
    print(f"{'=' * 75}")

    queries = [
        "Как изменить запись в реестре мебели?",
        "Какая завтра погода в Москве?",
        "",
    ]

    for question in queries:
        context, payload = await search_qdrant(question)

        if context is None:
            assert payload is None
            print(f"[None, None] {question!r}")

        else:
            assert isinstance(context, str)
            assert isinstance(payload, dict)
            assert "source" in payload
            assert "page" in payload

            print(
                f"[OK] {question!r} -> "
                f"{payload['source']}, "
                f"page={payload['page']}, "
                f"score={payload.get('score')}"
            )

    print("Контракт соблюдён")


# ============================================================
# 9. ЗАПУСК ВСЕХ ТЕСТОВ
# ============================================================

def main() -> None:
    normal, hit1, hit5, mrr = run_normal()
    trick = run_trick()

    show_threshold(normal, trick)


if __name__ == "__main__":
    main()
    asyncio.run(check_contract())
