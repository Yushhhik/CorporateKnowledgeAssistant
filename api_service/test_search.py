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

KW_FUNC = "функциональные"
KW_BP = "командировка"
KW_FURN = "мебели"

TOP_N = 5


TEST_CASES = [
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
    ("Реестр мебели", "Опиши поэтапно как в БО добавить запись в реестр техники?", KW_FURN),
    ("Реестр мебели", "Какие статусы мебели перечислены в реестре техники?", KW_FURN),
]


HARD_TEST_CASES = [
    ("БАРС Офис", "Где в системе найти основное меню?", KW_FUNC),
    ("БАРС Офис", "Как мне перейти к производственному календарю?", KW_FUNC),
    ("БАРС Офис", "Где посмотреть свой профиль в БАРС?", KW_FUNC),
    ("БАРС Офис", "Что находится в панели быстрого доступа?", KW_FUNC),

    ("БАРС Офис", "Как отфильтровать документы по нужному условию?", KW_FUNC),
    ("БАРС Офис", "Можно ли оставить настройки фильтра для следующего открытия?", KW_FUNC),
    ("БАРС Офис", "Как убрать фильтр, который я раньше создал?", KW_FUNC),
    ("БАРС Офис", "Почему фильтр с условием !> 200 не срабатывает?", KW_FUNC),
    ("БАРС Офис", "Как вывести записи в группы по колонке?", KW_FUNC),
    ("БАРС Офис", "Можно ли посмотреть документ в отдельном окне?", KW_FUNC),
    ("БАРС Офис", "Как выгрузить данные документа в Excel?", KW_FUNC),

    ("БАРС Офис", "Где находится главноe меню БАРСа?", KW_FUNC),
    ("БАРС Офис", "Как настроить фильтр для докумнта?", KW_FUNC),
    ("БАРС Офис", "Где в БО находится профиль пользователя?", KW_FUNC),

    ("Реестр мебели", "Зачем вообще нужен реестр мебели?", KW_FURN),
    ("Реестр мебели", "Кто может работать с этим реестром?", KW_FURN),
    ("Реестр мебели", "Что записывают в реестр по каждому предмету мебели?", KW_FURN),
    ("Реестр мебели", "Что нужно указывать в поле Модель?", KW_FURN),
    ("Реестр мебели", "В каком случае предмет мебели считается занятым?", KW_FURN),
    ("Реестр мебели", "Как оформить списание предмета мебели?", KW_FURN),
]



KW_GARANT = "1 доп док"
KW_DOCSVISION = "2 доп док"

NEW_DOC_TEST_CASES = [
    ("Garantexpress", "Как создать маршрут согласования документа?", KW_GARANT),
    ("Garantexpress", "Как добавить участников в процесс согласования?", KW_GARANT),
    ("Garantexpress", "Как проходит согласование документа?", KW_GARANT),
    ("Garantexpress", "Как узнать, на каком этапе находится согласование?", KW_GARANT),
    ("Garantexpress", "Что делать, если документ не согласован?", KW_GARANT),

    ("Docsvision", "Как создать шаблон бизнес-процесса в Docsvision?", KW_DOCSVISION),
    ("Docsvision", "Как построить схему бизнес-процесса?", KW_DOCSVISION),
    ("Docsvision", "Какие элементы используются в конструкторе бизнес-процессов?", KW_DOCSVISION),
    ("Docsvision", "Как запустить выполнение бизнес-процесса?", KW_DOCSVISION),
    ("Docsvision", "Как отслеживать выполнение бизнес-процесса?", KW_DOCSVISION),
]

TRICK_CASES = [
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

    "Как восстановить забытый пароль от БАРС Офиса?",
    "Как создать новую учетную запись сотрудника в БАРС Офисе?",
    "Как изменить пароль другого пользователя?",
    "Как прикрепить фотографию предмета мебели к записи?",
    "Как заказать новую мебель для подразделения?",
    "Как перенести мебель из одного кабинета в другой?",
    "Как рассчитать стоимость мебели по данным реестра?",
    "Как настроить маршрут согласования в 1С?",
    "Как создать бизнес-процесс в SAP?",
    "Как настроить согласование документов в 1С?",
    "Как запустить workflow в SAP?",
]


def show(hits: list[dict]) -> None:
    for i, h in enumerate(hits, 1):
        print(
            f"  {i}. [{h['score']:.4f}] "
            f"({h['type']}) {format_source(h)}"
        )
        print(f"     {h['text'][:150]}...")


def run_normal() -> tuple[list[float], int, int, float]:
    print(f"\n{'=' * 75}")
    print("ПОЛОЖИТЕЛЬНЫЕ ТЕСТЫ")
    print(f"{'=' * 75}")

    all_cases = TEST_CASES + HARD_TEST_CASES + NEW_DOC_TEST_CASES
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


def show_threshold(
    normal: list[float],
    trick: list[float]
) -> None:
    print(f"\n{'=' * 75}")
    print("СРАВНЕНИЕ ПОРОГОВ УВЕРЕННОСТИ")
    print(f"{'=' * 75}")

    if not normal or not trick:
        print("Недостаточно данных для анализа порога.")
        return

    print(
        f"Вопросы с ответом: "
        f"min={min(normal):.4f}, max={max(normal):.4f}"
    )
    print(
        f"Вопросы без ответа: "
        f"min={min(trick):.4f}, max={max(trick):.4f}"
    )

    def evaluate(threshold: float) -> tuple[float, float, float, int, int]:
        accepted = sum(s >= threshold for s in normal)

        rejected = sum(s < threshold for s in trick)

        false_rejections = len(normal) - accepted
        false_acceptances = len(trick) - rejected

        recall = accepted / len(normal)
        specificity = rejected / len(trick)
        balanced_accuracy = (recall + specificity) / 2

        return (
            balanced_accuracy,
            recall,
            specificity,
            false_acceptances,
            false_rejections
        )

    thresholds = [0.80, 0.8359]

    candidates = sorted(set(normal + trick + [1.0]))

    best_threshold = None
    best_result = None

    for threshold in candidates:
        result = evaluate(threshold)

        if (
            best_result is None
            or (result[0], -result[3], threshold)
            > (best_result[0], -best_result[3], best_threshold)
        ):
            best_threshold = threshold
            best_result = result

    if best_threshold not in thresholds:
        thresholds.append(best_threshold)

    print("\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ")

    for threshold in thresholds:
        ba, recall, specificity, fp, fn = evaluate(threshold)

        accepted = sum(s >= threshold for s in normal)
        rejected = sum(s < threshold for s in trick)

        print(f"\nПорог: {threshold:.4f}")
        print(f"Правильные вопросы приняты: {accepted}/{len(normal)}")
        print(f"Вопросы без ответа отклонены: {rejected}/{len(trick)}")
        print(f"Ложные ответы: {fp}")
        print(f"Ложные отказы: {fn}")
        print(f"Recall: {recall:.2%}")
        print(f"Specificity: {specificity:.2%}")
        print(f"Balanced Accuracy: {ba:.2%}")

    print("\nЛУЧШИЙ ПОРОГ ПО РЕЗУЛЬТАТАМ ТЕСТА")
    print(f"Порог: {best_threshold:.4f}")
    print(f"Balanced Accuracy: {best_result[0]:.2%}")


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


def main() -> None:
    normal, hit1, hit5, mrr = run_normal()
    trick = run_trick()

    show_threshold(normal, trick)


def check_question_details():
    questions = [
        "Как восстановить забытый пароль от БАРС Офиса?",
        "Как создать новую учетную запись сотрудника в БАРС Офисе?",
        "Как прикрепить фотографию предмета мебели к записи?",
        "Как заказать новую мебель для подразделения?",
        "Как перенести мебель из одного кабинета в другой?",
        "Как рассчитать стоимость мебели по данным реестра?"
    ]

    print("\n" + "=" * 75)
    print("ПРОВЕРКА СПОРНЫХ ЗАПРОСОВ")
    print("=" * 75)

    for question in questions:
        print(f"\nВОПРОС: {question}")

        results = search(question, top_k=5)

        if not results:
            print("Ничего не найдено")
            continue

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.get('score', 0):.4f}")
            print(f"Источник: {result.get('source')}")
            print(f"Страница: {result.get('page')}")
            print(f"Тип: {result.get('type')}")
            print(f"Фрагмент: {result.get('text', '')[:1000]}")

if __name__ == "__main__":
    main()
    check_question_details()
    asyncio.run(check_contract())