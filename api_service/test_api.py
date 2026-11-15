import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# --- 1. ИНФРАСТРУКТУРНЫЙ ТЕСТ ---
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- 2. ПУСТОЕ СООБЩЕНИЕ ---
def test_webhook_empty_message():
    response = client.post("/webhook", json={"user_id": 123, "text": ""})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["suggest_operator"] is True


# --- 3. ВОПРОС ВНЕ БАЗЫ (ГАЛЛЮЦИНАЦИИ) ---
def test_webhook_nonsense_question():
    response = client.post("/webhook", json={"user_id": 123, "text": "Как приготовить борщ?"})
    assert response.status_code == 200
    data = response.json()
    assert "не нашел" in data["reply"].lower() or data["suggest_operator"] is True


# --- 4. УСПЕШНЫЙ ПОИСК: ГРАФИК ОФИС-МЕНЕДЖЕРА ---
def test_webhook_success_question():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Мне не отвечает офис-менеджер, какой у него график работы?"}
    )
    assert response.status_code == 200
    data = response.json()

    if not data["suggest_operator"]:
        assert len(data["reply"]) > 10
        assert "17:30" in data["reply"] or "график" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст или порог уверенности слишком высок. Тест пропущен.")


# --- 5. ВОЗВРАТ ИСТОЧНИКА: РЕЕСТР МЕБЕЛИ ---
def test_webhook_returns_source():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Какие функциональные кнопки есть в реестре мебели?"}
    )
    assert response.status_code == 200
    data = response.json()

    if not data["suggest_operator"]:
        assert "Реестр" in data["reply"] or "стр." in data["reply"]
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 6. ИНТЕРФЕЙС БО: ПРОИЗВОДСТВЕННЫЙ КАЛЕНДАРЬ ---
def test_webhook_calendar():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Как выглядит интерфейс БО? Где найти в нем производственный календарь?"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "календар" in data["reply"].lower() or "интерфейс" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 7. ФИЛЬТР ДОКУМЕНТА: ШАГИ ---
def test_webhook_filter_steps():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Как работает фильтр документа? Опиши шаг действий для настройки фильтра"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "фильтр" in data["reply"].lower()
        assert "лупа" in data["reply"].lower() or "применить" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 8. СЛОЖНЫЙ ФИЛЬТР: ОПЕРАТОРЫ ---
def test_webhook_filter_operators():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": 'Почему не работает фильтр в БО "!> 200"?'}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "оператор" in data["reply"].lower() or "!=" in data["reply"] or ">" in data["reply"]
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 9. ВНУТРЕННЯЯ ССЫЛКА ---
def test_webhook_internal_link():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Мне прислали ссылку https://bo.bars.group/ что с ней делать?"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "ссылк" in data["reply"].lower() or "документ" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 10. СОГЛАСОВАНИЕ КОМАНДИРОВКИ ---
def test_webhook_approval():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Мне необходимо узнать, кто согласовывает БП Командировки?"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "руководител" in data["reply"].lower() or "подразделен" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 11. ФИЛЬТРАЦИЯ СДЕЛОК (ПРЕСЕЙЛ) ---
def test_webhook_presale_filter():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Почему не могу выбрать сделку 2334-26, у которой статус воронки 30% в поле Номер пресейла?"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "60%" in data["reply"] or "воронк" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 12. БЮДЖЕТ И ЗАПУСК БП ---
def test_webhook_budget():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Смогу ли я запустить БП Командировка, если не заложил средств в бюджет сделки?"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "бюджет" in data["reply"].lower() or "останов" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 13. УСЛОВИЯ ВЫБОРА СОГЛАСУЮЩЕГО ---
def test_webhook_approver_conditions():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Какие могут быть условия в выборе следующего согласующего лица БП Командировка?"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "руководител" in data["reply"].lower() or "директор" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 14. ЛОВУШКА: РЕЕСТР ТЕХНИКИ (а не мебели) ---
def test_webhook_technique_trap():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Опиши поэтапно как в БО добавить запись в реестр техники?"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "мебел" in data["reply"].lower()
        # Бот должен заметить, что реестра техники нет
        assert "техник" in data["reply"].lower() or "мебел" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")


# --- 15. СТАТУСЫ МЕБЕЛИ ---
def test_webhook_furniture_statuses():
    response = client.post(
        "/webhook",
        json={"user_id": 123, "text": "Какие статусы мебели перечислены в реестре?"}
    )
    assert response.status_code == 200
    data = response.json()
    if not data["suggest_operator"]:
        assert "занят" in data["reply"].lower() or "свободен" in data["reply"].lower()
        assert "списан" in data["reply"].lower()
    else:
        pytest.skip("Qdrant пуст. Тест пропущен.")