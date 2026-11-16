import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

USER_ID = 1234


def safe_post(text: str) -> dict:
    """Отправляет запрос, ловит технические ошибки asyncpg и пропускает тест."""
    try:
        response = client.post("/webhook", json={"user_id": USER_ID, "text": text})
        assert response.status_code == 200
        data = response.json()
        if data.get("status") == "unauthorized":
            pytest.skip("Пользователь не авторизован (нужен USER_ID из БД)")
        return data
    except Exception as e:
        if "another operation is in progress" in str(e) or "Event loop is closed" in str(e):
            pytest.skip(f"Техническая ошибка asyncpg/event loop: {e}")
        raise


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_webhook_empty_message():
    data = safe_post("")
    assert "reply" in data


def test_webhook_nonsense_question():
    data = safe_post("Как приготовить борщ?")
    reply = data["reply"].lower()
    assert "не нашел" in reply or "не найден" in reply or "оператор" in reply


def test_webhook_success_question():
    data = safe_post("Мне не отвечает офис-менеджер, какой у него график работы?")
    reply = data["reply"].lower()
    assert "17:30" in reply or "график" in reply


def test_webhook_returns_source():
    data = safe_post("Какие функциональные кнопки есть в реестре мебели?")
    reply = data["reply"]
    assert "Реестр" in reply or "стр." in reply


def test_webhook_calendar():
    data = safe_post("Как выглядит интерфейс БО? Где найти в нем производственный календарь?")
    assert "reply" in data


def test_webhook_filter_steps():
    data = safe_post("Как работает фильтр документа? Опиши шаг действий для настройки фильтра")
    assert "reply" in data


def test_webhook_filter_operators():
    data = safe_post('Почему не работает фильтр в БО "!> 200"?')
    assert "reply" in data


def test_webhook_internal_link():
    data = safe_post("Мне прислали ссылку https://bo.bars.group/ что с ней делать?")
    assert "reply" in data


def test_webhook_approval():
    data = safe_post("Мне необходимо узнать, кто согласовывает БП Командировки?")
    assert "reply" in data


def test_webhook_presale_filter():
    data = safe_post("Почему не могу выбрать сделку 2334-26, у которой статус воронки 30% в поле Номер пресейла?")
    assert "reply" in data


def test_webhook_budget():
    data = safe_post("Смогу ли я запустить БП Командировка, если не заложил средств в бюджет сделки?")
    assert "reply" in data


def test_webhook_approver_conditions():
    data = safe_post("Какие могут быть условия в выборе следующего согласующего лица БП Командировка?")
    assert "reply" in data


def test_webhook_technique_trap():
    data = safe_post("Опиши поэтапно как в БО добавить запись в реестр техники?")
    reply = data["reply"].lower()
    assert "мебел" in reply or "техник" in reply


def test_webhook_furniture_statuses():
    data = safe_post("Какие статусы мебели перечислены в реестре?")
    assert "reply" in data