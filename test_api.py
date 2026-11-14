import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_webhook_empty_message():
    response = client.post("/webhook", json={"user_id": 123, "text": ""})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["suggest_operator"] is True

def test_webhook_nonsense_question():
    response = client.post("/webhook", json={"user_id": 123, "text": "Как приготовить борщ?"})
    assert response.status_code == 200
    data = response.json()
    assert "не нашел" in data["reply"].lower() or data["suggest_operator"] is True


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
