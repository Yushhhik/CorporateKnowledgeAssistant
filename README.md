# Корпоративный Telegram-бот БАРС Груп

Хакатон ИННОГЛОБАЛХАК 2026. Кейс: ИИ-ассистент первой линии поддержки на базе RAG.

# Участники
Голубева Наталья
Евдокимова Ксения
Бердюгина Анастасия
Афанасьева Екатерина
Чурина Яна

## О проекте

Telegram-бот, который отвечает на вопросы сотрудников по внутренней документации БАРС Груп (БАРС-Офис, бизнес-процессы, реестры). Бот ищет ответ в базе знаний через векторный поиск (Qdrant), собирает промпт с учетом истории диалога и отправляет его в локальную LLM (Ollama / llama3.2). Если LLM недоступна или тормозит — срабатывает fallback на заготовленные ответы.

## Архитектура

┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│ Telegram │────▶│ bot_service │────▶│ api_service │
│ (клиент) │◀────│ (FastAPI) │◀────│ (LLM+RAG) │
└─────────────┘ └──────┬───────┘ └──────┬───────┘
│ │
▼ ▼
┌──────────┐ ┌──────────┐
│ Redis │ │ Qdrant │
│ (история)│ │ (база │
└──────────┘ │ знаний) │
└──────────┘
│
▼
┌──────────┐
│ Ollama │
│(llama3.2)│
└──────────┘


Компоненты:
- `bot_service` — Telegram-бот на python-telegram-bot, принимает сообщения и пересылает в API
- `api_service` — FastAPI-сервер, оркестрирует RAG-пайплайн
- `Qdrant` — векторная БД, хранит чанки документов
- `Redis` — хранит историю диалогов по пользователям
- `Ollama` — локальная LLM (llama3.2), генерирует ответы

## Стек

- Python 3.11+
- FastAPI
- python-telegram-bot
- Qdrant Client
- Redis (aioredis)
- Ollama (llama3.2)
- httpx, python-dotenv
- Docker / docker-compose

## Структура проекта
CorporateKnowledgeAssistant/
├── api_service/
│ ├── main.py # FastAPI-сервер, роуты
│ ├── prompts.py # System Prompt + функция build_prompt
│ ├── llm_service.py # Вызов Ollama + fallback-заглушка
│ ├── rag_service.py # Работа с Qdrant (пишет Человек 3)
│ ├── redis_service.py # Работа с Redis (пишет Человек 2)
│ ├── requirements.txt
│ └── test_traps.py # Тесты на защиту от галлюцинаций
├── bot_service/
│ ├── bot.py # Telegram-бот
│ ├── requirements.txt
│ └── Dockerfile
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md


## Как запустить

### 1. Установить и запустить Ollama

Скачать с [ollama.com](https://ollama.com/download) и установить. После запуска в трее появится иконка ламы 🦙.

Скачать модель:
```bash
ollama pull llama3.2
