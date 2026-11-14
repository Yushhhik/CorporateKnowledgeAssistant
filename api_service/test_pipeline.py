"""
Офлайн-тесты логики API (без Qdrant, модели эмбеддингов, Redis и Ollama).

Проверяют код проекта: определение системы, ранжирование, порог, формирование контекста,
промпт, обработку ответа LLM и контракт /webhook. Качество эмбеддингов они НЕ проверяют:
для этого нужен test_search.py с запущенным Qdrant.

Запуск (из папки api_service):
    python -m unittest test_pipeline -v
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGES_TMP = tempfile.mkdtemp(prefix="rag_images_")
os.environ["IMAGES_DIR"] = IMAGES_TMP


def _stub(name: str, **attrs) -> None:
    try:
        __import__(name)
        return
    except ImportError:
        pass

    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module

    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)


class _Rec:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeFastAPI:
    def __init__(self, *a, **kw):
        self.state = SimpleNamespace()

    def _deco(self, *a, **kw):
        return lambda fn: fn

    get = post = on_event = _deco

    def mount(self, *a, **kw):
        pass


class _FakeBaseModel:
    def __init__(self, **kw):
        self.__dict__.update(kw)


_stub("qdrant_client", QdrantClient=_Rec)
_stub(
    "qdrant_client.models",
    Distance=SimpleNamespace(COSINE="Cosine"),
    FieldCondition=_Rec,
    Filter=_Rec,
    MatchValue=_Rec,
    PayloadSchemaType=SimpleNamespace(KEYWORD="keyword"),
    PointStruct=_Rec,
    VectorParams=_Rec,
)
_stub("sentence_transformers", SentenceTransformer=_Rec)
_stub("openai", OpenAI=_Rec)
_stub("redis")
_stub("redis.asyncio", Redis=_Rec)
_stub("fastapi", FastAPI=_FakeFastAPI)
_stub("fastapi.staticfiles", StaticFiles=_Rec)
_stub("pydantic", BaseModel=_FakeBaseModel)

import llm_service
import main
import prompts
import qdrant_store
import search

NO_ANSWER = "Я не нашел точного ответа в документации. Переключить на оператора?"

IMG_NAME = "Реестр мебели_abcd1234.png"
FURNITURE = "Реестр мебели.pdf"


def _payload(index, text, *, type_="text", page=9, section="5 Инструкция", source=FURNITURE,
             system="bars", image_path=None):
    return {
        "type": type_, "source": source, "section": section, "page": page, "text": text,
        "image_path": image_path, "system": system, "chunk_index": index,
    }


class FakeQdrant:
    def __init__(self, scored, extra=()):
        self.scored = scored
        self.store = {}
        for _, p in scored:
            self.store[qdrant_store.chunk_point_id(p["source"], p["chunk_index"])] = p
        for p in extra:
            self.store[qdrant_store.chunk_point_id(p["source"], p["chunk_index"])] = p
        self.last_filter = None

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        self.last_filter = query_filter
        return SimpleNamespace(points=[SimpleNamespace(score=s, payload=p) for s, p in self.scored])

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        return [SimpleNamespace(payload=self.store[i]) for i in ids if i in self.store]

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        wanted = {c.key: c.match.value for c in scroll_filter.must}
        found = [p for p in self.store.values() if all(p.get(k) == v for k, v in wanted.items())]
        return [SimpleNamespace(payload=p) for p in found], None


def _furniture_client():
    main_chunk = _payload(10, "Нажать кнопку «Добавить». Заполнить поля: Модель, Подразделение, Кабинет.")
    prev_chunk = _payload(9, "Для добавления записи откройте реестр мебели.")
    next_chunk = _payload(11, "После заполнения нажать «Сохранить».")
    image = _payload(12, "Скриншот формы добавления.", type_="image",
                     image_path=f"images/{IMG_NAME}")
    other_doc = _payload(3, "Маршрут согласования создаётся в Garantexpress.",
                         source="1 доп док.pdf", page=2, system="garantexpress")
    client = FakeQdrant(
        [(0.90, main_chunk), (0.86, image), (0.80, other_doc)],
        extra=[prev_chunk, next_chunk],
    )
    return client


class SearchLogicTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(search, "embed_query", return_value=[0.0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def _use(self, client):
        patcher = mock.patch.object(search, "_client", client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_detect_system(self):
        cases = {
            "Как создать бизнес-процесс в SAP?": "sap",
            "Как настроить согласование в 1С?": "1c",
            "Как создать шаблон в Docsvision?": "docsvision",
            "Как создать маршрут в Garantexpress?": "garantexpress",
            "Как выглядит интерфейс БАРС-Офис?": "bars",
            "Как приготовить борщ?": None,
            "Какой у нас график 1сентября?": None,
        }
        for question, expected in cases.items():
            self.assertEqual(search.detect_system(question), expected, question)

    def test_system_filter_is_applied(self):
        client = _furniture_client()
        self._use(client)

        search.search("Как настроить согласование в 1С?")
        conds = {c.key: c.match.value for c in client.last_filter.must}
        self.assertEqual(conds, {"system": "1c"})

        search.search("Как добавить запись в реестр мебели?")
        self.assertIsNone(client.last_filter)

        search.search("Как выглядит интерфейс БАРС?", chunk_type="text")
        conds = {c.key: c.match.value for c in client.last_filter.must}
        self.assertEqual(conds, {"type": "text", "system": "bars"})

    def test_lexical_bonus_reorders_but_score_stays_raw(self):
        unrelated = _payload(1, "Совсем другой текст про печать документов", page=1)
        relevant = _payload(2, "График дежурства офис-менеджеров: понедельник пятница", page=3)
        self._use(FakeQdrant([(0.850, unrelated), (0.845, relevant)]))

        hits = search.search("график офис-менеджеров дежурства")

        self.assertEqual(hits[0]["page"], 3)
        self.assertEqual(hits[0]["score"], 0.845)
        self.assertEqual(hits[1]["score"], 0.85)

    def test_is_confident_uses_threshold(self):
        t = search.CONFIDENCE_THRESHOLD
        self.assertFalse(search.is_confident([]))
        self.assertFalse(search.is_confident([{"score": t - 0.0001}]))
        self.assertTrue(search.is_confident([{"score": t}]))

    def test_context_building(self):
        self._use(_furniture_client())

        context, payload = search._search_qdrant_sync("Как добавить запись в реестр мебели?")

        self.assertIsNotNone(context)
        self.assertTrue(context.startswith("[Источник: Реестр мебели, стр. 9]"))
        self.assertIn("откройте реестр мебели", context)
        self.assertIn("нажать «Сохранить»", context)
        self.assertNotIn("Garantexpress", context)
        self.assertLessEqual(len(context), search.MAX_CONTEXT_CHARS)

        self.assertEqual(payload["source"], FURNITURE)
        self.assertEqual(payload["page"], 9)
        self.assertEqual(payload["sources"], ["Источник: Реестр мебели, стр. 9"])
        self.assertEqual(payload["images"], [f"images/{IMG_NAME}"])

    def test_below_threshold_gives_none(self):
        low = _payload(1, "Что-то случайное", page=1)
        self._use(FakeQdrant([(search.CONFIDENCE_THRESHOLD - 0.01, low)]))

        self.assertEqual(search._search_qdrant_sync("Как приготовить борщ?"), (None, None))

    def test_empty_query(self):
        self._use(_furniture_client())
        self.assertEqual(search.search("   "), [])

    def test_search_qdrant_error_contract(self):
        class Broken:
            def query_points(self, *a, **kw):
                raise ConnectionError("qdrant down")

        self._use(Broken())

        import asyncio
        self.assertEqual(asyncio.run(search.search_qdrant("вопрос")), (None, None))
        with self.assertRaises(ConnectionError):
            asyncio.run(search.search_qdrant("вопрос", raise_errors=True))


class PromptTests(unittest.TestCase):
    def test_build_prompt_does_not_crash_and_has_context(self):
        msgs = prompts.build_prompt("Контекст {с скобками}", [], "Вопрос?")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Контекст {с скобками}", msgs[0]["content"])
        self.assertIn(NO_ANSWER, msgs[0]["content"])
        self.assertEqual(msgs[-1], {"role": "user", "content": "Вопрос?"})

    def test_empty_context_forces_refusal(self):
        for ctx in ("", "   ", None):
            content = prompts.build_prompt(ctx, [], "Как приготовить борщ?")[0]["content"]
            self.assertIn(NO_ANSWER, content)
            self.assertNotIn("Вот контекст", content)

    def test_history_is_chronological(self):
        history = ["Бот: ответ 2", "Пользователь: вопрос 2", "Бот: ответ 1", "Пользователь: вопрос 1"]
        msgs = prompts.build_prompt("ctx", history, "вопрос 3")[1:]
        self.assertEqual([m["content"] for m in msgs],
                         ["вопрос 1", "ответ 1", "вопрос 2", "ответ 2", "вопрос 3"])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant", "user"])

    def test_no_answer_constant_is_verbatim(self):
        self.assertEqual(prompts.NO_ANSWER, NO_ANSWER)


class LLMServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_clean_answer(self):
        self.assertEqual(llm_service.clean_answer("Шаг 1.\n\nИсточник: X.pdf, стр. 99"), "Шаг 1.")
        self.assertEqual(llm_service.clean_answer(None), NO_ANSWER)
        self.assertEqual(llm_service.clean_answer("  "), NO_ANSWER)
        self.assertEqual(
            llm_service.clean_answer("Я не нашел точного ответа в корпоративной документации. Возможно..."),
            NO_ANSWER,
        )

    async def test_ask_llm_passes_context_and_question(self):
        captured = {}

        def create(**kw):
            captured.update(kw)
            msg = SimpleNamespace(content="Нажмите «Добавить».\nИсточник: выдуманный.pdf, стр. 1")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with mock.patch.object(llm_service, "_get_client", return_value=client):
            answer = await llm_service.ask_llm("КОНТЕКСТ-123", [], "Как добавить?")

        self.assertEqual(answer, "Нажмите «Добавить».")
        self.assertIn("КОНТЕКСТ-123", captured["messages"][0]["content"])
        self.assertEqual(captured["messages"][-1]["content"], "Как добавить?")
        self.assertEqual(captured["model"], llm_service.OLLAMA_MODEL)
        self.assertLessEqual(captured["temperature"], 0.2)

    async def test_ask_llm_wraps_errors(self):
        def create(**kw):
            raise ConnectionError("ollama down")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with mock.patch.object(llm_service, "_get_client", return_value=client):
            with self.assertRaises(llm_service.LLMError):
                await llm_service.ask_llm("ctx", [], "q")


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def lpush(self, key, value):
        self.data.setdefault(key, []).insert(0, value)

    async def ltrim(self, key, start, end):
        self.data[key] = self.data[key][start:end + 1]

    async def expire(self, key, seconds):
        pass

    async def lrange(self, key, start, end):
        return list(self.data.get(key, []))


class BrokenRedis:
    async def lpush(self, *a):
        raise ConnectionError("redis down")

    ltrim = expire = lrange = lpush


PAYLOAD = {
    "source": FURNITURE, "page": 9, "image_path": None, "images": [f"images/{IMG_NAME}"],
    "sources": ["Источник: Реестр мебели, стр. 9"],
}


class WebhookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.redis = FakeRedis()
        for patcher in (mock.patch.object(main, "r", self.redis),):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def _ask(self, text="Вопрос", user_id=1):
        return await main.receive_message(main.TelegramMessage(user_id=user_id, text=text))

    async def test_answer_with_sources_and_image(self):
        (Path(IMAGES_TMP) / IMG_NAME).write_bytes(b"png")

        with mock.patch.object(main, "search_qdrant", mock.AsyncMock(return_value=("ctx", PAYLOAD))), \
             mock.patch.object(main, "ask_llm", mock.AsyncMock(return_value="Нажмите «Добавить».")) as llm:
            res = await self._ask("Как добавить запись?")

        llm.assert_awaited_once_with("ctx", [], "Как добавить запись?")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["reply"], "Нажмите «Добавить».\n\nИсточник: Реестр мебели, стр. 9")
        self.assertFalse(res["suggest_operator"])
        self.assertEqual(res["image_path"], f"/images/{IMG_NAME}")
        self.assertEqual(self.redis.data["chat:1"][1], "Пользователь: Как добавить запись?")

    async def test_missing_image_file_is_not_returned(self):
        (Path(IMAGES_TMP) / IMG_NAME).unlink(missing_ok=True)

        with mock.patch.object(main, "search_qdrant", mock.AsyncMock(return_value=("ctx", PAYLOAD))), \
             mock.patch.object(main, "ask_llm", mock.AsyncMock(return_value="Ответ")):
            res = await self._ask()

        self.assertIsNone(res["image_path"])

    async def test_nothing_found_returns_verbatim_refusal_without_llm(self):
        with mock.patch.object(main, "search_qdrant", mock.AsyncMock(return_value=(None, None))), \
             mock.patch.object(main, "ask_llm", mock.AsyncMock()) as llm:
            res = await self._ask("Как приготовить борщ?")

        llm.assert_not_awaited()
        self.assertEqual(res["reply"], NO_ANSWER)
        self.assertTrue(res["suggest_operator"])
        self.assertIsNone(res["image_path"])

    async def test_llm_refusal_has_no_sources(self):
        with mock.patch.object(main, "search_qdrant", mock.AsyncMock(return_value=("ctx", PAYLOAD))), \
             mock.patch.object(main, "ask_llm", mock.AsyncMock(return_value=NO_ANSWER)):
            res = await self._ask("Как оформить загранпаспорт?")

        self.assertEqual(res["reply"], NO_ANSWER)
        self.assertTrue(res["suggest_operator"])
        self.assertNotIn("Источник", res["reply"])

    async def test_llm_down_returns_error_without_source(self):
        with mock.patch.object(main, "search_qdrant", mock.AsyncMock(return_value=("ctx", PAYLOAD))), \
             mock.patch.object(main, "ask_llm", mock.AsyncMock(side_effect=llm_service.LLMError("down"))):
            res = await self._ask()

        self.assertEqual(res["status"], "error")
        self.assertNotIn("Источник", res["reply"])

    async def test_qdrant_down_is_error_not_refusal(self):
        with mock.patch.object(main, "search_qdrant", mock.AsyncMock(side_effect=ConnectionError("qdrant"))):
            res = await self._ask()

        self.assertEqual(res["status"], "error")
        self.assertNotEqual(res["reply"], NO_ANSWER)

    async def test_redis_down_does_not_break_answers(self):
        with mock.patch.object(main, "r", BrokenRedis()), \
             mock.patch.object(main, "search_qdrant", mock.AsyncMock(return_value=("ctx", PAYLOAD))), \
             mock.patch.object(main, "ask_llm", mock.AsyncMock(return_value="Ответ")):
            res = await self._ask()
            history = await main.history_endpoint(1)

        self.assertEqual(res["status"], "success")
        self.assertEqual(history, {"history": ""})

    async def test_history_is_passed_to_llm_and_endpoint_is_chronological(self):
        with mock.patch.object(main, "search_qdrant", mock.AsyncMock(return_value=("ctx", PAYLOAD))), \
             mock.patch.object(main, "ask_llm", mock.AsyncMock(return_value="Ответ")) as llm:
            await self._ask("первый", user_id=7)
            await self._ask("второй", user_id=7)

        second_history = llm.await_args_list[1].args[1]
        self.assertEqual(second_history[1], "Пользователь: первый")

        text = (await main.history_endpoint(7))["history"].splitlines()
        self.assertEqual(text[0], "Пользователь: первый")
        self.assertTrue(text[1].startswith("Бот: Ответ"))


if __name__ == "__main__":
    unittest.main(verbosity=2)