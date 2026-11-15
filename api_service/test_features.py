"""Тесты дополнительных возможностей (features.py): уточняющие вопросы, кэш, аналитика, отзывы.

Запуск:  python -m unittest test_features -v
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features


class FakeRedis:
    def __init__(self):
        self.kv, self.h, self.l = {}, {}, {}

    async def get(self, k): return self.kv.get(k)
    async def set(self, k, v, ex=None): self.kv[k] = v
    async def hincrby(self, k, f, n): self.h.setdefault(k, {}); self.h[k][f] = int(self.h[k].get(f, 0)) + n
    async def hincrbyfloat(self, k, f, n): self.h.setdefault(k, {}); self.h[k][f] = float(self.h[k].get(f, 0)) + n
    async def hgetall(self, k): return {a: str(b) for a, b in self.h.get(k, {}).items()}
    async def lpush(self, k, v): self.l.setdefault(k, []).insert(0, v)
    async def ltrim(self, k, a, b): self.l[k] = self.l.get(k, [])[a:b + 1]
    async def lrange(self, k, a, b): return self.l.get(k, [])[a:b + 1 if b >= 0 else None]


class BrokenRedis:
    def __getattr__(self, name):
        async def boom(*a, **kw): raise ConnectionError("down")
        return boom


run = asyncio.run
HISTORY = ["Бот: Фильтр настраивается так.", "Пользователь: Как настроить фильтр документа?"]


class Contextualize(unittest.TestCase):
    def test_followup_gets_previous_topic(self):
        q = features.contextualize("а как его удалить?", HISTORY)
        self.assertIn("фильтр", q.lower())
        self.assertTrue(q.endswith("а как его удалить?"))

    def test_standalone_question_unchanged(self):
        q = "Где находится производственный календарь в системе?"
        self.assertEqual(features.contextualize(q, HISTORY), q)

    def test_no_history(self):
        self.assertEqual(features.contextualize("а как его удалить?", []), "а как его удалить?")

    def test_two_word_question_is_followup(self):
        self.assertIn("фильтр", features.contextualize("Удалить?", HISTORY).lower())


class Cache(unittest.TestCase):
    def test_key_ignores_case_and_punctuation(self):
        self.assertEqual(features.cache_key("Как найти меню?"), features.cache_key("  как   найти меню "))

    def test_roundtrip(self):
        r = FakeRedis()
        run(features.cache_set(r, "Вопрос?", {"reply": "ок"}))
        self.assertEqual(run(features.cache_get(r, "вопрос")), {"reply": "ок"})

    def test_broken_redis_is_silent(self):
        r = BrokenRedis()
        self.assertIsNone(run(features.cache_get(r, "q")))
        run(features.cache_set(r, "q", {}))
        run(features.track(r, "found", 1.0))
        self.assertFalse(run(features.save_feedback(r, {"rating": "up"})))
        self.assertEqual(run(features.analytics(r)), {"available": False})


class Related(unittest.TestCase):
    def test_distinct_sections_only(self):
        best = {"section": "Док > A", "score": 0.9}
        hits = [best, {"section": "Док > A", "score": 0.9}, {"section": "Док > Фильтры", "score": 0.88},
                {"section": "стр. 3", "score": 0.88}, {"section": "Док > Далёкое", "score": 0.5}]
        self.assertEqual(features.related_titles(hits, best, 0.84), ["Фильтры"])


class Analytics(unittest.TestCase):
    def test_summary(self):
        r = FakeRedis()
        run(features.track(r, "found", 2.0, source="a.pdf"))
        run(features.track(r, "cached", 0.1, source="a.pdf"))
        run(features.track(r, "not_found", 1.0, question="Как приготовить борщ?"))
        run(features.track(r, "not_found", 1.0, question="как приготовить борщ"))
        run(features.save_feedback(r, {"rating": "👍", "question": "q", "answer": "a"}))
        run(features.save_feedback(r, {"rating": "down"}))
        a = run(features.analytics(r))
        self.assertEqual(a["questions"], 4)
        self.assertEqual(a["found"], 2)
        self.assertEqual(a["not_found"], 2)
        self.assertEqual(a["answer_rate"], 0.5)
        self.assertEqual(a["feedback"], {"up": 1, "down": 1, "satisfaction": 0.5})
        self.assertEqual(a["top_sources"], [("a.pdf", 2)])
        self.assertEqual(a["knowledge_gaps"][0], ("как приготовить борщ", 2))


if __name__ == "__main__":
    unittest.main()
