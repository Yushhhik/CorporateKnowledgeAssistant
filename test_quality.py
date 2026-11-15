"""Тесты против ложных отказов «нет информации в документации».

Запуск:  python -m unittest test_quality -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_service
import prompts
import search
from prompts import NO_ANSWER


def hit(text, score, section="Док > Раздел", source="Док.pdf"):
    return {"text": text, "score": score, "section": section, "source": source,
            "page": 1, "type": "text", "image_path": None, "system": "bars"}


T = search.CONFIDENCE_THRESHOLD


class SoftZone(unittest.TestCase):
    def test_above_threshold_always_accepted(self):
        self.assertTrue(search.accepts(hit("что угодно", T), set()))

    def test_soft_zone_needs_question_words(self):
        q = search._key_words("Как убрать сохранённый фильтр документа?")
        good = hit("Чтобы убрать фильтр документа, нажмите крестик у фильтра", T - 0.03)
        other = hit("Печать QR-кодов для мебели", T - 0.03)
        self.assertTrue(search.accepts(good, q))
        self.assertFalse(search.accepts(other, q))

    def test_below_soft_zone_rejected(self):
        q = search._key_words("Как настроить фильтр документа?")
        self.assertFalse(search.accepts(hit("настроить фильтр документа", T - 0.2), q))

    def test_single_keyword_question(self):
        q = search._key_words("Почему не работает фильтр?")
        self.assertTrue(search.accepts(hit("Фильтр применяется кнопкой", T - 0.02), q))

    def test_unrelated_question_stays_rejected(self):
        q = search._key_words("Как приготовить борщ?")
        self.assertFalse(search.accepts(hit("Реестр мебели: добавить запись", T - 0.02), q))

    def test_placeholder_images_not_used(self):
        h = hit("Скриншот или схема из раздела «X». Рядом в документе: ...", 0.99)
        self.assertFalse(search.accepts(h, set()))

    def test_select_hits_limit_and_order(self):
        hits = [hit(f"фильтр документа {i}", 0.95 - i * 0.01) for i in range(8)]
        picked = search.select_hits(hits, "фильтр документа")
        self.assertEqual(len(picked), search.CONTEXT_CHUNKS)
        self.assertEqual(picked[0]["text"], "фильтр документа 0")


class HistoryDoesNotPoison(unittest.TestCase):
    def test_refusals_removed_with_their_question(self):
        history = [
            f"Бот: {NO_ANSWER}", "Пользователь: борщ?",
            "Бот: Нажмите «Добавить».\n\nИсточник: X.pdf, стр. 1", "Пользователь: как добавить?",
        ]
        msgs = prompts.build_prompt("ctx", history, "а удалить?")[1:]
        self.assertEqual([m["content"] for m in msgs], ["как добавить?", "Нажмите «Добавить».", "а удалить?"])

    def test_history_is_trimmed(self):
        history = []
        for i in range(10):
            history = [f"Бот: ответ {i}", f"Пользователь: вопрос {i}"] + history
        msgs = prompts.build_prompt("ctx", history, "новый")[1:]
        self.assertLessEqual(len(msgs), prompts.HISTORY_MESSAGES + 1)
        self.assertEqual(msgs[-1]["content"], "новый")

    def test_long_messages_truncated(self):
        msgs = prompts.build_prompt("ctx", ["Бот: " + "я" * 5000], "q")[1:]
        self.assertLessEqual(len(msgs[0]["content"]), prompts.HISTORY_ITEM_CHARS)

    def test_prompt_allows_partial_answers(self):
        content = prompts.build_prompt("ctx", [], "q")[0]["content"]
        self.assertIn("часть", content)


class CleanAnswer(unittest.TestCase):
    def test_long_useful_answer_with_hedge_is_kept(self):
        text = "1. Нажмите «Добавить».\n2. Заполните поля. " * 8 + "Я не нашел точного ответа про печать."
        self.assertGreater(len(text), 250)
        self.assertNotEqual(llm_service.clean_answer(text), NO_ANSWER)

    def test_short_refusal_normalized(self):
        self.assertEqual(llm_service.clean_answer("__NO_ANSWER__"), NO_ANSWER)
        self.assertEqual(llm_service.clean_answer("Я не нашел точного ответа."), NO_ANSWER)

    def test_source_lines_removed_not_added(self):
        self.assertEqual(llm_service.clean_answer("Шаг 1.\nИсточник: выдумка.pdf"), "Шаг 1.")


if __name__ == "__main__":
    unittest.main()
