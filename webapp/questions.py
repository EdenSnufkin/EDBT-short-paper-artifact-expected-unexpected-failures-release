# webapp/questions.py
"""Question content shown to participants, and the question bank the recommender works on.

The recommender only needs ``(index, item_id, topic, difficulty)`` (``adaptive_recommender.QuestionBank``); this module
adds what is needed to *display and score* a question: text, options and the correct answer.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from adaptive_recommender import Question, QuestionBank


@dataclass(frozen=True)
class ContentQuestion:
    """Render-relevant fields of one question, looked up by ``item_id``."""

    item_id: int
    text: str
    options: list[str] | None          # None for free-response questions
    correct_answer_index: int | None   # index into options, when options is not None
    correct_answer_text: str | None    # ground-truth answer, when options is None


def _options(raw) -> list[str] | None:
    if isinstance(raw, str):                       # the study files store the list as its Python repr
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    return [str(o) for o in raw] if isinstance(raw, list) and raw else None


def load_questions(path: str | Path, topic: str) -> tuple[QuestionBank, dict[int, ContentQuestion]]:
    """Read a question file and return the bank for the recommender and the display content by ``item_id``.

    The file is a JSON list (or an object with a ``questions`` list); each question has ``id``, ``question``,
    ``difficulty`` in [0, 1] and either ``options`` + ``correct_answer`` (index of the right option) or ``answer`` (free
    text). All questions are assigned to ``topic``. Questions are indexed in file order.
    """
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["questions"] if isinstance(payload, dict) else payload

    bank_questions, content = [], {}
    for index, row in enumerate(rows):
        item_id = int(row.get("id", index))
        if item_id in content:
            raise ValueError(f"duplicate question id {item_id}")
        bank_questions.append(Question(index, item_id, topic, float(row["difficulty"])))
        options = _options(row.get("options"))
        if options is not None:
            content[item_id] = ContentQuestion(item_id, str(row["question"]), options, int(row["correct_answer"]), None)
        else:
            content[item_id] = ContentQuestion(item_id, str(row["question"]), None, None, str(row.get("answer", "")))
    return QuestionBank(bank_questions), content


def is_correct_answer(content: ContentQuestion, chosen_answer: str | None) -> bool:
    """Score a participant's raw answer (option index as a string, or free text)."""
    if content.options is not None:
        try:
            return int(chosen_answer) == content.correct_answer_index
        except (TypeError, ValueError):
            return False
    truth, given = (content.correct_answer_text or "").strip(), (chosen_answer or "").strip()
    try:
        return float(truth) == float(given)
    except ValueError:
        return truth.casefold() == given.casefold()


def answer_display_text(content: ContentQuestion, answer: str) -> str:
    """A participant-facing answer, with the option letter for multiple-choice questions."""
    if content.options is None:
        return str(answer)
    try:
        index = int(answer)
    except (TypeError, ValueError):
        return str(answer)
    if not 0 <= index < len(content.options):
        return str(answer)
    return f"{chr(ord('A') + index)}. {content.options[index]}"
