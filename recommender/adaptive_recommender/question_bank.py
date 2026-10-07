"""The pool of questions the recommender chooses from.

A question is described by four fields only: its position in the bank (``index``), the identifier it has in the
study logs (``item_id``), its ``topic`` and its estimated ``difficulty`` in [0, 1]. Question texts are not needed
by the recommender and are not part of this package.

All random choices are made with a ``random.Random`` instance passed by the caller, so a whole session is
reproducible from one seed.
"""
from __future__ import annotations

import csv
import random
import warnings
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Question:
    index: int          # position in the bank; the key used by the learner model
    item_id: int        # identifier used in the study logs (``attempts.item_id``)
    topic: str
    difficulty: float   # estimated difficulty in [0, 1]


class QuestionBank:
    """A list of questions, indexed by topic and difficulty level."""

    def __init__(self, questions: Iterable[Question]):
        self.questions: list[Question] = list(questions)
        if [q.index for q in self.questions] != list(range(len(self.questions))):
            raise ValueError("question indices must be 0..n-1 in order")
        self.topics: list[str] = sorted({q.topic for q in self.questions})
        groups: dict[str, dict[float, list[Question]]] = {}
        for q in self.questions:
            groups.setdefault(q.topic, {}).setdefault(float(q.difficulty), []).append(q)
        self._groups = groups
        self._levels = {topic: tuple(sorted(by_level)) for topic, by_level in groups.items()}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_csv(cls, path: str | Path) -> "QuestionBank":
        """Read ``question_index, item_id, topic, difficulty`` (extra columns are ignored)."""
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda r: int(r["question_index"]))
        return cls(Question(int(r["question_index"]), int(r["item_id"]), r["topic"], float(r["difficulty"])) for r in rows)

    def __len__(self) -> int:
        return len(self.questions)

    def by_item_id(self, item_id: int) -> Question:
        return next(q for q in self.questions if q.item_id == item_id)

    # ------------------------------------------------------------------ selection primitives
    def random_batch(self, k: int, topic: str, exclude: Iterable[int] = (), rng: random.Random | None = None) -> list[Question]:
        """``k`` distinct random questions of ``topic`` whose index is not in ``exclude``.

        When fewer than ``k`` unseen questions remain, the batch is completed with random questions of the topic
        (seen ones included), as the study platform did.
        """
        rng = rng or random.Random()
        excluded = set(exclude)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if topic not in self.topics:
            raise ValueError(f"unknown topic {topic!r}; available: {self.topics}")
        pool = [q for q in self.questions if q.topic == topic and q.index not in excluded]
        if k > len(pool):
            warnings.warn(f"only {len(pool)} unseen questions left for {topic!r}; adding previously seen ones")
            same_topic = [q for q in self.questions if q.topic == topic]
            pool.extend(rng.sample(same_topic, k=k - len(pool)))
        return rng.sample(pool, k=k)

    def neighbour(self, question: Question, direction: int, exclude: Iterable[int] = (), rng: random.Random | None = None) -> Question:
        """A random question at the closest difficulty level below (``direction=-1``) or above (``+1``) ``question``.

        Levels that only contain excluded questions are skipped. When no level is left, ``question`` itself is
        returned (with a warning), so that the hill climber simply cannot move in that direction.
        """
        rng = rng or random.Random()
        excluded = exclude if isinstance(exclude, set) else set(exclude)
        levels, groups = self._levels.get(question.topic, ()), self._groups.get(question.topic, {})
        difficulty = float(question.difficulty)
        pos = bisect_left(levels, difficulty) - 1 if direction < 0 else bisect_right(levels, difficulty)
        while 0 <= pos < len(levels):
            available = [c for c in groups[levels[pos]] if c.index not in excluded]
            if available:
                return rng.choice(available)
            pos += direction
        warnings.warn(f"no unseen question {'below' if direction < 0 else 'above'} difficulty {difficulty} for {question.topic!r}")
        return question

    def initial_questions(self, k: int, topic: str, exclude: Iterable[int] = (), rng: random.Random | None = None) -> list[Question]:
        """``k`` distinct questions spread evenly over the difficulty range of ``topic`` (used for the pretest)."""
        rng = rng or random.Random()
        excluded = set(exclude)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if topic not in self.topics:
            raise ValueError(f"unknown topic {topic!r}; available: {self.topics}")
        available = [q for q in self.questions if q.topic == topic and q.index not in excluded]
        if k > len(available):
            raise ValueError(f"requested {k} initial questions for {topic!r}, only {len(available)} available")
        lo, hi = min(q.difficulty for q in available), max(q.difficulty for q in available)
        targets = [(lo + hi) / 2.0] if k == 1 else [lo + i * (hi - lo) / (k - 1) for i in range(k)]
        remaining, selected = available.copy(), []
        for target in targets:
            nearest = min(abs(q.difficulty - target) for q in remaining)
            choice = rng.choice([q for q in remaining if abs(q.difficulty - target) == nearest])
            selected.append(choice)
            remaining.remove(choice)
        return selected
