"""Learner model: the estimated mastery that drives the recommendations.

Mastery is an *estimate*, kept per topic in [0, 1]. It follows the "consecutive correct answers" rule of the study:

* mastery rises to the difficulty of a question once the learner has answered ``ncc_window`` (default 2)
  consecutive questions **of that exact difficulty** correctly;
* a failure erases the evidence collected at the failed difficulty and at every harder difficulty;
* mastery is never lowered.

The learner also remembers which questions were cleared and which were failed, because the recommendation
objectives are defined relative to them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .question_bank import Question


@dataclass
class Learner:
    learner_id: str
    topics: list[str]
    mastery: dict[str, float]
    ncc_window: int = 2
    ncc_cache: dict[str, dict[float, list[bool]]] = field(default_factory=dict)   # recent outcomes per topic and difficulty
    failed: dict[int, Question] = field(default_factory=dict)    # question index -> question, failed and not cleared since
    cleared: dict[int, Question] = field(default_factory=dict)   # question index -> question, answered correctly
    responses: list[tuple[int, bool]] = field(default_factory=list)   # (question index, correct) in answer order
    mastery_history: list[dict[str, float]] = field(default_factory=list)

    @classmethod
    def new(cls, learner_id: str, topics: list[str], initial_mastery: float = 0.2, ncc_window: int = 2) -> "Learner":
        mastery = {topic: float(initial_mastery) for topic in topics}
        return cls(learner_id=learner_id, topics=list(topics), mastery=mastery, ncc_window=ncc_window,
                   ncc_cache={topic: {} for topic in topics}, mastery_history=[mastery.copy()])

    # ------------------------------------------------------------------ bookkeeping
    def record(self, question: Question, correct: bool) -> None:
        """Remember the answer (does not change mastery)."""
        self.responses.append((question.index, bool(correct)))
        if correct:
            self.failed.pop(question.index, None)
            self.cleared[question.index] = question
        else:
            self.failed[question.index] = question

    def update_mastery(self, question: Question, correct: bool) -> None:
        """Apply the consecutive-correct-answers rule to one answer."""
        topic, difficulty = question.topic, float(question.difficulty)
        if topic in self.mastery:
            cache = self.ncc_cache.setdefault(topic, {})
            recent = cache.setdefault(difficulty, [])
            recent.append(bool(correct))
            del recent[:-self.ncc_window]
            if correct and len(recent) >= self.ncc_window and all(recent):
                self.mastery[topic] = min(1.0, max(0.0, max(self.mastery[topic], difficulty)))
            elif not correct:
                for level in list(cache):
                    if level >= difficulty:
                        cache[level] = []
        self.mastery_history.append(self.mastery.copy())

    # ------------------------------------------------------------------ views used by the objectives
    def failed_difficulties(self) -> list[float]:
        return [q.difficulty for q in self.failed.values()]

    def cleared_difficulties(self) -> list[float]:
        return [q.difficulty for q in self.cleared.values()]
