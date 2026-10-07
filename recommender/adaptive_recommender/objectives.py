"""The three objectives of the recommender and Pareto dominance between candidate batches.

A *batch* is a list of ``k`` questions. Each objective is to be **maximised**:

``aptitude``
    average of ``difficulty - mastery`` over the batch: progression potential, higher means harder than the
    learner's current level.
``expected_performance``
    minus the average distance between the batch's difficulties and the difficulties of the questions the learner
    has already cleared: similarity with what the learner is known to succeed at (``nan`` before any success).
``negative_gap``
    minus the average distance between the batch's difficulties and the difficulties of the questions the learner
    has failed: closeness to the learner's known weaknesses (``nan`` before any failure).

An objective that is undefined (``nan``) neither helps nor hurts a batch in a dominance comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .learner import Learner
from .question_bank import Question

OBJECTIVES = ("aptitude", "expected_performance", "negative_gap")


def aptitude(batch: Sequence[Question], learner: Learner) -> float:
    if not batch:
        return -math.inf
    return sum(q.difficulty - learner.mastery[q.topic] for q in batch) / len(batch)


def _negative_mean_distance(batch: Sequence[Question], reference: list[float]) -> float:
    if not reference or not batch:
        return math.nan
    total = sum(sum(abs(q.difficulty - d) for d in reference) / len(reference) for q in batch)
    return -(total / len(batch))


def expected_performance(batch: Sequence[Question], learner: Learner) -> float:
    return _negative_mean_distance(batch, learner.cleared_difficulties())


def negative_gap(batch: Sequence[Question], learner: Learner) -> float:
    return _negative_mean_distance(batch, learner.failed_difficulties())


@dataclass
class Candidate:
    """A batch together with the value of every objective."""
    batch: list[Question]
    scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def evaluate(cls, batch: list[Question], learner: Learner) -> "Candidate":
        return cls(batch, {"aptitude": aptitude(batch, learner),
                           "expected_performance": expected_performance(batch, learner),
                           "negative_gap": negative_gap(batch, learner)})


def dominates(a: Candidate, b: Candidate, objectives: Sequence[str]) -> bool:
    """``a`` dominates ``b`` when it is not worse on any objective and strictly better on at least one."""
    strictly_better = False
    for name in objectives:
        x, y = a.scores[name], b.scores[name]
        if x < y:
            return False
        if x > y:
            strictly_better = True
    return strictly_better


def non_dominated(candidates: Sequence[Candidate], objectives: Sequence[str]) -> list[Candidate]:
    """The Pareto front: candidates that no other candidate dominates."""
    return [c for c in candidates
            if not any(other is not c and dominates(other, c, objectives) for other in candidates)]
