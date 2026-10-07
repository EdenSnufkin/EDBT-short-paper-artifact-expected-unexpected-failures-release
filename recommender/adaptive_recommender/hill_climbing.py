"""Multi-objective hill climbing recommender (the AdUp batch recommender used in the study).

``recommend`` builds a batch of ``k`` questions in three steps:

1. **Random restarts.** ``restarts`` random batches of unseen questions are drawn.
2. **Hill climbing.** Each batch is improved by local moves: replacing one question by a random question of the
   closest lower or higher difficulty level. A move is accepted when its batch is on the Pareto front of the
   neighbours and is not dominated by the current batch; the climb stops when the current batch dominates every
   Pareto-optimal neighbour (or after ``max_steps`` moves).
3. **Selection.** The climbed batches are reduced to their Pareto front. One of them is returned: the one with
   the best ``negative_gap`` for the ``MOO`` variant, a random one for the other variants.

Variants (the *Pareto objectives* are the ones optimised by the climb):

========  ============================================  ==========================
variant   Pareto objectives                             final choice
========  ============================================  ==========================
``MOO``   aptitude, expected_performance                best negative_gap (the paper's HC)
``MOEG``  expected_performance, negative_gap            random Pareto batch
``MOAG``  aptitude, negative_gap                        random Pareto batch
``MOAE``  aptitude, expected_performance                random Pareto batch
========  ============================================  ==========================
"""
from __future__ import annotations

import random

from .learner import Learner
from .objectives import Candidate, dominates, non_dominated
from .question_bank import Question, QuestionBank

VARIANTS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    # name: (objective used to pick the final batch or None for a random pick, Pareto objectives)
    "MOO": ("negative_gap", ("aptitude", "expected_performance")),
    "MOEG": (None, ("expected_performance", "negative_gap")),
    "MOAG": (None, ("aptitude", "negative_gap")),
    "MOAE": (None, ("aptitude", "expected_performance")),
}


class HillClimbingRecommender:
    def __init__(self, bank: QuestionBank, learner: Learner, batch_size: int = 3, restarts: int = 20,
                 variant: str = "MOO", max_steps: int = 500, rng: random.Random | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; use one of {list(VARIANTS)}")
        self.bank, self.learner = bank, learner
        self.batch_size, self.restarts, self.max_steps = batch_size, restarts, max_steps
        self.variant = variant
        self.final_objective, self.pareto_objectives = VARIANTS[variant]
        self.rng = rng or random.Random()

    # ------------------------------------------------------------------ public API
    def recommend(self, topic: str) -> list[Question]:
        climbed = []
        for _ in range(self.restarts):
            start = self.bank.random_batch(self.batch_size, topic, exclude=list(self.learner.cleared), rng=self.rng)
            climbed.append(self._climb(Candidate.evaluate(start, self.learner)))
        front = non_dominated(climbed, self.pareto_objectives)
        if self.final_objective is not None:
            best = max(front, key=lambda c: c.scores[self.final_objective])
        else:
            best = self.rng.sample(front, 1)[0]
        return best.batch

    # ------------------------------------------------------------------ internals
    def _neighbours(self, current: Candidate) -> list[Candidate]:
        """Every batch obtained by moving one question to the next lower / higher difficulty level."""
        blocked = set(self.learner.cleared) | {q.index for q in current.batch}
        neighbours = []
        for question in current.batch:
            for direction in (-1, +1):
                replacement = self.bank.neighbour(question, direction, exclude=blocked, rng=self.rng)
                batch = current.batch.copy()
                batch.remove(question)
                batch.append(replacement)
                neighbours.append(Candidate.evaluate(batch, self.learner))
        return neighbours

    def _climb(self, current: Candidate) -> Candidate:
        for _ in range(self.max_steps):
            front = non_dominated(self._neighbours(current), self.pareto_objectives)
            if all(dominates(current, n, self.pareto_objectives) for n in front):
                return current                                   # local optimum
            current = next(n for n in front if not dominates(current, n, self.pareto_objectives))
        return current
