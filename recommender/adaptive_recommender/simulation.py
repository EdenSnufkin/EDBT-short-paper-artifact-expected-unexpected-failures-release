"""A simple simulated learner, to try the recommender without participants.

This is **not** the response model of the study (the study used real participants). The learner has a latent
``ability``; the probability of a correct answer is a logistic function of ``ability - difficulty``. With
``learning_rate > 0`` the ability grows a little after every correct answer.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .question_bank import Question
from .session import LearningSession


@dataclass
class SimulatedLearner:
    ability: float = 0.6
    slope: float = 10.0
    learning_rate: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def p_correct(self, question: Question) -> float:
        return 1.0 / (1.0 + math.exp(-self.slope * (self.ability - question.difficulty)))

    def answer(self, question: Question) -> bool:
        correct = self.rng.random() < self.p_correct(question)
        if correct:
            self.ability = min(1.0, self.ability + self.learning_rate)
        return correct


def run_session(session: LearningSession, learner: SimulatedLearner) -> LearningSession:
    """Play a whole session (pretest + learning loop) with a simulated learner."""
    pretest = session.pretest_questions()
    session.submit_pretest(pretest, [learner.answer(q) for q in pretest])
    while not session.is_done:
        batch = session.next_batch()
        session.submit_batch([learner.answer(q) for q in batch])
    return session
