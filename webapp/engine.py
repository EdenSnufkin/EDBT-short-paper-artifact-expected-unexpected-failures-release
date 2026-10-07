# webapp/engine.py
"""Per-participant learning engine: a thin layer over ``adaptive_recommender.LearningSession``, configured from settings."""
from __future__ import annotations

import random
from dataclasses import dataclass

from adaptive_recommender import AnswerRecord, BatchRecord, LearningSession, Question, QuestionBank

import settings


@dataclass
class BatchOutcome:
    """What the platform logs after a batch."""
    record: BatchRecord
    answers: list[AnswerRecord]
    reward: float | None          # bandit reward (mastery progression), None for plain hill climbing


class StudyEngine:
    """Owns the recommender and learner state of one participant."""

    def __init__(self, bank: QuestionBank, student_id: str, seed: int, recommendation_method: str | None = None,
                 hillclimbing_algo: str | None = None, mab_algo: str | None = None):
        self.seed = seed
        self.recommendation_method = recommendation_method or settings.RECOMMENDATION_METHOD
        self.hillclimbing_algo = hillclimbing_algo or settings.HILLCLIMBING_ALGO
        self.mab_algo = mab_algo or settings.MAB_ALGO
        self.session = LearningSession(
            bank, method=self.recommendation_method, variant=self.hillclimbing_algo, policy=self.mab_algo,
            arms=tuple(settings.MAB_ARMS), topic=settings.TOPIC, batch_size=settings.BATCH_SIZE, restarts=settings.HC_TIMES,
            n_batches=settings.N_BATCHES, stop_mastery=settings.MASTERY_STOP_THRESHOLD, pretest_size=settings.PRETEST_N,
            initial_mastery=settings.INIT_MASTERY, ncc_window=settings.NCC_WINDOW, delta=settings.DELTA, lam=settings.LAMBDA,
            seed=seed, learner_id=student_id)

    @staticmethod
    def new_seed() -> int:
        return settings.RANDOM_SEED if settings.RANDOM_SEED is not None else random.SystemRandom().randrange(2**31)

    @property
    def mastery(self) -> float:
        return float(self.session.mastery)

    @property
    def aufs(self) -> float:
        return float(self.session.scores.aufs)

    @property
    def aefs(self) -> float:
        return float(self.session.scores.aefs)

    def pretest_questions(self) -> list[Question]:
        return self.session.pretest_questions()

    def submit_pretest(self, questions: list[Question], correct: list[bool]) -> list[AnswerRecord]:
        return self.session.submit_pretest(questions, correct)

    def recommend_batch(self) -> tuple[list[Question], str, int | None]:
        """Returns (questions, algorithm used, bandit arm index or None). Safe to run in a worker thread: it only touches
        this engine, and the answers of the batch are submitted after it returns."""
        batch = self.session.next_batch()
        arm, variant = self.session.pending
        return batch, variant, arm

    def submit_batch(self, correct: list[bool]) -> BatchOutcome:
        n_before = len(self.session.answers)
        record = self.session.submit_batch(correct)
        reward = None
        if record.arm is not None:
            reward = record.mastery_after - record.mastery_before
        return BatchOutcome(record, self.session.answers[n_before:], reward)

    def is_learning_done(self) -> bool:
        return self.session.is_done
