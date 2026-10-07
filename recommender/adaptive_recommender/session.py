"""One learner's adaptive session, in the order of the study.

1. **Pretest**: ``pretest_size`` questions spread evenly over the difficulty range; the answers feed the learner
   model like any other answer.
2. **Learning loop**: before each batch the recommender (hill climbing or bandit) proposes ``batch_size``
   questions; the learner answers them; mastery and failure scores are updated question by question.
3. **Stop** when ``n_batches`` batches have been done or mastery reaches ``stop_mastery``.

The session never decides *how* answers are obtained: the caller passes the correctness of each answer, so the
same code serves a real participant (a web application) or a simulated learner (see ``simulation.py``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .bandit import BanditRecommender
from .failure_scores import FailureScores
from .hill_climbing import HillClimbingRecommender
from .learner import Learner
from .question_bank import Question, QuestionBank


@dataclass
class AnswerRecord:
    question: Question
    correct: bool
    mastery_before: float
    mastery_after: float
    ufs: float
    efs: float
    aufs: float            # accumulated scores right after this answer
    aefs: float
    phase: str             # "pretest" or "learning"
    batch_index: int | None


@dataclass
class BatchRecord:
    batch_index: int
    questions: list[Question]
    arm: int | None        # index of the bandit arm, None for plain hill climbing
    variant: str           # hill-climbing variant that produced the batch
    mastery_before: float
    mastery_after: float
    aufs_after: float
    aefs_after: float


class LearningSession:
    def __init__(self, bank: QuestionBank, *, method: str = "hillclimbing", variant: str = "MOO", policy: str = "thompson",
                 arms: tuple[str, ...] = ("MOO", "MOAE", "MOAG", "MOEG"), topic: str | None = None, batch_size: int = 3,
                 restarts: int = 20, n_batches: int = 15, stop_mastery: float = 0.9, pretest_size: int = 5,
                 initial_mastery: float = 0.2, ncc_window: int = 2, delta: float = 0.20, lam: float = 0.20,
                 seed: int | None = None, learner_id: str = "learner"):
        if method not in ("hillclimbing", "mab"):
            raise ValueError("method must be 'hillclimbing' or 'mab'")
        self.bank = bank
        self.topic = topic or (bank.topics[0] if len(bank.topics) == 1 else None)
        if self.topic is None:
            raise ValueError(f"the bank has several topics {bank.topics}; pass topic=")
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.method, self.variant = method, variant
        self.batch_size, self.n_batches, self.stop_mastery, self.pretest_size = batch_size, n_batches, stop_mastery, pretest_size
        self.learner = Learner.new(learner_id, bank.topics, initial_mastery, ncc_window)
        self.scores = FailureScores(delta=delta, lam=lam)
        if method == "mab":
            self.recommender = BanditRecommender(bank, self.learner, batch_size, restarts, policy, arms, self.rng, self.np_rng)
        else:
            self.recommender = HillClimbingRecommender(bank, self.learner, batch_size, restarts, variant, rng=self.rng)
        self.answers: list[AnswerRecord] = []
        self.batches: list[BatchRecord] = []
        self._pending: tuple[list[Question], int | None, str, float] | None = None

    # ------------------------------------------------------------------ state
    @property
    def mastery(self) -> float:
        return self.learner.mastery[self.topic]

    @property
    def is_done(self) -> bool:
        return len(self.batches) >= self.n_batches or self.mastery >= self.stop_mastery

    # ------------------------------------------------------------------ one answer
    def record_answer(self, question: Question, correct: bool, phase: str = "learning", batch_index: int | None = None) -> AnswerRecord:
        """Process one answer: update the learner model and the failure scores (also used to replay logged answers)."""
        before = self.mastery
        self.learner.record(question, correct)
        u, e = self.scores.update(before, question.difficulty, correct)      # scores use the mastery *before* the answer
        self.learner.update_mastery(question, correct)
        record = AnswerRecord(question, bool(correct), before, self.mastery, u, e, self.scores.aufs, self.scores.aefs, phase, batch_index)
        self.answers.append(record)
        return record

    # ------------------------------------------------------------------ pretest
    def pretest_questions(self) -> list[Question]:
        return self.bank.initial_questions(self.pretest_size, self.topic, rng=self.rng)

    def submit_pretest(self, questions: Sequence[Question], correct: Sequence[bool]) -> list[AnswerRecord]:
        return [self.record_answer(q, c, "pretest", None) for q, c in zip(questions, correct)]

    # ------------------------------------------------------------------ learning loop
    def next_batch(self) -> list[Question]:
        """Recommend the next batch (call ``submit_batch`` with the learner's answers afterwards)."""
        if self.is_done:
            raise RuntimeError("the session is finished")
        if self._pending is not None:
            raise RuntimeError("submit the answers of the pending batch first")
        if self.method == "mab":
            batch, arm = self.recommender.recommend(self.topic)
            variant = self.recommender.arms[arm].variant
        else:
            batch, arm, variant = self.recommender.recommend(self.topic), None, self.variant
        self._pending = (batch, arm, variant, self.mastery)
        return batch

    def submit_batch(self, correct: Sequence[bool]) -> BatchRecord:
        if self._pending is None:
            raise RuntimeError("call next_batch() first")
        batch, arm, variant, mastery_before = self._pending
        if len(correct) != len(batch):
            raise ValueError(f"expected {len(batch)} answers, got {len(correct)}")
        index = len(self.batches)
        for question, ok in zip(batch, correct):
            self.record_answer(question, ok, "learning", index)
        if arm is not None:
            self.recommender.update(arm, mastery_before, self.mastery)
        record = BatchRecord(index, batch, arm, variant, mastery_before, self.mastery, self.scores.aufs, self.scores.aefs)
        self.batches.append(record)
        self._pending = None
        return record
