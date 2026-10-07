"""Multi-armed bandit over the hill-climbing variants (the MAB recommender of the study).

Each *arm* is a hill-climbing variant (``MOO``, ``MOAE``, ``MOAG``, ``MOEG``). Before every batch the bandit
policy picks an arm, the arm recommends the batch, and after the learner has answered the policy is rewarded with
the **mastery progression** of the batch (``mastery_after - mastery_before``). For Thompson sampling, which needs a
binary signal, the reward is 1 when mastery increased and 0 otherwise.

Policies: ``thompson`` (the one used in the study), ``egreedy`` (epsilon = 0.1), ``softmax`` (temperature = 0.05),
``ucb`` and ``random``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .hill_climbing import HillClimbingRecommender
from .learner import Learner
from .question_bank import Question, QuestionBank


# ---------------------------------------------------------------------- policies
class RandomPolicy:
    def __init__(self, n_arms: int, rng: random.Random, np_rng: np.random.RandomState):
        self.n_arms, self.rng, self.np_rng = n_arms, rng, np_rng

    def select(self) -> int:
        return self.rng.randrange(self.n_arms)

    def update(self, arm: int, reward: float) -> None:
        pass


def _argmax_random_tiebreak(values: np.ndarray, np_rng: np.random.RandomState) -> int:
    candidates = np.flatnonzero(values == np.max(values))
    return int(np_rng.choice(candidates))


class EpsilonGreedyPolicy(RandomPolicy):
    def __init__(self, n_arms, rng, np_rng, epsilon: float = 0.1):
        super().__init__(n_arms, rng, np_rng)
        self.epsilon = epsilon
        self.counts, self.values = np.zeros(n_arms), np.zeros(n_arms)

    def select(self) -> int:
        if self.np_rng.rand() < self.epsilon:
            return int(self.np_rng.randint(0, self.n_arms))
        return _argmax_random_tiebreak(self.values, self.np_rng)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward


class SoftmaxPolicy(RandomPolicy):
    def __init__(self, n_arms, rng, np_rng, temperature: float = 0.05):
        super().__init__(n_arms, rng, np_rng)
        self.temperature = temperature
        self.means, self.trials = np.zeros(n_arms), np.zeros(n_arms)

    def select(self) -> int:
        e_x = np.exp(self.means / self.temperature)
        return int(self.np_rng.choice(range(self.n_arms), p=e_x / e_x.sum()))

    def update(self, arm: int, reward: float) -> None:
        self.means[arm] = (self.means[arm] * self.trials[arm] + reward) / (self.trials[arm] + 1)
        self.trials[arm] += 1


class ThompsonPolicy(RandomPolicy):
    def __init__(self, n_arms, rng, np_rng):
        super().__init__(n_arms, rng, np_rng)
        self.successes, self.failures = np.zeros(n_arms), np.zeros(n_arms)

    def select(self) -> int:
        return int(np.argmax(self.np_rng.beta(self.successes + 1, self.failures + 1)))

    def update(self, arm: int, reward: float) -> None:
        if reward > 0:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1


class UcbPolicy(RandomPolicy):
    def __init__(self, n_arms, rng, np_rng):
        super().__init__(n_arms, rng, np_rng)
        self.counts, self.values, self.total = np.zeros(n_arms), np.zeros(n_arms), 0

    def select(self) -> int:
        ucb = self.values + np.sqrt(2 * np.log(self.total + 1) / (self.counts + 1e-5))
        return _argmax_random_tiebreak(ucb, self.np_rng)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.total += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward


POLICIES = {"random": RandomPolicy, "egreedy": EpsilonGreedyPolicy, "softmax": SoftmaxPolicy, "thompson": ThompsonPolicy, "ucb": UcbPolicy}


# ---------------------------------------------------------------------- recommender
@dataclass
class Arm:
    arm_id: int
    variant: str
    recommender: HillClimbingRecommender
    times_used: int = 0
    total_progression: float = 0.0


class BanditRecommender:
    def __init__(self, bank: QuestionBank, learner: Learner, batch_size: int = 3, restarts: int = 20,
                 policy: str = "thompson", arms: tuple[str, ...] = ("MOO", "MOAE", "MOAG", "MOEG"),
                 rng: random.Random | None = None, np_rng: np.random.RandomState | None = None):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; use one of {list(POLICIES)}")
        rng = rng or random.Random()
        np_rng = np_rng or np.random.RandomState()
        self.policy_name = policy
        self.arms = [Arm(i, variant, HillClimbingRecommender(bank, learner, batch_size, restarts, variant, rng=rng))
                     for i, variant in enumerate(arms)]
        self.policy = POLICIES[policy](len(self.arms), rng, np_rng)

    def recommend(self, topic: str) -> tuple[list[Question], int]:
        """Pick an arm and return its batch together with the arm index."""
        arm = self.policy.select()
        self.arms[arm].times_used += 1
        return self.arms[arm].recommender.recommend(topic), arm

    def update(self, arm: int, mastery_before: float, mastery_after: float) -> float:
        """Reward the policy with the mastery progression of the batch; returns the reward given to the policy."""
        progression = mastery_after - mastery_before
        self.arms[arm].total_progression += progression
        reward = (1.0 if mastery_after > mastery_before else 0.0) if self.policy_name == "thompson" else progression
        self.policy.update(arm, reward)
        return reward
