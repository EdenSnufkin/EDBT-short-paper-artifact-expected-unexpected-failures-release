"""Expected / unexpected failure scores (Sec. 3 of the paper).

For a **failed** answer to a question of difficulty ``d`` when the learner's estimated mastery is ``m``:

    UFS = max(0, m - d + delta)         unexpected failure severity (tolerance ``delta``)
    EFS = max(0, d - m)                 expected failure severity (strict)

Both are 0 for a correct answer. They are accumulated over the answers of a session:

    AUFS <- max(0, AUFS + UFS - lambda * [answer is correct])
    AEFS <- max(0, AEFS + EFS - lambda * [answer is correct])

and ``TOFT`` is the share of steps at which AUFS exceeds a threshold ``tau``.

The study platform logged AUFS only (``delta = 0.20``, ``lambda = 0.20``) and used it to trigger an extra survey when
it reached ``tau = 0.50``; AEFS was computed afterwards, with the same accumulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def ufs(mastery: float, difficulty: float, correct: bool, delta: float = 0.20) -> float:
    return 0.0 if correct else max(0.0, mastery - difficulty + delta)


def efs(mastery: float, difficulty: float, correct: bool) -> float:
    return 0.0 if correct else max(0.0, difficulty - mastery)


def toft(aufs_per_step: Sequence[float], tau: float = 0.50) -> float:
    """Time Over Failure Threshold: share of steps with AUFS strictly above ``tau``."""
    return sum(a > tau for a in aufs_per_step) / len(aufs_per_step) if aufs_per_step else 0.0


@dataclass
class FailureScores:
    delta: float = 0.20
    lam: float = 0.20
    aufs: float = 0.0
    aefs: float = 0.0

    def update(self, mastery: float, difficulty: float, correct: bool) -> tuple[float, float]:
        """Account for one answer given the mastery *before* it; returns the instantaneous (UFS, EFS)."""
        u, e = ufs(mastery, difficulty, correct, self.delta), efs(mastery, difficulty, correct)
        decay = self.lam if correct else 0.0
        self.aufs = max(0.0, self.aufs + u - decay)
        self.aefs = max(0.0, self.aefs + e - decay)
        return u, e
