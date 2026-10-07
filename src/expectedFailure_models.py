from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class BaseExpectedFailure(ABC):
    def __init__(self, expectedFailure: float = 0, beta : float = 1):
        self.expectedFailure = expectedFailure
        self.beta = beta
        self.name = "base"

    def reset(self,expectedFailure: float = 0) -> None:
        self.expectedFailure = expectedFailure

    def _get_expectedFailure(self) -> float :
        return self.expectedFailure

    def _set_expectedFailure(self, expectedFailure: float = 0) -> None:
        self.expectedFailure = expectedFailure

    def _get_wellbeing(self) -> float :
        return np.exp(-self.beta * self.expectedFailure)

    def _update_expectedFailure(self, expectedFailure: float, mastery) -> None :
        self.expectedFailure = max(0., expectedFailure)

    def _get_parameters(self) -> Dict[str,Any]:
        config = {
            "beta" : self.beta,
        }
        return config

    @abstractmethod
    def calculate_expectedFailure(self, difficulties , mastery) -> Tuple[float, float]:
        pass

    @abstractmethod
    def _get_normalized_expectedFailure(self) -> float:
        pass

    @abstractmethod
    def _get_expectedFailure_reward(self) -> float :
        pass

class DifficultyBased(BaseExpectedFailure):
    def __init__(self, expectedFailure: float = 0, beta: float = 0.05, lam: float = 0.1):
        super().__init__(expectedFailure, beta)
        self.lam = lam
        self.name = "difficulty"
        self.threshold = 4.0

    def calculate_expectedFailure(self, difficulties: List[float], mastery: Optional[float] = None) -> Tuple[float, float]:
        expectedFailure_inst = self._get_expectedFailure_instant(difficulties, mastery)
        expectedFailure = max(0., self.expectedFailure + expectedFailure_inst)
        return expectedFailure, expectedFailure_inst

    def _get_normalized_expectedFailure(self) -> float:
        return max(1.0, self.expectedFailure / 20.0)

    def _get_expectedFailure_reward(self) -> float:
        return 1.0 if self.expectedFailure < self.threshold else 0.0

    def _get_parameters(self) -> Dict[str, Any]:
        config = super()._get_parameters()
        config.update({
            "lam" : self.lam
        })
        return config

    def _get_expectedFailure_instant(self, difficulties: list[float], mastery: float | None) -> float:
        # Expected failures are the mirror of unexpected ones: severity grows with how far
        # the test difficulty sits *above* mastery (d(q) - m_t), not below it.
        if difficulties == [] or mastery is None:
            expectedFailure_inst = - self.lam
        else:
            gaps = [max(0., d - mastery) for d in difficulties]
            expectedFailure_inst = sum(gaps) / len(gaps)
        return expectedFailure_inst

class LearningPotentialBased(BaseExpectedFailure):
    def __init__(self, expectedFailure: float = 0., beta: float = 0.5, lam: float = 0.05, mastery = 0.4):
        super().__init__(expectedFailure,beta)
        self.lam = lam
        self.mastery = mastery
        self.name = "learningpotential"
        self.threshold = 2.0

    def calculate_expectedFailure(self, difficulties: list[float], mastery: float) -> Tuple[float, float]:
        expectedFailure_inst = self._get_expectedFailure_instant(difficulties,mastery)
        expectedFailure = max(0., self.expectedFailure + expectedFailure_inst)
        return expectedFailure, expectedFailure_inst

    def reset(self, expectedFailure: float = 0., mastery: float = 0.4):
        self.mastery = mastery
        self.expectedFailure = expectedFailure

    def _get_normalized_expectedFailure(self) -> float:
        return max(1.0, self.expectedFailure / 10.0)

    def _get_expectedFailure_reward(self) -> float:
        return 1.0 if self.expectedFailure < self.threshold else 0.0

    def _get_parameters(self) -> Dict[str, Any]:
        config = super()._get_parameters()
        config.update({
            "lam" : self.lam
        })
        return config

    def _update_expectedFailure(self, expectedFailure: float, mastery) -> None :
        self.expectedFailure = max(0., expectedFailure)
        self.mastery = mastery

    def _get_expectedFailure_instant(self, difficulties: list[float], mastery: float) -> float:
        # Mirrors LearningPotentialBased's unexpected-failure formula (max(0, mastery - x + 0.2))
        # with the difficulty/mastery difference flipped: severity comes from the test being
        # harder than the learner's mastery, the same +0.2 buffer softening the exact boundary.
        if difficulties == []:
            expectedFailure_inst = - self.lam
        else:
            l = [max(0., x - mastery + 0.2) for x in difficulties]
            expectedFailure_inst = sum(l)/len(l)
        return expectedFailure_inst
