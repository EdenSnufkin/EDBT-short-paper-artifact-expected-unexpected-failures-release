from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class BaseunexpectedFailure(ABC):
    def __init__(self, unexpectedFailure: float = 0, beta : float = 1):
        self.unexpectedFailure = unexpectedFailure
        self.beta = beta
        self.name = "base"

    def reset(self,unexpectedFailure: float = 0) -> None:
        self.unexpectedFailure = unexpectedFailure

    def _get_unexpectedFailure(self) -> float :
        return self.unexpectedFailure
    
    def _set_unexpectedFailure(self, unexpectedFailure: float = 0) -> None:
        self.unexpectedFailure = unexpectedFailure
    
    def _get_wellbeing(self) -> float :
        return np.exp(-self.beta * self.unexpectedFailure)

    def _update_unexpectedFailure(self, unexpectedFailure: float, mastery) -> None :
        self.unexpectedFailure = max(0., unexpectedFailure)

    def _get_parameters(self) -> Dict[str,Any]:
        config = {
            "beta" : self.beta,
        }
        return config

    @abstractmethod
    def calculate_unexpectedFailure(self, difficulties , mastery) -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def _get_normalized_unexpectedFailure(self) -> float:
        pass

    @abstractmethod
    def _get_unexpectedFailure_reward(self) -> float :
        pass

class DifficultyBased(BaseunexpectedFailure):
    def __init__(self, unexpectedFailure: float = 0, beta: float = 0.05, lam: float = 0.1):
        super().__init__(unexpectedFailure, beta)
        self.lam = lam
        self.name = "difficulty"
        self.threshold = 4.0

    def calculate_unexpectedFailure(self, difficulties: List[float], mastery = None) -> Tuple[float, float]:
        unexpectedFailure_inst = self._get_unexpectedFailure_instant(difficulties,mastery)
        unexpectedFailure = max(0., self.unexpectedFailure + unexpectedFailure_inst)
        return unexpectedFailure, unexpectedFailure_inst
    
    def _get_normalized_unexpectedFailure(self) -> float:
        return max(1.0, self.unexpectedFailure / 20.0)
    
    def _get_unexpectedFailure_reward(self) -> float:
        return 1.0 if self.unexpectedFailure < self.threshold else 0.0

    def _get_parameters(self) -> Dict[str, Any]:
        config = super()._get_parameters()
        config.update({
            "lam" : self.lam
        })
        return config
    
    def _get_unexpectedFailure_instant(self, difficulties: list[float], mastery: float | None) -> float:
        if difficulties == []:
            unexpectedFailure_inst = - self.lam
        else:
            unexpectedFailure_inst = 1.0 - (sum(difficulties)/len(difficulties))
        return unexpectedFailure_inst

class LearningPotentialBased(BaseunexpectedFailure):
    def __init__(self, unexpectedFailure: float = 0., beta: float = 0.5, lam: float = 0.05, mastery = 0.4):
        super().__init__(unexpectedFailure,beta)
        self.lam = lam
        self.mastery = mastery
        self.name = "learningpotential"
        self.threshold = 2.0

    def calculate_unexpectedFailure(self, difficulties: list[float], mastery: float) -> Tuple[float, float]:
        unexpectedFailure_inst = self._get_unexpectedFailure_instant(difficulties,mastery)
        unexpectedFailure = max(0., self.unexpectedFailure + unexpectedFailure_inst)
        return unexpectedFailure, unexpectedFailure_inst

    def reset(self, unexpectedFailure: float = 0., mastery: float = 0.4):
        self.mastery = mastery
        self.unexpectedFailure = unexpectedFailure

    def _get_normalized_unexpectedFailure(self) -> float:
        return max(1.0, self.unexpectedFailure / 10.0)
    
    def _get_unexpectedFailure_reward(self) -> float:
        return 1.0 if self.unexpectedFailure < self.threshold else 0.0

    def _get_parameters(self) -> Dict[str, Any]:
        config = super()._get_parameters()
        config.update({
            "lam" : self.lam
        })
        return config
    
    def _update_unexpectedFailure(self, unexpectedFailure: float, mastery) -> None :
        self.unexpectedFailure = max(0., unexpectedFailure)
        self.mastery = mastery

    def _get_unexpectedFailure_instant(self, difficulties: list[float], mastery: float) -> float:
        if difficulties == []:
            unexpectedFailure_inst = - self.lam
        else:
            l = [max(0., mastery - x + 0.2) for x in difficulties]
            unexpectedFailure_inst = sum(l)/len(l)
        return unexpectedFailure_inst

class SmoothLearningPotentialBased(BaseunexpectedFailure):
    #NOT USED IN THE PAPER, JUST AN EXPERIMENTAL VARIANT OF LEARNING POTENTIAL BASED unexpectedFailure
    #DO NOT USE
    def __init__(self, unexpectedFailure: float = 0, beta: float = 0.5, lam: float = 0.05, mastery: float = 0.4, delta: float = 0, tau :float = 0):
        super().__init__(unexpectedFailure,beta)
        self.lam = lam
        self.mastery = mastery
        self.delta = delta
        self.tau = tau
        self.name = "smoothlearningpotential"

    def calculate_unexpectedFailure(self, difficulties: List[float], mastery: float) -> Tuple[float, float]:
        if difficulties == []:
            unexpectedFailure_inst = - self.lam
        else:
            l = [max(0., mastery - x + 0.2) for x in difficulties]
            unexpectedFailure_inst = sum(l)/len(l)

        unexpectedFailure = self.unexpectedFailure + unexpectedFailure_inst
        return unexpectedFailure, unexpectedFailure_inst      

    def reset(self, unexpectedFailure: float = 0, mastery: float = 0.4):
        self.mastery = mastery
        self.unexpectedFailure = unexpectedFailure

    def _get_normalized_unexpectedFailure(self) -> float:
        return max(1.0, self.unexpectedFailure / 5.0)
    
    def _get_unexpectedFailure_reward(self) -> float:
        return 1.0 if self.unexpectedFailure < 1.0 else 0.0
    
    def _get_parameters(self) -> Dict[str, Any]:
        config = super()._get_parameters()
        config.update({
            "lam" : self.lam,
            "delta" : self.delta,
            "tau" : self.tau,
        })
        return config
    
    def _update_unexpectedFailure(self, unexpectedFailure: float, mastery) -> None :
        self.unexpectedFailure = max(0., unexpectedFailure)
        self.mastery = mastery