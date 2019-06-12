from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict

class BaseType(ABC):

    @abstractmethod
    def toList(self) -> List[float]:
        raise Exception("Not Implemented!")

    @abstractmethod
    def toString(self) -> str:
        raise Exception("Not Implemented!")

    def toNp(self) -> np.array:
        return np.array(self.toList())

    @classmethod
    @abstractmethod
    def fromList(cls, list: List[float]):
        raise Exception("Not Implemented!")

    @classmethod
    @abstractmethod
    def fromDict(cls, dict: Dict):
        raise Exception("Not Implemented!")

