from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
from typing import List, Dict

class Vector2D(BaseType):

    x: float = None
    y: float = None

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

        super().__init__()

    @classmethod
    def fromList(cls, list: List[float]):
        return cls(list[0], list[1])

    @classmethod
    def fromDict(cls, dict: Dict):
        return cls(float(dict["x"]), float(dict["y"]))

    def toString(self) -> str:
        return "[" + str(self.x) + str(self.y) + "]"

    def toList(self) -> List[float]:
        return [self.x, self.y]
