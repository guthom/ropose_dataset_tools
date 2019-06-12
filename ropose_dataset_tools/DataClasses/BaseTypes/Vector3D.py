from typing import List, Dict
from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
import math
import numpy as np

class Vector3D(BaseType):

    x: float = None
    y: float = None
    z: float = None

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

        super().__init__()

    @classmethod
    def fromList(cls, list: List[float]):
        return cls(list[0], list[1], list[2])

    @classmethod
    def fromDict(cls, dict: Dict):
        return cls(float(dict["x"]), float(dict["y"]), float(dict["z"]))

    def toString(self) -> str:
        return "[" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + "]"

    def toList(self) -> List[float]:
        return [self.x, self.y, self.z]

    def toNp(self) -> np.array:
        return np.array([[self.x], [self.y], [self.z]])

    def toNp4(self) -> np.array:
        return np.array([[self.x], [self.y], [self.z], [1.0]])

    def Normalized(self) -> 'Vector3D':
        d = math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2) + math.pow(self.z, 2))
        x = self.x / d
        y = self.y / d
        z = self.z / d

        return Vector3D(x, y, z)


