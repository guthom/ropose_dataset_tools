from typing import Tuple, List, Dict
from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
from ropose_dataset_tools.DataClasses.BaseTypes.Vector3D import Vector3D
import numpy as np

class AxisAngles(BaseType):

    axis: Vector3D = None
    angle: float = None

    def __init__(self, axis: Vector3D, angle: float):
        self.axis = axis
        self.angle = angle
        super().__init__()

    def toList(self) -> List:
        return [self.axis.toList(), self.angle]

    @classmethod
    def fromQuat(cls, quat: 'Quaternion'):
        vector, angle = cls.AxisAngleFromQuat(quat)
        return cls(vector, angle)

    @classmethod
    def fromList(cls, list: List):
        vector = Vector3D.fromList(list[0])
        return cls(vector, list[1])

    @classmethod
    def fromDict(cls, dict: Dict):
        vector = Vector3D.fromList(dict["axis"])
        return cls(vector, float(dict["angle"]))

    def toString(self) -> str:
        return "[" + self.axis.toString() + str(self.angle) + "]"

    @staticmethod
    def AxisAngleFromQuat(quat: 'Quaternion') -> Tuple[Vector3D, float]:
        from math import acos, sqrt, pow

        angle = 2 * acos(quat.w)
        s = sqrt(1 - pow(quat.w, 2))

        #handle singularities
        if s < 0.001:
            x = quat.x
            y = quat.y
            z = quat.z
        else:
            x = quat.x / s
            y = quat.y / s
            z = quat.z / s

        return Vector3D(x=x, y=y, z=z), angle

