from typing import List, Dict, Tuple
from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
import math
import numpy as np

class EulerAngles(BaseType):
    roll: float = None #X-Axis, gamma
    pitch: float = None #Y-Axis, beta
    yaw: float = None #Z-Axis, alpha

    order: str = None

    def __init__(self, roll: float, pitch: float, yaw: float, order="xyz"):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.order = order

        super().__init__()

    def GetRPY(self) -> Tuple[float, float, float]:
        return self.roll, self.pitch, self.yaw

    @classmethod
    def fromList(cls, list: List[float], order: str= "xyz"):
        if list.__len__() != 3:
            raise Exception("List size is not correct! Should be 3 instead of : " + str(list.__len__()))
        return cls(list[0], list[1], list[2], order)

    @classmethod
    def fromRotM(cls, rotM: 'RotationMatrix', order: str="xyz"):
        roll, pitch, yaw = cls.RPYFromRotM(rotM, order=order)
        return cls(roll=roll, pitch=pitch, yaw=yaw, order=order)

    @classmethod
    def fromDict(cls, dict: Dict):
        return cls(float(dict["x"]), float(dict["y"]), float(dict["z"]))

    def toString(self) -> str:
        return "[" + str(self.roll) + ", " + str(self.pitch) + ", " + str(self.yaw) + "]"

    def toList(self) -> List[float]:
        return [self.roll, self.pitch, self.yaw]

    @staticmethod
    def RPYFromRotM(rotM: 'RotationMatrix', order="xyz") -> Tuple[float, float, float]:
        #taken from https://github.com/mrdoob/three.js/blob/34dc2478c684066257e4e39351731a93c6107ef5/src/math/Euler.js
        from math import atan2, asin, pow, sqrt, pi
        rotM = rotM.rotM
        #xyz
        roll = pitch = yaw = 0.0

        m11 = rotM[0, 0]
        m12 = rotM[0, 1]
        m13 = rotM[0, 2]

        m21 = rotM[1, 0]
        m22 = rotM[1, 1]
        m23 = rotM[1, 2]

        m31 = rotM[2, 0]
        m32 = rotM[2, 1]
        m33 = rotM[2, 2]

        if order is "xyz":
            pitch = asin(max(min(m13, 1), -1))

            if abs(m13) < 0.99999:
                roll = atan2(-m23, m33)
                yaw = atan2(-m12, m11)
            else:
                roll = atan2(m32, m22)
                yaw = 0

        elif order is "zyx":
            pitch = asin(max(min(m12, 1), -1))
            if abs(m13) < 0.99999:
                roll = atan2(m32, m22)
                yaw = atan2(-m13, m11)
            else:
                roll = atan2(- m23, m33)
                yaw = 0

        return roll, pitch, yaw

    @staticmethod
    def RPYFromQuat(quat: 'Quaternion', order="xyz") -> Tuple[float, float, float]:
        raise Exception("Do not support this order yet!")
