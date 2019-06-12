from typing import List, Dict, Tuple, Optional
from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
import math
import numpy as np

class RotationMatrix(BaseType):

    rotM: np.matrix = None

    order: str = None

    def __init__(self, rotM: np.matrix, order: str="xyz"):
        self.rotM = rotM
        self.order = order

        super().__init__()

    @classmethod
    def fromList(cls, list: List[float]):
        if list.__len__() != 9:
            raise Exception("List size is not correct! Should be 9 instead of : " + str(list.__len__()))

        return cls(np.matrix(list))

    @classmethod
    def fromQuat(cls, quat: 'Quaternion', order="xyz"):
        rotM = cls.RotMFromQuat(quat)
        return cls(rotM, order=order)

    @classmethod
    def fromEulerAngles(cls, eulerAng: 'EulerAngles', order="xyz"):
        rotM = cls.RotMFromEulerAngles(eulerAng, order)
        return cls(rotM, order=order)

    @classmethod
    def fromDict(cls, dict: Dict):
        raise Exception("Not Implemented!")

    @staticmethod
    def RotMFromQuat(quat: 'Quaternion') -> 'RotationMatrix':
        w = quat.w
        x = quat.x
        y = quat.y
        z = quat.z

        row0 = [w*w + x*x - y*y - z*z, 2*(x*y - w*z), 2*(z*x + w*y)]
        row1 = [2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)]
        row2 = [2*(z*x - w*y), 2*(y*z + w*x), w*w - x*x - y*y + z*z]

        rotM = []
        rotM.append(row0)
        rotM.append(row1)
        rotM.append(row2)

        return np.matrix(rotM)

    @staticmethod
    def RotMFromEulerAngles(eulerAng: 'EulerAngles', order: str="xyz") -> np.matrix:
        from math import cos, sin
        args = eulerAng.toList()
        Rs = []
        orderList = list(order)
        for i in range(0, orderList.__len__()):
            if order[i] is 'x':
                Rs.append(RotationMatrix.rotXAxis(args[0]))
            if order[i] is 'y':
                Rs.append(RotationMatrix.rotYAxis(args[1]))
            if order[i] is 'z':
                Rs.append(RotationMatrix.rotZAxis(args[2]))

        M = Rs[0] * Rs[1] * Rs[2]

        return M

    def toString(self) -> str:
        return str(self.rotM)

    def toList(self) -> List[float]:
        return self.rotM.tolist()

    def get4x4(self) -> np.matrix:
        tempMatrix = np.zeros((4, 4))
        tempMatrix[0:3, 0:3] = self.rotM
        tempMatrix[3, 3] = 1.0
        M = tempMatrix
        return np.matrix(M)

    @staticmethod
    def rotZAxis(angle: float):
        Rz = np.matrix([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0.0, 0.0, 1.0]
        ])
        return Rz

    @staticmethod
    def rotXAxis(angle: float) -> np.matrix:
        Rx = np.matrix([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)]
        ])
        return Rx

    @staticmethod
    def rotYAxis(angle: float) -> np.matrix:
        Ry = np.matrix([
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)]
        ])
        return Ry
