from typing import List, Dict, Tuple
from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
from ropose_dataset_tools.DataClasses.BaseTypes.EulerAngles import EulerAngles
from ropose_dataset_tools.DataClasses.BaseTypes.RotationMatrix import RotationMatrix
from ropose_dataset_tools.DataClasses.BaseTypes.AxisAngles import AxisAngles
import math
import numpy as np

class Quaternion(BaseType):

    x: float = None
    y: float = None
    z: float = None
    w: float = None

    def __init__(self, x: float, y: float, z: float, w: float):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        super().__init__()


    def __mul__(self, other: 'Quaternion'):
        #from http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/code/index.htm

        x = self.x*other.w + self.w*other.x + self.y*other.z - self.z*other.y
        y = self.y*other.w + self.w*other.y + self.z*other.x - self.x*other.z
        z = self.z*other.w + self.w*other.z + self.x*other.y - self.y*other.x
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z

        return Quaternion(x=x, y=y, z=z, w=w).Normalized()

    def GetXYZW(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.z, self.w

    @classmethod
    def fromList(cls, list: List[float]):
        if list.__len__() != 4:
            raise Exception("List size is not correct! Should be 4 instead of : " + str(list.__len__()))
        return cls(list[0], list[1], list[2], list[3])

    @classmethod
    def fromDict(cls, dict: Dict):
        return cls(float(dict["x"]), float(dict["y"]), float(dict["z"]), float(dict["w"]))

    @classmethod
    def fromRotM(cls, rotM: 'RotationMatrix'):
        x, y, z, w = cls.QuatFromRotM(rotM.rotM)
        return cls(x=x, y=y, z=z, w=w).Normalized()

    @classmethod
    def fromEuler(cls, eulerAngles: EulerAngles, order='xyz'):
        x, y, z, w = cls.QuatFromEuler(eulerAngles, order=order)
        return cls(x=x, y=y, z=z, w=w).Normalized()

    @classmethod
    def fromAxisAngle(cls, axisAngle: AxisAngles):
        x, y, z, w = cls.QuatFromAxisAngle(axisAngle)
        return cls(x=x, y=y, z=z, w=w).Normalized()

    def toString(self) -> str:
        return "[" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ", " + + str(self.w) + "]"

    def toList(self) -> List[float]:
        return [self.x, self.y, self.z, self.w]

    def Normalized(self) -> 'Quaternion':
        d = math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2) + math.pow(self.z, 2) + math.pow(self.w, 2))
        x = self.x / d
        y = self.y / d
        z = self.z / d
        w = self.w / d
        return Quaternion(x, y, z, w)

    @staticmethod
    def QuatFromRotM(rotM: np.matrix, order="xyz") -> Tuple[float, float, float, float]:
        #http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        #https://github.com/mrdoob/three.js/blob/dev/src/math/Quaternion.js
        from math import sqrt

        m11 = rotM[0, 0]
        m12 = rotM[0, 1]
        m13 = rotM[0, 2]
        m21 = rotM[1, 0]
        m22 = rotM[1, 1]
        m23 = rotM[1, 2]
        m31 = rotM[2, 0]
        m32 = rotM[2, 1]
        m33 = rotM[2, 2]

        trace = m11 + m22 + m33

        if trace > 0:
            s = 0.5 / sqrt(trace + 1.0)
            x = (m32 - m23) * s
            y = (m13 - m31) * s
            z = (m21 - m12) * s
            w = 0.25 / s
        elif m11 > m22 and m11 > m33:
            s = 2.0 * sqrt(1.0 + m11 - m22 - m33)
            x = 0.25 * s
            y = (m12 + m21) / s
            z = (m13 + m31) / s
            w = (m32 - m23) / s
        elif m22 > m33:
            s = 2.0 * sqrt(1.0 + m22 - m11 - m33)
            x = (m12 + m21) / s
            y = 0.25 * s
            z = (m23 + m32) / s
            w = (m13 - m31) / s
        else:
            s = 2.0 * sqrt(1.0 + m33 - m11 - m22)
            x = (m13 + m31) / s
            y = (m23 + m32) / s
            z = 0.25 * s
            w = (m21 - m12) / s

        return x, y, z, w


    @staticmethod
    def QuatFromAxisAngle(axisAngles: AxisAngles) -> Tuple[float, float, float, float]:
        from math import sin, cos
        # // http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        halfAngle = axisAngles.angle / 2
        s = sin(halfAngle)

        x = axisAngles.axis.x * s
        y = axisAngles.axis.y * s
        z = axisAngles.axis.z * s
        w = cos(halfAngle)

        return x, y, z, w


    @staticmethod
    def QuatFromEuler(eulerAngles: EulerAngles, order="xyz")-> Tuple[float, float, float, float]:

        #https://www.mathworks.com/matlabcentral/fileexchange/
        #   20696-function-to-convert-between-dcm-euler-angles-quaternions-and-euler-vectors

        c1 = math.cos(eulerAngles.roll / 2)
        s1 = math.sin(eulerAngles.roll / 2)
        c2 = math.cos(eulerAngles.pitch / 2)
        s2 = math.sin(eulerAngles.pitch / 2)
        c3 = math.cos(eulerAngles.yaw / 2)
        s3 = math.sin(eulerAngles.yaw / 2)

        x = y = z = 0
        w = 1.0

        if order is 'xyz':
            x = s1 * c2 * c3 + c1 * s2 * s3
            y = c1 * s2 * c3 - s1 * c2 * s3
            z = c1 * c2 * s3 + s1 * s2 * c3
            w = c1 * c2 * c3 - s1 * s2 * s3

        elif order is 'yxz':
            x = s1 * c2 * c3 + c1 * s2 * s3
            y = c1 * s2 * c3 - s1 * c2 * s3
            z = c1 * c2 * s3 - s1 * s2 * c3
            w = c1 * c2 * c3 + s1 * s2 * s3

        elif order is 'zxy':
            x = s1 * c2 * c3 - c1 * s2 * s3
            y = c1 * s2 * c3 + s1 * c2 * s3
            z = c1 * c2 * s3 + s1 * s2 * c3
            w = c1 * c2 * c3 - s1 * s2 * s3

        elif order is 'zyx':
            x = s1 * c2 * c3 - c1 * s2 * s3
            y = c1 * s2 * c3 + s1 * c2 * s3
            z = c1 * c2 * s3 - s1 * s2 * c3
            w = c1 * c2 * c3 + s1 * s2 * s3

        elif order is 'yzx':
            x = s1 * c2 * c3 + c1 * s2 * s3
            y = c1 * s2 * c3 + s1 * c2 * s3
            z = c1 * c2 * s3 - s1 * s2 * c3
            w = c1 * c2 * c3 - s1 * s2 * s3

        elif order is 'xzy':
            x = s1 * c2 * c3 - c1 * s2 * s3
            y = c1 * s2 * c3 - s1 * c2 * s3
            z = c1 * c2 * s3 + s1 * s2 * c3
            w = c1 * c2 * c3 + s1 * s2 * s3

        return x, y, z, w

    def Inverse(self):
        return Quaternion(x=-self.x, y=-self.y, z=-self.z, w=self.w).Normalized()

