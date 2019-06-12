from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
from ropose_dataset_tools.DataClasses.BaseTypes.Vector3D import Vector3D
from ropose_dataset_tools.DataClasses.BaseTypes.Quaternion import Quaternion
from ropose_dataset_tools.DataClasses.BaseTypes.RotationMatrix import RotationMatrix
from ropose_dataset_tools.DataClasses.BaseTypes.EulerAngles import EulerAngles
from ropose_dataset_tools.DataClasses.BaseTypes.AxisAngles import AxisAngles
import numpy as np
from typing import List, Dict, Tuple

class Rotation(BaseType):

    quat: Quaternion = None
    euler: EulerAngles = None
    rotM: RotationMatrix = None
    axiAngles: AxisAngles = None

    order: str = None

    def __init__(self, order: str = 'xyz'):
        super().__init__()

        if order not in ["xyz"]:
            raise Exception("This order is not supported yet!")

        self.order = order

    def Distance(self, other: 'Rotation') -> 'Rotation':
        #find rotation difference by finding the resulting quaternion
        diffQuat = self.quat * other.quat.Inverse()

        return Rotation.fromQuat(diffQuat)


    @classmethod
    def fromList(cls, newList: List[float], order="xyz"):
        raise Exception("Not implemented yet!")
        ret = Rotation(order=order)
        if list.__len__() is 4:
            ret.setxyzw(newList[0], newList[1], newList[2], newList[3])
            return ret
        else:
            raise Exception("List shape is not correct!")

    @classmethod
    def fromQuat(cls, quat: Quaternion, order: str = 'xyz'):
        ret = Rotation(order=order)
        ret.setQuat(quat)
        return ret

    @classmethod
    def fromEuler(cls, eulerAng: EulerAngles):
        ret = Rotation(order=eulerAng.order)
        ret.setEuler(eulerAng)
        return ret

    @classmethod
    def fromAxisAngles(cls, axisAng: AxisAngles, order: str = 'xyz'):
        ret = Rotation(order=order)
        ret.setAxisAngles(axisAng)
        return ret

    @classmethod
    def fromRotationMatrix(cls, rotM: RotationMatrix):
        ret = Rotation(order=rotM.order)
        ret.setRotationMatrix(rotM)
        return ret

    @classmethod
    def fromAxisAngles(cls, axisAng: AxisAngles, order: str = 'xyz'):
        ret = Rotation(order=order)
        ret .setAxisAngles(axisAng)
        return ret

    @classmethod
    def fromDict(cls, dict: Dict, order: str = 'xyz'):
        ret = Rotation(order=order)

        if "x" and "y" and "z" and "w" in dict:
            ret.setQuat(Quaternion(float(dict["x"]), float(dict["y"]), float(dict["z"]), float(dict["w"])))
            return ret

        if "roll" and "pitch" and "yaw" in dict:
            ret.setEuler(EulerAngles(float(dict["roll"]), float(dict["pitch"]), float(dict["yaw"])))
            return ret

        if "rotM" in dict:
            ret.setRotationMatrix(RotationMatrix(dict["rotM"]))
            return ret

    def setAxisAngles(self, axisAng: AxisAngles):
        self.axiAngles = axisAng
        self.quat = Quaternion.fromAxisAngle(axisAng)
        self.rotM = RotationMatrix.fromQuat(self.quat, order=self.order)
        self.euler = EulerAngles.fromRotM(self.rotM, order=self.order)
        pass

    def setQuat(self, quat: Quaternion):
        self.quat = quat
        self.rotM = RotationMatrix.fromQuat(self.quat, order=self.order)
        self.euler = EulerAngles.fromRotM(self.rotM, order=self.order)
        self.axiAngles = AxisAngles.fromQuat(self.quat)

    def setEuler(self, euler: EulerAngles):
        self.euler = euler
        self.rotM = RotationMatrix.fromEulerAngles(euler, order=self.order)
        self.quat = Quaternion.fromEuler(self.euler, order=self.order)
        self.axiAngles = AxisAngles.fromQuat(self.quat)

    def setRotationMatrix(self, rotM: RotationMatrix):
        self.rotM = rotM
        self.euler = EulerAngles.fromRotM(self.rotM, order=self.order)
        self.quat = Quaternion.fromRotM(self.rotM)
        self.axiAngles = AxisAngles.fromQuat(self.quat)

    def toString(self) -> str:
        raise Exception("Not Implemented!")

    def toList(self) -> List[float]:
        raise Exception("Not Implemented!")

    def distance(self, rotation: 'Rotation') -> float:
        raise Exception("Not Implemented!")
