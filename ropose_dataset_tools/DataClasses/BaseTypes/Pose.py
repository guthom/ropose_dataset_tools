from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
from ropose_dataset_tools.DataClasses.BaseTypes.Vector3D import Vector3D
from ropose_dataset_tools.DataClasses.BaseTypes.Rotation import Rotation
from ropose_dataset_tools.DataClasses.BaseTypes.Quaternion import Quaternion

from typing import List, Dict, Tuple

import numpy as np

class Pose(BaseType):

    trans: Vector3D = None
    rotation:  Rotation = None

    def __init__(self, trans: Vector3D, quat: Rotation):
        self.trans = trans
        self.rotation = quat

        super().__init__()

    @classmethod
    def fromDict(cls, data: Dict[str, Dict]):
        trans = Vector3D.fromDict(data["translation"])
        quat = Rotation.fromDict(data["rotation"])
        return cls(trans, quat)

    @classmethod
    def fromList(cls, list: List[List[float]]):
        trans = Vector3D.fromList(list[0])
        quat = Rotation.fromQuat(Quaternion.fromList(list[1]))
        return cls(trans=trans, quat=quat)

    def toOpenCvTvecRvec(self) -> Tuple[np.array, np.array]:
        raise Exception("Not Implemented!")
        rvec = None
        tvec = None
        return rvec, tvec

    def toString(self) -> str:
        return "[" + self.trans.toString() + self.rotation.toString() + "]"

    def toList(self) -> List[List[float]]:
        return [self.trans.toList(), self.rotation.quat.toList()]

    def to4x4(self) -> np.matrix:
        R = self.rotation.rotM.rotM
        t = self.trans.toNp()

        M = np.matrix(R)
        M = np.concatenate((M, t), axis=1)
        dummy = np.transpose(np.array([[0.0], [0.0], [0.0], [1.0]]))
        M = np.concatenate((M, dummy), axis=0)

        return M

