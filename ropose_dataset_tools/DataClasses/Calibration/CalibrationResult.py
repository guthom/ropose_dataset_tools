from ropose_dataset_tools.DataClasses.BaseTypes.BaseType import BaseType
from ropose_dataset_tools.DataClasses.BaseTypes.Pose import Pose
from typing import List, Dict
import numpy as np

class CalibrationResult(BaseType):

    def __init__(self, pose: Pose = None, reprojectionError: float = None):
        self.pose = pose
        self.reprojectionError = reprojectionError

    def toList(self) -> List[float]:
        return [self.pose.toList(), self.reprojectionError]

    def toString(self) -> str:
        return str(self.toList())

    def toNp(self) -> np.array:
        return np.array(self.toList())

    @classmethod
    def fromList(cls, list: List[List[float]]):
        pose = Pose.fromList(list[0])
        reproError = list[1][0]
        return cls(pose, reproError)

    @classmethod
    def fromDict(cls, dict: Dict):
        raise Exception("Not Implemented!")