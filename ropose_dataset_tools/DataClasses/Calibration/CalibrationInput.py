from DataClasses.BaseTypes.BaseType import BaseType
from DataClasses.BaseTypes.Pose import Pose
from DataClasses.BaseTypes.Vector2D import Vector2D
from typing import List, Dict
import numpy as np

class CalibrationInput(object):

    def __init__(self, positions2D: List[Vector2D], poses3D: List[Pose], intMat: np.matrix = None,
                 extrinsicGuess: Pose = None):
        self.positions2D = positions2D
        self.poses3D = poses3D
        self.extrinsicGuess = extrinsicGuess
        self.intMat = intMat

    def __add__(self, other: 'CalibrationInput'):
        self.positions2D.extend(other.positions2D)
        self.poses3D.extend(other.poses3D)
        return self

    @classmethod
    def fromDict(cls, data: Dict):
        # data["pred_2d"]
        # data["gt_2d"]
        # data["gt_3d"]
        # data["sensor_pose"] -> gt sensor pose
        # data["int_mat"]

        pos2d = []

        for pose in data["pred_2d"]:
            pos2d.append(Vector2D.fromList(pose))

        poses = []
        for pose in data["gt_3d"]:
            poses.append(Pose.fromList(pose))

        intMat = data["int_mat"]

        if "extrinsic_guess" in data:
            extrinsicGuess = Pose.fromList(data["extrinsic_guess"])
        else:
            extrinsicGuess = None

        return cls(positions2D=pos2d, poses3D=poses, intMat=intMat, extrinsicGuess=extrinsicGuess)

    @classmethod
    def fromDictTestWithGT(cls, data: Dict):
        # data["pred_2d"]
        # data["gt_2d"]
        # data["gt_3d"]
        # data["sensor_pose"] -> gt sensor pose
        # data["int_mat"]

        pos2d = []

        for pose in data["gt_2d"]:
            pos2d.append(Vector2D.fromList(pose))

        poses = []
        for pose in data["gt_3d"]:
            poses.append(Pose.fromList(pose))

        intMat = data["int_mat"]

        if "extrinsic_guess" in data:
            extrinsicGuess = Pose.fromList(data["extrinsic_guess"])
        else:
            extrinsicGuess = None

        return cls(positions2D=pos2d, poses3D=poses, intMat=intMat, extrinsicGuess=extrinsicGuess)

    def Get2DTranslations(self) -> List[List[float]]:
        ret = []

        for pose in self.positions2D:
            ret.append(pose.toList())

        return ret

    def Get3DTranslations(self) -> List[List[float]]:
        ret = []

        for pose in self.poses3D:
            ret.append(pose.trans.toList())

        return ret


    def Get3DQuaternions(self) -> List[List[float]]:
        ret = []

        for pose in self.poses3D:
            ret.append(pose.rotation.toList())

        return ret
