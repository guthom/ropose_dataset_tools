from guthoms_helpers.base_types.Pose3D import Pose3D
from guthoms_helpers.base_types.Pose2D import Pose2D
import numpy as np
from ropose_dataset_tools.DataClasses.Dataset.SensorBase import SensorBase
from ropose_dataset_tools.DataClasses.Dataset.CameraInfo import CameraInfo
from ropose_dataset_tools.DataClasses.Dataset.FrameTypes import FrameTypes
from guthoms_helpers.filesystem.FileHelper import FileHelper

from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from typing import List, Tuple, Optional


class Image(SensorBase):
    def __init__(self, filePath: str, cameraInfo: CameraInfo=None, sensorPose: Pose3D = None,
                 transforms: List[type(Pose3D)] = None):
        super().__init__(FrameTypes.Image, sensorPose=sensorPose)

        self.filePath: str = filePath
        self.cameraInfo: CameraInfo = cameraInfo
        self.resizedReprojectedPoints: List[Pose2D] = []
        self.resizedReprojectedGT: List[Pose2D] = []
        self.transforms: List[Pose3D] = transforms
        self.usedPadding: Optional[Tuple] = None
        self.imageSize = FileHelper.GetImageSize(filePath)

        self.boundingBox: Optional[BoundingBox] = None
        self.resizedBoundingBox: Optional[BoundingBox] = None
        self.projectedJoints: List[Pose2D] = []

        if transforms is not None:
            self.SetProjections(transforms)
            self.SetBoundingBox()


    def SetBoundingBox(self):
        self.boundingBox = BoundingBox.CreateBoundingBox(self.projectedJoints, expandBox=True, expandRatio=0.2)
        self.boundingBox = self.boundingBox.ClipToShape((self.cameraInfo.width, self.cameraInfo.height))

    def SetProjections(self, transforms: List[Pose3D]):
        self.projectedJoints = []

        for trans in transforms:
            point3D = trans.trans.toNp4()

            K = self.cameraInfo.K
            ox = K[0, 2]
            oy = K[1, 2]

            focalx = K[0, 0]
            focaly = K[1, 1]

            u = focalx * point3D[0] / point3D[2] + ox
            v = focaly * point3D[1] / point3D[2] + oy

            visible = True
            if v > self.cameraInfo.height or u > self.cameraInfo.width:
                u[0] = v[0] = -1
                visible = False

            self.projectedJoints.append(Pose2D.fromData(x=u[0], y=v[0], angle=0, visible=visible))
            self.resizedReprojectedPoints.append(self.projectedJoints[-1])


