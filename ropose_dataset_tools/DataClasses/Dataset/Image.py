from ropose_dataset_tools.DataClasses.Dataset.SensorBase import SensorBase
from ropose_dataset_tools.DataClasses.BaseTypes.Pose import Pose
from ropose_dataset_tools.DataClasses.Dataset.CameraInfo import CameraInfo
from ropose_dataset_tools.DataClasses.Dataset.FrameTypes import FrameTypes
from ropose_dataset_tools.DataClasses.Dataset.BoundingBox import BoundingBox
from typing import List, Tuple


class Image(SensorBase):

    filePath: str = None
    cameraInfo: CameraInfo = None
    projectedJoints: List[List[float]] = None
    resizedReprojectedPoints: List[List[float]] = None
    resizedReprojectedGT: List[List[float]] = None
    transforms: List[Pose] = None
    usedPadding: Tuple = None

    boundingBox: BoundingBox = None
    resizedBoundingBox: BoundingBox = None

    def __init__(self, filePath: str, cameraInfo: CameraInfo=None, sensorPose: Pose = None,
                 transforms: List[type(Pose)] = None):
        super().__init__(FrameTypes.Image, sensorPose=sensorPose)

        self.filePath = filePath
        self.cameraInfo = cameraInfo
        self.resizedReprojectedPoints = []
        self.resizedReprojectedGT = []

        self.transforms = transforms
        if transforms is not None:
            self.SetProjections(transforms)
            self.SetBoundingBox()

    def SetBoundingBox(self):
        self.boundingBox = BoundingBox.CreateBoundingBox(self.projectedJoints)

    def SetProjections(self, transforms: List[Pose]):
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

            self.projectedJoints.append([u[0], v[0]])
            self.resizedReprojectedPoints.append([u[0], v[0]])


