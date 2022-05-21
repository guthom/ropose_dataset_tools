from abc import ABC
from guthoms_helpers.base_types.Pose3D import Pose3D
from ropose_dataset_tools.DataClasses.Dataset.FrameTypes import FrameTypes

class SensorBase(ABC):


    def __init__(self, frameType: FrameTypes, sensorPose: type(Pose3D)):
        self.frameType: FrameTypes = frameType
        self.sensorPose: Pose3D = sensorPose
