from abc import ABC
from ropose_dataset_tools.DataClasses.BaseTypes.Pose import Pose
from ropose_dataset_tools.DataClasses.Dataset.FrameTypes import FrameTypes

class SensorBase(ABC):

    frameType: FrameTypes = None
    sensorPose: Pose = None

    def __init__(self, frameType: FrameTypes, sensorPose: type(Pose)):
        self.frameType = frameType
        self.sensorPose = sensorPose
