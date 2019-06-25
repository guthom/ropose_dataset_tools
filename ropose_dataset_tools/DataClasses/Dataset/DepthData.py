from ropose_dataset_tools.DataClasses.Dataset.SensorBase import SensorBase
from ropose_dataset_tools.DataClasses.Dataset.FrameTypes import FrameTypes
from guthoms_helpers.base_types.Pose3D import Pose3D

class DepthData(SensorBase):

    def __init__(self, sensorPose: type(Pose3D)):
        super().__init__(FrameTypes.Depth, sensorPose=sensorPose)
