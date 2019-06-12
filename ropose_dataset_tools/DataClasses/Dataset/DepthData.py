from ropose_dataset_tools.DataClasses.Dataset.SensorBase import SensorBase
from ropose_dataset_tools.DataClasses.Dataset.FrameTypes import FrameTypes
from ropose_dataset_tools.DataClasses.BaseTypes.Pose import Pose

class DepthData(SensorBase):

    def __init__(self, sensorPose: type(Pose)):
        super().__init__(FrameTypes.Depth, sensorPose=sensorPose)
